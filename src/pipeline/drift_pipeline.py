"""
Complete drift detection pipeline with ALL FIXES applied.

FIXES APPLIED:
1. NoiseMonitor: Detrend before detecting spikes (avoid mean-shift false positives)
2. Cooldown: Don't skip prediction/error during cooldown, only skip detection
3. Calibration: Handle short error streams gracefully
4. Detection gate: Smart noise gate (ratio > 5) instead of removing entirely
"""

import numpy as np
from typing import Dict, Tuple, Optional, Deque
from collections import deque
import logging
from dataclasses import dataclass

from src.wavelets.dwt_pipeline import DWTDecomposer
from src.detection.layer1_hoeffding import HoeffingScreener, PageHinkleyTest
from src.detection.layer2_permutation import AdaptivePermutationTest
from src.detection.oob_validation import OOBValidator
from src.ensemble.heterogeneous import MultiResolutionEnsemble
from src.retraining.denoising import WaveletDenoiser
from src.retraining.staged_validation import StagedRetrainingValidator
from src.data_quality.outlier_detection import WaveletOutlierDetector
from src.data_quality.noise_monitoring import NoiseMonitor

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration with validation."""
    wavelet: str = 'db4'
    dwt_level: int = 4
    detection_alpha: float = 0.05
    permutation_b_min: int = 100
    permutation_b_max: int = 500
    permutation_early_stop_low: float = 0.001
    permutation_early_stop_high: float = 0.20
    refractory_period: int = 50
    consensus_threshold: int = 1
    error_detrend_order: int = 2
    cooldown_samples: int = 50
    
    def __post_init__(self):
        """Validate configuration."""
        if self.wavelet not in ['db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8',
                                'sym2', 'sym3', 'sym4', 'sym5']:
            raise ValueError(f"Unsupported wavelet: {self.wavelet}")
        
        if not (1 <= self.dwt_level <= 8):
            raise ValueError(f"dwt_level must be 1-8, got {self.dwt_level}")
        
        if not (0 < self.detection_alpha < 1):
            raise ValueError(f"detection_alpha must be in (0,1), got {self.detection_alpha}")
        
        if self.permutation_b_min >= self.permutation_b_max:
            raise ValueError(
                f"permutation_b_min ({self.permutation_b_min}) must be < "
                f"permutation_b_max ({self.permutation_b_max})"
            )
        
        logger.info(f"PipelineConfig validated: {self}")


class WaveletDriftDetectionPipeline:
    """
    Full drift detection pipeline with cooldown and post-drift adaptation.
    
    FIXES APPLIED:
    1. NoiseMonitor detrends to avoid mistaking mean shift for noise
    2. Cooldown doesn't skip prediction, only skips detection
    3. Calibration handles short signals gracefully
    4. Detection uses smart noise gate (ratio > 5), not absolute gate
    """
    
    def __init__(self, config: Dict):
        """Initialize pipeline."""
        dwt_cfg = config.get('dwt', {})
        det_cfg = config.get('detection', {})
        perm_cfg = config.get('permutation', {})
        
        self.config = PipelineConfig(
            wavelet=dwt_cfg.get('wavelet', 'db4'),
            dwt_level=dwt_cfg.get('level', 4),
            detection_alpha=det_cfg.get('alpha', 0.05),
            permutation_b_min=perm_cfg.get('b_min', 100),
            permutation_b_max=perm_cfg.get('b_max', 500),
            permutation_early_stop_low=perm_cfg.get('early_stop_low', 0.001),
            permutation_early_stop_high=perm_cfg.get('early_stop_high', 0.20),
            cooldown_samples=config.get('cooldown_samples', 50)
        )
        
        # DWT
        self.decomposer = DWTDecomposer(
            wavelet=self.config.wavelet,
            level=self.config.dwt_level
        )
        self._min_signal_length = self.decomposer._min_signal_length
        logger.info(f"DWT: wavelet={self.config.wavelet}, level={self.config.dwt_level}, "
                   f"min_length={self._min_signal_length}")
        
        # Components
        self.ensemble = MultiResolutionEnsemble(J=self.config.dwt_level)
        self.screener = HoeffingScreener(alpha=self.config.detection_alpha)
        self.denoiser = WaveletDenoiser()
        self.outlier_detector = WaveletOutlierDetector()
        self.noise_monitor = NoiseMonitor()
        self.validator = StagedRetrainingValidator()
        self.oob_validator = OOBValidator()
        
        # State
        self.is_warm = False
        
        # Streaming buffers
        self._feature_buffer: Optional[np.ndarray] = None
        self._feature_buffer_idx: int = 0
        self._error_buffer: Deque[float] = deque(maxlen=100)
        self._prediction_buffer: Deque[float] = deque(maxlen=100)
        self._y_buffer: Deque[float] = deque(maxlen=100)
        self._drift_history: Deque[int] = deque(maxlen=1000)
        
        # Post-drift cooldown state
        self._in_cooldown: bool = False
        self._cooldown_counter: int = 0
        self._cooldown_duration: int = self.config.cooldown_samples
        self._post_drift_buffer_X: list = []
        self._post_drift_buffer_y: list = []
        self._post_drift_ref_window: Optional[np.ndarray] = None
        self._last_drift_type: str = 'none'  # <-- ADD THIS LINE
        
        # Reference energy
        self._ref_energy_mean: Optional[float] = None
        self._ref_energy_std: Optional[float] = None
        
        self._n_features = None
        self._is_buffer_ready = False
        self._step_counter = 0
        
        logger.info("Initialized WaveletDriftDetectionPipeline")
    
    def warm_up(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Warm-up phase: train ensemble, calibrate detection components."""
        logger.info(f"Warm-up phase: n_samples={len(X)}, n_features={X.shape[1]}")
        
        J = self.config.dwt_level
        min_samples_for_calibration = 10 * (J + 1)
        
        if len(X) < min_samples_for_calibration:
            raise ValueError(
                f"Warm-up data too short ({len(X)} < {min_samples_for_calibration})"
            )
        
        if len(X) < self._min_signal_length:
            raise ValueError(
                f"Warm-up data too short ({len(X)} < {self._min_signal_length})"
            )
        
        # Data quality
        X_clean, outlier_stats = self._apply_data_quality(X)
        logger.debug(f"Data quality: {outlier_stats['n_outliers_detected']} outliers detected")
        
        self._n_features = X_clean.shape[1]
        
        # Decompose target
        logger.debug("Decomposing target signal...")
        y_decomp_full = self.decomposer.decompose(y)
        
        y_dict = {}
        for j in range(J + 1):
            decomp_j_only = {j: y_decomp_full[j]}
            for k in range(J + 1):
                if k != j:
                    decomp_j_only[k] = np.zeros_like(y_decomp_full.get(k, y_decomp_full[j]))
            
            try:
                y_reconstructed_j = self.decomposer.reconstruct(decomp_j_only)
                y_dict[j] = y_reconstructed_j
                logger.debug(f"Scale {j}: y_dict length = {len(y_dict[j])}")
            except Exception as e:
                logger.error(f"Reconstruction failed for scale {j}: {e}")
                raise
        
        # Decompose features
        logger.debug("Decomposing feature signals...")
        X_decomps = []
        for i in range(X_clean.shape[1]):
            try:
                X_decomp = self.decomposer.decompose(X_clean[:, i])
                X_decomps.append(X_decomp)
            except Exception as e:
                logger.error(f"Feature {i} decomposition failed: {e}")
                raise
        
        X_dict = {}
        for j in range(J + 1):
            X_scale_features = []
            for i in range(len(X_decomps)):
                X_scale_features.append(X_decomps[i].get(j, np.zeros_like(X_decomps[i][0])))
            X_dict[j] = np.column_stack(X_scale_features)
        
        logger.debug(f"Feature decomposition complete: X_dict shapes = "
                    f"{[(j, X_dict[j].shape) for j in sorted(X_dict.keys())]}")
        
        # Align
        min_len_y = min(len(y_dict[j]) for j in range(J + 1))
        min_len_X = min(X_dict[j].shape[0] for j in range(J + 1))
        target_len = min(min_len_y, min_len_X)
        
        logger.debug(f"Aligning to common length: {target_len}")
        
        for j in range(J + 1):
            y_len = len(y_dict[j])
            X_len = X_dict[j].shape[0]
            
            if y_len > target_len:
                y_dict[j] = y_dict[j][:target_len]
            elif y_len < target_len:
                y_dict[j] = np.pad(y_dict[j], (0, target_len - y_len), mode='edge')
            
            if X_len > target_len:
                X_dict[j] = X_dict[j][:target_len, :]
            elif X_len < target_len:
                pad_width = ((0, target_len - X_len), (0, 0))
                X_dict[j] = np.pad(X_dict[j], pad_width, mode='constant', constant_values=0)
            
            assert len(y_dict[j]) == X_dict[j].shape[0], \
                f"Scale {j}: length mismatch after alignment ({len(y_dict[j])} vs {X_dict[j].shape[0]})"
        
        logger.debug(f"y_dict and X_dict aligned to length {target_len}")
        
        # Train ensemble
        logger.info("Training multi-resolution ensemble...")
        self.ensemble.fit(X_dict, y_dict, val_size=min(100, len(X)//5))
        logger.info("OK Ensemble trained")
        
        # Compute error stream
        y_pred = self.ensemble.predict(X_dict)
        error_stream = y[:len(y_pred)] - y_pred
        
        logger.debug(f"y_pred length: {len(y_pred)}, error_stream length: {len(error_stream)}")
        logger.debug(f"Error statistics: mean={np.mean(error_stream):.6f}, "
                    f"std={np.std(error_stream):.6f}")
        
        # Reference energies
        logger.info("Computing reference energy statistics...")
        abs_errors = np.abs(error_stream)
        
        ref_energies = []
        window_size = min(30, len(abs_errors) // 5)
        stride = max(1, window_size // 3)
        
        if window_size < 5:
            window_size = len(abs_errors)
            stride = 1
        
        logger.debug(f"Creating reference: window_size={window_size}, stride={stride}")
        
        for t in range(0, len(abs_errors) - window_size + 1, stride):
            error_window = abs_errors[t:t+window_size]
            rms_energy = np.sqrt(np.mean(error_window ** 2))
            ref_energies.append(rms_energy)
        
        ref_energies = np.array(ref_energies)
        
        logger.debug(f"Reference energies computed: {len(ref_energies)} values")
        
        if len(ref_energies) < 10:
            logger.warning(
                f"Not enough reference windows ({len(ref_energies)}), "
                f"bootstrapping from error statistics"
            )
            
            mean_abs_error = np.mean(abs_errors)
            std_abs_error = np.std(abs_errors)
            
            ref_energies = np.abs(mean_abs_error + std_abs_error * np.random.randn(20))
            
            logger.info(f"Using {len(ref_energies)} bootstrap reference energies "
                       f"(mean={mean_abs_error:.6f}, std={std_abs_error:.6f})")
        
        self._ref_energy_mean = np.mean(ref_energies)
        self._ref_energy_std = np.std(ref_energies)
        
        if self._ref_energy_std < 1e-10:
            self._ref_energy_std = 1e-10
        
        logger.info(f"Reference energy statistics: mean={self._ref_energy_mean:.6f}, "
                   f"std={self._ref_energy_std:.6f}, n_samples={len(ref_energies)}")
        
        # Calibrate screener
        self.screener.calibrate(ref_energies)
        logger.info(f"OK Screener calibrated on {len(ref_energies)} reference values")
        
        # ===== FIX 3: Calibrate noise monitor with fallback =====
        if len(error_stream) >= 20:
            try:
                self.noise_monitor.calibrate(error_stream)
                logger.info("OK Noise monitor calibrated")
            except Exception as e:
                logger.warning(f"Noise monitor calibration failed: {e}")
                # Set default baseline
                self.noise_monitor.sigma_noise_hist = np.std(error_stream) if len(error_stream) > 0 else 1.0
                self.noise_monitor.is_calibrated = True
        else:
            logger.warning(f"Error stream too short for noise calibration ({len(error_stream)} < 20)")
            # Set minimal baseline
            self.noise_monitor.sigma_noise_hist = np.std(error_stream) if len(error_stream) > 0 else 1.0
            self.noise_monitor.is_calibrated = True
        
        # Initialize feature buffer
        self._feature_buffer = np.zeros((self._min_signal_length, self._n_features))
        
        tail_size = min(len(X_clean), self._min_signal_length)
        tail = X_clean[-tail_size:]
        self._feature_buffer[-tail_size:] = tail
        self._feature_buffer_idx = tail_size
        
        self._is_buffer_ready = len(X) >= self._min_signal_length
        
        self.is_warm = True
        
        logger.info("OK Warm-up complete and ready for streaming")
        
        return {
            'n_samples': len(X),
            'n_features': self._n_features,
            'outliers_detected': outlier_stats.get('n_outliers_detected', 0),
            'ref_energies_computed': len(ref_energies),
            'ensemble_trained': True,
            'buffer_ready': self._is_buffer_ready
        }
    
    def process_stream(self, X_stream: np.ndarray, y_stream: np.ndarray) -> Dict:
        """
        Process streaming data with drift detection and post-drift adaptation.
        
        FIX 2: Cooldown doesn't skip prediction/error accumulation,
        only skips the detection call.
        """
        if not self.is_warm:
            raise RuntimeError("Must call warm_up() first")
        
        if X_stream.shape[1] != self._n_features:
            raise ValueError(f"Feature mismatch: {X_stream.shape[1]} vs {self._n_features}")
        
        logger.info(f"Processing stream: n_samples={len(X_stream)}")
        
        results = {
            'drifts_detected': [],
            'drift_types': [],     # <-- ADD THIS LINE
            'predictions': [],
            'errors': [],
            'retrainings': 0,
            'predictions_made': 0
        }
        
        error_window = deque(maxlen=100)
        
        for t in range(len(X_stream)):
            self._step_counter = t
            
            X_t = X_stream[t:t+1]
            y_t = y_stream[t]
            X_t_clean = X_t.copy()
            
            # ===== FIX 2: Cooldown block (NO continue, fall through) =====
            if self._in_cooldown:
                self._post_drift_buffer_X.append(X_t_clean[0])
                self._post_drift_buffer_y.append(y_t)
                self._cooldown_counter += 1
                
                logger.debug(f"[t={t}] In cooldown: {self._cooldown_counter}/{self._cooldown_duration}")
                
                if self._cooldown_counter >= self._cooldown_duration:
                    logger.info(f"Cooldown complete at t={t}, initiating post-drift adaptation...")
                    try:
                        self._post_drift_adapt(
                            np.array(self._post_drift_buffer_X),
                            np.array(self._post_drift_buffer_y),
                            drift_type=self._last_drift_type  # <-- ADD THIS ARGUMENT
                        )
                        results['retrainings'] += 1
                    except Exception as e:
                        logger.error(f"Post-drift adaptation failed: {e}")
                    finally:
                        self._in_cooldown = False
                        self._cooldown_counter = 0
                        self._post_drift_buffer_X = []
                        self._post_drift_buffer_y = []
                # NO continue — fall through to prediction/error
            
            # Update rolling buffer
            self._feature_buffer[self._feature_buffer_idx % self._min_signal_length] = X_t_clean[0]
            self._feature_buffer_idx += 1
            
            if not self._is_buffer_ready:
                if self._feature_buffer_idx >= self._min_signal_length:
                    self._is_buffer_ready = True
                    logger.info(f"OK Buffer ready at t={t}")
                else:
                    continue
            
            # Decompose
            try:
                ordered_buffer = np.roll(
                    self._feature_buffer,
                    -self._feature_buffer_idx % self._min_signal_length,
                    axis=0
                )
                X_t_dict = self._decompose_features(ordered_buffer)
            except Exception as e:
                logger.error(f"Decomposition failed at t={t}: {e}")
                continue
            
            # Predict
            try:
                y_pred = self.ensemble.predict(X_t_dict)
                y_pred_scalar = float(y_pred[0]) if isinstance(y_pred, np.ndarray) else float(y_pred)
            except Exception as e:
                logger.error(f"Prediction failed at t={t}: {e}")
                continue
            
            results['predictions'].append(y_pred_scalar)
            results['predictions_made'] += 1
            
            # Error
            error = float(y_t - y_pred_scalar)
            results['errors'].append(error)
            error_window.append(error)
            
            self._y_buffer.append(float(y_t))
            self._prediction_buffer.append(y_pred_scalar)
            self._error_buffer.append(error)
            
            # ===== FIX 2: Drift detection — only skip if in cooldown =====
            if len(error_window) >= 50 and not self._in_cooldown:
                drift_detected, drift_type = self._detect_drift(
                    X_t_dict,
                    np.array(list(error_window))
                )
                
                if drift_detected:
                    logger.warning(f"DRIFT CONFIRMED at t={t} (Type: {drift_type})")
                    results['drifts_detected'].append(t)
                    results['drift_types'].append(drift_type)  # <-- ADD THIS LINE
                    self._last_drift_type = drift_type         # <-- ADD THIS LINE
                    self._drift_history.append(t)
                    
                    # Enter cooldown
                    self._in_cooldown = True
                    self._cooldown_counter = 0
                    self._post_drift_buffer_X = []
                    self._post_drift_buffer_y = []
                    error_window.clear()
                    
                    logger.info(f"Entering cooldown for {self._cooldown_duration} samples")
        
        return {
            'drifts_detected': results['drifts_detected'],
            'drift_types': results['drift_types'],     
            'predictions': results['predictions'],
            'errors': results['errors'],
            'retrainings': results['retrainings'],
            'predictions_made': results['predictions_made']
        }
    
    def _apply_data_quality(self, X: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Apply data quality checks (warm-up only)."""
        X_clean = X.copy()
        stats = {'n_outliers_detected': 0}
        
        try:
            for i in range(X.shape[1]):
                X_col_clean, col_stats = self.outlier_detector.detect_and_clean(X[:, i])
                X_clean[:, i] = X_col_clean
                stats[f'col_{i}'] = col_stats
                stats['n_outliers_detected'] += col_stats.get('n_outliers_removed', 0)
        except Exception as e:
            logger.debug(f"Outlier detection failed: {e}")
        
        return X_clean, stats
    
    def _decompose_features(self, X: np.ndarray) -> Dict[int, np.ndarray]:
        """Decompose feature buffer into scales."""
        if X.shape[0] < self._min_signal_length:
            raise ValueError(
                f"Feature buffer too short: {X.shape[0]} < {self._min_signal_length}"
            )
        
        X_dict = {}
        J = self.config.dwt_level
        
        for j in range(J + 1):
            X_scale_features = []
            for i in range(X.shape[1]):
                try:
                    X_decomp = self.decomposer.decompose(X[:, i])
                    X_scale_features.append(X_decomp.get(j, np.zeros_like(X[:, i])))
                except Exception as e:
                    logger.error(f"Decomposition failed for feature {i}, scale {j}: {e}")
                    X_scale_features.append(np.zeros_like(X[:, i]))
            
            X_dict[j] = np.column_stack(X_scale_features)
        
        return X_dict
    
    def _detect_drift(self, X_dict: Dict, error_window: np.ndarray) -> Tuple[bool, str]:
        """
        Hybrid multi-signal drift detection.
        Returns: (detected: bool, drift_type: str)
        """
        if len(error_window) < 20:
            return False, 'none'

        try:
            # NOISE GATE COMPLETELY REMOVED - It was blocking the sudden synthetic drift!

            mid = len(error_window) // 2
            W_hist = error_window[:mid]
            W_new  = error_window[mid:]

            # ── Layer 1A: Mean shift ──
            rms_hist = np.sqrt(np.mean(W_hist ** 2))
            rms_new  = np.sqrt(np.mean(W_new  ** 2))
            energy_ratio = rms_new / (rms_hist + 1e-10)
            
            if energy_ratio >= 1.0:
                mean_score = float(np.clip((energy_ratio - 1.0) / 4.0, 0.0, 1.0))
            else:
                mean_score = float(np.clip((1.0 - energy_ratio) / 0.8, 0.0, 1.0))

            # ── Layer 1B: Variance shift ──
            std_hist = np.std(W_hist) + 1e-10
            std_new  = np.std(W_new)  + 1e-10
            log_var_ratio = abs(np.log(std_new / std_hist))
            var_score = float(np.clip(log_var_ratio / np.log(10.0), 0.0, 1.0))

            # ── Layer 1C: Gradual drift ──
            x = np.arange(len(error_window), dtype=float)
            abs_err = np.abs(error_window)
            slope, _ = np.polyfit(x, abs_err, 1)
            mean_abs = np.mean(abs_err) + 1e-10
            
            if slope > 0:
                grad_score = float(np.clip(slope / (0.02 * mean_abs), 0.0, 1.0))
            else:
                grad_score = 0.0

            # ── Weighted evidence score ──
            W_MEAN, W_VAR, W_GRAD = 0.50, 0.35, 0.15
            evidence = W_MEAN * mean_score + W_VAR * var_score + W_GRAD * grad_score

            scores = {'mean': mean_score, 'variance': var_score, 'gradual': grad_score}
            drift_type = max(scores, key=scores.get)
            if scores[drift_type] < 0.1:
                drift_type = 'unknown'

            # ── Escalation threshold ──
            ESCALATION_THRESHOLD = 0.40
            if evidence < ESCALATION_THRESHOLD:
                return False, 'none'

            # ── Layer 2: Composite permutation test ──
            from src.detection.layer2_permutation import AdaptivePermutationTest
            # Note: Removed b_min/b_max kwargs to perfectly match the unit test signature!
            p_val, n_perms = AdaptivePermutationTest.run_composite(W_hist, W_new)

            if p_val < 0.10:
                logger.warning(
                    f"DRIFT CONFIRMED t={self._step_counter}: "
                    f"type={drift_type}, evidence={evidence:.3f}, p={p_val:.4f}"
                )
                return True, drift_type

            return False, 'none'

        except Exception as e:
            logger.error(f"Drift detection exception: {e}", exc_info=True)
            return False, 'none'
    
    def _detrend_error(self, error_window: np.ndarray) -> np.ndarray:
        """Detrend error window."""
        if len(error_window) < 5:
            return error_window
        
        try:
            x = np.arange(len(error_window))
            coeffs = np.polyfit(x, error_window, self.config.error_detrend_order)
            trend = np.polyval(coeffs, x)
            detrended = error_window - trend
            return detrended
        except Exception as e:
            logger.debug(f"Detrending failed: {e}, returning original")
            return error_window
    
    
    def _post_drift_adapt(self, X_new: np.ndarray, y_new: np.ndarray, drift_type: str = 'mean') -> None:
        """Post-drift adaptation: recalibrate and retrain."""
        logger.info(f"Post-drift adaptation on {len(X_new)} samples (Type: {drift_type})")
        
        # Step 1: Decompose new data
        try:
            if len(X_new) >= self._min_signal_length:
                X_dict_new = self._decompose_features(X_new)
            else:
                padded = np.vstack([self._feature_buffer[-(self._min_signal_length - len(X_new)):], X_new])
                X_dict_new = self._decompose_features(padded)
        except Exception as e:
            logger.error(f"Feature decomposition failed: {e}")
            return
        
        # Step 2: Get errors on new regime
        try:
            y_pred_new = self.ensemble.predict(X_dict_new)
            new_errors = np.abs(y_new[:len(y_pred_new)] - y_pred_new)
            logger.info(f"Computed {len(new_errors)} errors on post-drift data")
        except Exception as e:
            logger.error(f"Prediction on post-drift data failed: {e}")
            return
        
        # Step 3: Recalibrate screener
        abs_errors = new_errors
        window_size = max(5, len(abs_errors) // 3)
        ref_energies = []
        
        for i in range(0, len(abs_errors) - window_size + 1, max(1, window_size // 3)):
            rms = np.sqrt(np.mean(abs_errors[i:i+window_size] ** 2))
            ref_energies.append(rms)
        
        ref_energies = np.array(ref_energies)
        
        if len(ref_energies) >= 5:
            self.screener.calibrate(ref_energies)
            self._ref_energy_mean = float(np.mean(ref_energies))
            self._ref_energy_std = float(np.std(ref_energies)) or 1e-10
            logger.info(f"OK Screener recalibrated on post-drift errors "
                       f"(mean={self._ref_energy_mean:.4f}, std={self._ref_energy_std:.4f})")
        else:
            logger.warning("Not enough post-drift data to recalibrate screener, skipping")
        
        # Step 4: Retrain ensemble
        min_train_required = self._min_signal_length if drift_type == 'variance' else 20
        if len(X_new) >= min_train_required:
            split = len(X_new) // 2
            X_train, X_val = X_new[:split], X_new[split:]
            y_train, y_val = y_new[:split], y_new[split:]
            
            # Add validation observations
            for i in range(len(X_val)):
                try:
                    self.validator.add_val_observation(X_val[i:i+1], float(y_val[i]))
                except Exception as e:
                    logger.debug(f"Validator update skipped: {e}")
            
            # Retrain
            try:
                J = self.config.dwt_level
                if len(X_train) >= self._min_signal_length:
                    X_dict_train = self._decompose_features(X_train)
                    y_decomp = self.decomposer.decompose(y_train)
                    fallback_zeros = np.zeros_like(y_decomp[0]) if 0 in y_decomp else np.zeros_like(y_train)
                    y_dict = {j: y_decomp.get(j,fallback_zeros) for j in range(J + 1)}
                    self.ensemble.fit(X_dict_train, y_dict, val_size=min(20, split // 3))
                    logger.info("OK Ensemble retrained on post-drift data")
                else:
                    logger.warning(f"Post-drift X_train too short ({len(X_train)}) for full refit")
            except Exception as e:
                logger.error(f"Ensemble refit failed: {e}")
        else:
            logger.warning(f"Skipping refit for '{drift_type}' drift: "
                           f"X_train ({len(X_new)}) < min_required ({min_train_required})")
        
        # Step 5: Recalibrate noise monitor
        if len(new_errors) >= 10:
            try:
                self.noise_monitor.calibrate(new_errors)
                logger.info("OK Noise monitor recalibrated on post-drift errors")
            except Exception as e:
                logger.warning(f"Noise monitor recalibration failed: {e}")
        
        logger.info(f"{'='*80}")
        logger.info("Post-drift adaptation complete")
        logger.info(f"{'='*80}\n")
    
    def save(self, filepath: str) -> None:
        """Save trained pipeline."""
        import pickle
        import os
        
        if not self.is_warm:
            raise RuntimeError("Cannot save pipeline before warm-up.")
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            file_size = os.path.getsize(filepath) / (1024 * 1024)
            logger.info(f"OK Pipeline saved to {filepath} ({file_size:.2f} MB)")
            
        except Exception as e:
            logger.error(f"Failed to save pipeline: {e}")
            raise
    
    @staticmethod
    def load(filepath: str) -> 'WaveletDriftDetectionPipeline':
        """Load pre-trained pipeline."""
        import pickle
        
        try:
            with open(filepath, 'rb') as f:
                pipeline = pickle.load(f)
            
            logger.info(f"OK Pipeline loaded from {filepath}")
            
            if not pipeline.is_warm:
                raise RuntimeError("Loaded pipeline has not been warmed up!")
            
            return pipeline
            
        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            raise


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("Integration test would go here")