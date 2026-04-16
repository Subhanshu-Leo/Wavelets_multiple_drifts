"""
Wavelet coherence for feature-scale attribution with CORRECTED formula.

CRITICAL FIXES:
1. Move imports to TOP (Dict import-after-use bug)
2. Fix coherence formula: smooth COMPLEX cross-spectrum FIRST, then take modulus
3. Use consistent implementation across all methods
4. Add comprehensive logging and validation
5. Cache error CWT properly to avoid redundant computation

Reference: Torrence & Compo (1998), Grinsted et al. (2004)
"""

from typing import Dict, Tuple, Optional  # ← MOVED TO TOP (CRITICAL FIX)
import numpy as np
import pywt
import logging

logger = logging.getLogger(__name__)


class WaveletCoherence:
    """Compute wavelet coherence between two signals."""
    
    @staticmethod
    def _smooth_complex(cross_spectrum: np.ndarray, 
                       window_size: int = 3) -> np.ndarray:
        """
        Smooth complex cross-spectrum preserving phase information.
        
        CRITICAL FIX: Smooth the complex values directly, then take modulus.
        NOT: take modulus, then smooth (which loses phase).
        
        Args:
            cross_spectrum: Complex array (n_scales, n_time)
            window_size: Smoothing window size
        
        Returns:
            Smoothed complex cross-spectrum
        """
        kernel = np.ones(window_size) / window_size
        
        # Smooth real and imaginary parts separately
        real_smooth = np.array([
            np.convolve(cross_spectrum[j, :].real, kernel, mode='same')
            for j in range(cross_spectrum.shape[0])
        ])
        
        imag_smooth = np.array([
            np.convolve(cross_spectrum[j, :].imag, kernel, mode='same')
            for j in range(cross_spectrum.shape[0])
        ])
        
        # Reconstruct complex
        smooth_complex = real_smooth + 1j * imag_smooth
        
        return smooth_complex
    
    @staticmethod
    def compute(X_signal: np.ndarray, 
               E_signal: np.ndarray,
               scales: Optional[np.ndarray] = None,
               wavelet: str = 'morlet',
               smooth_window: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute squared wavelet coherence between feature X and error E.
        
        Coherence measures phase-locked coupling at each time-frequency point.
        
        CORRECTED FORMULA:
        C_j(t) = |<W_X(s,t) * conj(W_E(s,t))>|^2 / (<|W_X|^2> * <|W_E|^2>)
        
        where <.> denotes local smoothing (BEFORE taking modulus).
        
        Args:
            X_signal: Feature signal (1D array)
            E_signal: Error signal (1D array)
            scales: Scales to analyze (default: auto)
            wavelet: Mother wavelet name
            smooth_window: Smoothing window size (time and scale)
        
        Returns:
            (coherence, scales)
            - coherence: (n_scales, n_time), values in [0, 1]
            - scales: Scale values used
        
        Raises:
            ValueError: If inputs invalid
        
        Reference: Torrence & Compo (1998), Grinsted et al. (2004)
        """
        # Input validation
        if X_signal.ndim != 1 or E_signal.ndim != 1:
            raise ValueError("Both signals must be 1D")
        
        if len(X_signal) != len(E_signal):
            raise ValueError(
                f"Signals must have same length: {len(X_signal)} vs {len(E_signal)}"
            )
        
        if len(X_signal) < 10:
            raise ValueError("Signals too short for coherence (min 10 samples)")
        
        if not np.isfinite(X_signal).all() or not np.isfinite(E_signal).all():
            raise ValueError("Signals contain NaN or inf values")
        
        if scales is None:
            scales = np.arange(1, min(len(X_signal), 64))
        
        try:
            # Continuous Wavelet Transform
            logger.debug(f"Computing CWT for X_signal (len={len(X_signal)}) and E_signal")
            cX, _ = pywt.cwt(X_signal, scales, wavelet)
            cE, _ = pywt.cwt(E_signal, scales, wavelet)
        except Exception as e:
            logger.error(f"CWT failed: {e}")
            raise
        
        n_scales, n_time = cX.shape
        coherence = np.zeros((n_scales, n_time), dtype=float)
        
        # CRITICAL FIX: Compute and smooth cross-spectrum, THEN take modulus
        # Compute cross-spectrum (complex)
        cross_spectrum = cX * np.conj(cE)
        
        # Smooth the COMPLEX cross-spectrum (preserves phase in average)
        smooth_cross = WaveletCoherence._smooth_complex(cross_spectrum, smooth_window)
        
        # Smooth power spectra
        power_X = np.abs(cX) ** 2
        power_E = np.abs(cE) ** 2
        
        smooth_power_X = np.array([
            np.convolve(power_X[j, :], np.ones(smooth_window)/smooth_window, mode='same')
            for j in range(n_scales)
        ])
        
        smooth_power_E = np.array([
            np.convolve(power_E[j, :], np.ones(smooth_window)/smooth_window, mode='same')
            for j in range(n_scales)
        ])
        
        # Compute coherence: |smooth_cross|^2 / (smooth_power_X * smooth_power_E)
        coherence = np.abs(smooth_cross) ** 2 / (smooth_power_X * smooth_power_E + 1e-10)
        
        # Ensure values in [0, 1]
        coherence = np.clip(coherence, 0, 1)
        
        logger.debug(
            f"Computed coherence: shape={coherence.shape}, "
            f"mean={np.mean(coherence):.4f}, max={np.max(coherence):.4f}"
        )
        
        return coherence, scales
    
    @staticmethod
    def get_mean_coherence_at_scale(coherence: np.ndarray,
                                   scale_idx: int) -> float:
        """
        Get mean coherence at specific scale (averaged over time).
        
        Args:
            coherence: (n_scales, n_time) array
            scale_idx: Scale index
        
        Returns:
            Mean coherence at that scale
        
        Raises:
            ValueError: If scale_idx out of range
        """
        if scale_idx >= coherence.shape[0] or scale_idx < 0:
            raise ValueError(
                f"Scale {scale_idx} out of range [0, {coherence.shape[0]-1}]"
            )
        
        mean_coh = float(np.mean(coherence[scale_idx, :]))
        
        return mean_coh
    
    @staticmethod
    def compute_feature_scale_coherence_matrix(
            feature_signals: Dict[int, np.ndarray],
            error_signal: np.ndarray,
            scales: Optional[np.ndarray] = None,
            wavelet: str = 'morlet',
            cache_cwt: bool = True) -> np.ndarray:
        """
        Compute coherence between all features and error across scales.
        
        OPTIMIZATION: Cache error CWT to avoid redundant computation.
        
        Args:
            feature_signals: {feature_idx: signal} — Dict of feature signals
            error_signal: Error signal (all features coherence against same error)
            scales: Scales to analyze
            wavelet: Mother wavelet
            cache_cwt: If True, compute error CWT once (OPTIMIZATION)
        
        Returns:
            coherence_matrix: (n_features, n_scales)
            Each element [i, j] = mean coherence between feature i and error at scale j
        
        Raises:
            ValueError: If inputs invalid
        """
        # Input validation
        if not feature_signals:
            raise ValueError("feature_signals cannot be empty")
        
        if error_signal.ndim != 1:
            raise ValueError(f"error_signal must be 1D, got shape {error_signal.shape}")
        
        if len(error_signal) < 10:
            raise ValueError("error_signal too short for coherence")
        
        n_features = len(feature_signals)
        
        if scales is None:
            scales = np.arange(1, min(len(error_signal), 64))
        
        n_scales = len(scales)
        coherence_matrix = np.zeros((n_features, n_scales))
        
        logger.info(
            f"Computing feature-scale coherence matrix: "
            f"n_features={n_features}, n_scales={n_scales}"
        )
        
        # OPTIMIZATION: Pre-compute error CWT (cached for all features)
        if cache_cwt:
            logger.debug("Pre-computing error CWT (cached for all features)")
            try:
                cE, _ = pywt.cwt(error_signal, scales, wavelet)
                error_cft_cached = True
            except Exception as e:
                logger.error(f"Error CWT failed: {e}")
                raise
        else:
            error_cft_cached = False
            cE = None
        
        # Compute coherence for each feature
        for i, (feat_idx, feat_signal) in enumerate(feature_signals.items()):
            try:
                # Validate feature signal
                if feat_signal.ndim != 1:
                    logger.warning(f"Feature {feat_idx} not 1D, skipping")
                    continue
                
                if len(feat_signal) != len(error_signal):
                    logger.warning(
                        f"Feature {feat_idx} length mismatch ({len(feat_signal)} vs "
                        f"{len(error_signal)}), skipping"
                    )
                    continue
                
                if not np.isfinite(feat_signal).all():
                    logger.warning(f"Feature {feat_idx} contains NaN/inf, skipping")
                    continue
                
                # Compute CWT for this feature
                cX, _ = pywt.cwt(feat_signal, scales, wavelet)
            except Exception as e:
                logger.error(f"Feature {feat_idx} CWT failed: {e}")
                continue
            
            # Compute coherence for this feature (against cached error)
            if error_cft_cached:
                # CRITICAL FIX: Use correct formula with phase preservation
                cross = cX * np.conj(cE)
                smooth_cross = WaveletCoherence._smooth_complex(cross, 3)
                
                power_X = np.abs(cX) ** 2
                power_E = np.abs(cE) ** 2
                
                # Smooth powers
                smooth_power_X = np.array([
                    np.convolve(power_X[j, :], np.ones(3)/3, mode='same')
                    for j in range(len(scales))
                ])
                
                smooth_power_E = np.array([
                    np.convolve(power_E[j, :], np.ones(3)/3, mode='same')
                    for j in range(len(scales))
                ])
                
                # Coherence: |smooth_cross|^2 / (smooth_power_X * smooth_power_E)
                coh = np.abs(smooth_cross) ** 2 / (smooth_power_X * smooth_power_E + 1e-10)
                coh = np.clip(coh, 0, 1)
                
                # Mean over time for each scale
                coherence_matrix[i, :] = np.mean(coh, axis=1)
            else:
                # Fallback: compute full coherence (slower)
                coherence, _ = WaveletCoherence.compute(feat_signal, error_signal, scales, wavelet)
                coherence_matrix[i, :] = np.mean(coherence, axis=1)
            
            logger.debug(
                f"Feature {feat_idx}: mean coherence per scale = "
                f"{coherence_matrix[i, :].round(3)}"
            )
        
        logger.info(
            f"Coherence matrix complete: shape={coherence_matrix.shape}, "
            f"mean={np.mean(coherence_matrix):.4f}"
        )
        
        return coherence_matrix


# Example
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    print("="*70)
    print("WaveletCoherence: CORRECTED Formula Validation")
    print("="*70)
    
    np.random.seed(42)
    
    # Test 1: Coherent signals
    print("\nTest 1: Coherent signals (should have high coherence)")
    feature = np.sin(2*np.pi*np.arange(500)/50)
    error = 0.9 * feature + 0.1*np.random.randn(500)
    
    coherence, scales = WaveletCoherence.compute(feature, error)
    mean_coh = np.mean(coherence)
    
    print(f" Mean coherence: {mean_coh:.4f} (should be > 0.6 for strongly coupled)")
    assert mean_coh > 0.5, "Strongly coupled signals should have high coherence"
    
    # Test 2: Independent signals
    print("\nTest 2: Independent signals (should have low coherence)")
    feature_indep = np.random.randn(500)
    error_indep = np.random.randn(500)
    
    coherence_indep, _ = WaveletCoherence.compute(feature_indep, error_indep)
    mean_coh_indep = np.mean(coherence_indep)
    
    print(f" Mean coherence: {mean_coh_indep:.4f} (should be < 0.3 for independent)")
    assert mean_coh_indep < 0.4, "Independent signals should have low coherence"
    
    # Test 3: Feature-scale coherence matrix (with caching)
    print("\nTest 3: Feature-scale coherence matrix (with CWT caching)")
    feature_dict = {
        0: np.sin(2*np.pi*np.arange(500)/50),
        1: np.random.randn(500),
        2: 0.8 * np.sin(2*np.pi*np.arange(500)/50) + 0.2*np.random.randn(500)
    }
    
    error_signal = np.sin(2*np.pi*np.arange(500)/50) + 0.1*np.random.randn(500)
    
    coherence_matrix = WaveletCoherence.compute_feature_scale_coherence_matrix(
        feature_dict, error_signal, cache_cwt=True
    )
    
    print(f" Coherence matrix shape: {coherence_matrix.shape}")
    print(f" Feature 0 (sine) coherence: {np.mean(coherence_matrix[0, :]):.4f}")
    print(f" Feature 1 (noise) coherence: {np.mean(coherence_matrix[1, :]):.4f}")
    print(f" Feature 2 (mostly sine) coherence: {np.mean(coherence_matrix[2, :]):.4f}")
    
    # Feature 0 and 2 (sine-based) should have higher coherence than feature 1 (noise)
    assert coherence_matrix[0, :].mean() > coherence_matrix[1, :].mean()
    assert coherence_matrix[2, :].mean() > coherence_matrix[1, :].mean()
    print(" Coherence ranking correct: sine-based > pure noise")
    
    print("\n" + "="*70)
    print(" ALL COHERENCE TESTS PASSED!")
    print("="*70)