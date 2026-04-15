"""
Noise monitoring for drift detection pipeline.

Detects sudden spikes in prediction error variance that might indicate
sensor glitches or extreme events (not drift).
"""

import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class NoiseMonitor:
    """
    Monitor noise levels in prediction errors.
    
    Uses wavelet-based MAD (Median Absolute Deviation) to detect sudden
    spikes in error variance while being robust to mean shifts.
    
    FIX 1: Detrend errors before computing noise to avoid mistaking
    mean shifts (drift) for noise spikes.
    """
    
    def __init__(self, spike_threshold: float = 3.5, wavelet: str = 'db4'):
        """
        Initialize noise monitor.
        
        Args:
            spike_threshold: How many MAD above baseline = noise spike (raised from 2.0)
            wavelet: Wavelet for decomposition
        """
        self.spike_threshold = spike_threshold
        self.wavelet = wavelet
        
        # Calibrated baseline noise
        self.sigma_noise_hist = None  # Will be set during warm-up
        self.is_calibrated = False
        
        logger.info(f"Initialized NoiseMonitor: spike_threshold={spike_threshold}, wavelet={wavelet}")
    
    def calibrate(self, signal: np.ndarray) -> None:
        """
        Calibrate noise monitor on clean (pre-drift) error stream.
        
        FIX 3: Lower minimum length from 50 to 20 to allow calibration on shorter signals.
        
        Args:
            signal: Error signal from warm-up (should be stable, no drift)
        """
        if len(signal) < 20:  # CHANGED: was 50
            raise ValueError(f"Signal too short ({len(signal)} < 20)")
        
        try:
            # Estimate baseline noise using MAD
            sigma = self._estimate_noise_mad(signal)
            
            if sigma < 1e-10:
                logger.warning("Estimated noise is near-zero, setting to default")
                sigma = np.std(signal) if len(signal) > 0 else 1.0
            
            self.sigma_noise_hist = sigma
            self.is_calibrated = True
            
            logger.info(f"NoiseMonitor calibrated: baseline_noise={self.sigma_noise_hist:.6f}")
        
        except Exception as e:
            logger.error(f"Noise calibration failed: {e}")
            self.sigma_noise_hist = None
            self.is_calibrated = False
    
    def detect_noise_spike(self, signal: np.ndarray) -> Tuple[bool, float]:
        """
        Detect sudden noise spike in error signal.
        
        FIX 1: Detrend first to avoid mistaking mean shifts for noise.
        
        Args:
            signal: Error window (typically last 50-100 samples)
        
        Returns:
            (is_spike, ratio) where ratio = sigma_new / sigma_hist
        """
        if len(signal) < 10:
            return False, 0.0
        
        if self.sigma_noise_hist is None:
            # Not calibrated yet — use first call to establish baseline
            logger.debug("Noise monitor not calibrated, setting baseline from first call")
            self.sigma_noise_hist = self._estimate_noise_mad(signal)
            return False, 1.0
        
        try:
            # ===== FIX 1: DETREND FIRST =====
            # Remove linear trend so a mean shift doesn't look like a noise spike
            detrended = signal - np.linspace(signal[0], signal[-1], len(signal))
            
            # Estimate noise on detrended signal
            sigma_new = self._estimate_noise_mad(detrended)
            
            # Compute ratio
            ratio = sigma_new / (self.sigma_noise_hist + 1e-10)
            
            # Check if spike
            is_spike = ratio > self.spike_threshold
            
            logger.debug(f"Noise check: sigma_new={sigma_new:.4f}, "
                        f"sigma_hist={self.sigma_noise_hist:.4f}, "
                        f"ratio={ratio:.4f}, threshold={self.spike_threshold}")
            
            return is_spike, ratio
        
        except Exception as e:
            logger.debug(f"Noise spike detection failed: {e}")
            return False, 0.0
    
    @staticmethod
    def _estimate_noise_mad(signal: np.ndarray) -> float:
        """
        Estimate noise using Median Absolute Deviation (MAD).
        
        MAD is robust to outliers and mean shifts.
        
        Args:
            signal: Error signal
        
        Returns:
            Estimated noise standard deviation
        """
        if len(signal) < 2:
            return 0.0
        
        # Compute MAD
        median = np.median(signal)
        mad = np.median(np.abs(signal - median))
        
        # Convert MAD to sigma (for Gaussian: sigma = MAD / 0.6745)
        sigma = mad / 0.6745 if mad > 0 else np.std(signal)
        
        return sigma
    
    def reset(self) -> None:
        """Reset monitor (for testing)."""
        self.sigma_noise_hist = None
        self.is_calibrated = False
        logger.debug("NoiseMonitor reset")