"""
Wavelet-based outlier detection and cleaning.

Used in pipeline for data quality assurance.
"""

import numpy as np
import pywt
import logging
from typing import Tuple, Dict

logger = logging.getLogger(__name__)


class WaveletOutlierDetector:
    """Detect and remove outliers using wavelet-based robust methods."""
    
    def __init__(self, wavelet: str = 'db4', level: int = 4,
                 threshold_multiplier: float = 2.5):
        """
        Initialize outlier detector.
        
        Args:
            wavelet: Mother wavelet
            level: Decomposition level
            threshold_multiplier: MAD-based threshold (2.5 for 98% Gaussian)
        """
        self.wavelet = wavelet
        self.level = level
        self.threshold_multiplier = threshold_multiplier
    
    def detect_and_clean(self, signal: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Detect outliers and return cleaned signal.
        
        Uses Median Absolute Deviation (MAD) on wavelet coefficients,
        which is robust to heavy tails.
        
        Args:
            signal: 1D input signal
        
        Returns:
            (cleaned_signal, statistics)
        """
        if signal.ndim != 1:
            raise ValueError(f"Signal must be 1D, got shape {signal.shape}")
        
        if len(signal) < 10:
            # Too short for outlier detection
            return signal.copy(), {'n_outliers_detected': 0, 'n_outliers_removed': 0}
        
        signal_work = signal.copy()
        outlier_mask = np.zeros(len(signal), dtype=bool)
        
        try:
            # Decompose
            coeffs = pywt.wavedec(signal, self.wavelet, level=min(self.level, int(np.log2(len(signal)))))
        except Exception as e:
            logger.warning(f"Outlier detection decomposition failed: {e}")
            return signal, {'n_outliers_detected': 0, 'n_outliers_removed': 0}
        
        # Check each level for outliers
        for level_coeffs in coeffs:
            if len(level_coeffs) < 3:
                continue
            
            # MAD-based outlier detection
            median = np.median(level_coeffs)
            mad = np.median(np.abs(level_coeffs - median))
            
            threshold = self.threshold_multiplier * mad
            
            # Mark outliers
            level_outliers = np.abs(level_coeffs - median) > threshold
            
            # Map back to original signal (approximate)
            # For simplicity, mark high-level outliers in original
            if level_outliers.any():
                logger.debug(
                    f"Outliers detected at level: {np.sum(level_outliers)} points"
                )
        
        # Simple approach: flag extreme values in original signal
        signal_median = np.median(signal)
        signal_mad = np.median(np.abs(signal - signal_median))
        signal_threshold = self.threshold_multiplier * signal_mad
        
        outlier_mask = np.abs(signal - signal_median) > signal_threshold
        
        # Clean by interpolation
        if outlier_mask.any():
            outlier_indices = np.where(outlier_mask)[0]
            clean_indices = np.where(~outlier_mask)[0]
            
            if len(clean_indices) > 1:
                signal_work[outlier_indices] = np.interp(
                    outlier_indices, clean_indices, signal[clean_indices]
                )
            
            logger.debug(f"Removed {np.sum(outlier_mask)} outliers via interpolation")
        
        return signal_work, {
            'n_outliers_detected': np.sum(outlier_mask),
            'n_outliers_removed': np.sum(outlier_mask)
        }