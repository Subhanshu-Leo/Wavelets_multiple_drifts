"""
Wavelet denoising with CORRECTED dead branch removal.

CRITICAL FIXES:
1. Dead branch removed (was never executing)
2. Proper approximation preservation
3. Delegate to WaveletUtils
"""

import numpy as np
import pywt
import logging
from typing import Dict

from src.wavelets.wavelet_utils import WaveletUtils

logger = logging.getLogger(__name__)


class WaveletDenoiser:
    """Denoise signals using wavelet soft-thresholding."""
    
    def __init__(self, wavelet: str = 'db4', level: int = 4,
                 robust: bool = False, denoise_approximation: bool = False):
        """
        Initialize denoiser.
        
        Args:
            wavelet: Mother wavelet
            level: Decomposition level
            robust: Use IQR instead of MAD threshold
            denoise_approximation: If True, also denoise approximation (rare)
        """
        self.wavelet = wavelet
        self.level = level
        self.robust = robust
        self.denoise_approximation = denoise_approximation
        
        try:
            pywt.Wavelet(wavelet)
        except ValueError as e:
            raise ValueError(f"Invalid wavelet: {e}")
        
        logger.info(
            f"Initialized WaveletDenoiser: wavelet={wavelet}, level={level}, "
            f"robust={robust}, denoise_approx={denoise_approximation}"
        )
    
    def denoise(self, signal: np.ndarray) -> np.ndarray:
        """
        Denoise signal using wavelet soft-thresholding.
        
        Args:
            signal: 1D input signal
        
        Returns:
            Denoised signal (same length as input)
        """
        # Input validation
        if not isinstance(signal, np.ndarray):
            signal = np.array(signal)
        
        if signal.ndim != 1:
            raise ValueError(f"Signal must be 1D, got {signal.shape}")
        
        if len(signal) < 10:
            raise ValueError(f"Signal too short ({len(signal)} < 10)")
        
        if not np.isfinite(signal).all():
            raise ValueError("Signal contains NaN or inf")
        
        try:
            coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)
        except Exception as e:
            logger.error(f"Decomposition failed: {e}")
            raise
        
        # CRITICAL FIX: Dead branch removed. Proper implementation:
        # coeffs[0] is approximation (never threshold)
        # coeffs[1:] are detail coefficients (threshold all)
        
        coeffs_denoised = [coeffs[0]]  # Keep approximation unmodified
        
        # Threshold all detail coefficients
        for d_coeffs in coeffs[1:]:
            threshold = WaveletUtils.universal_threshold(d_coeffs, robust=self.robust)
            d_denoised = WaveletUtils.soft_threshold(d_coeffs, threshold)
            coeffs_denoised.append(d_denoised)
        
        # Reconstruct
        try:
            signal_denoised = pywt.waverec(coeffs_denoised, self.wavelet)
        except Exception as e:
            logger.error(f"Reconstruction failed: {e}")
            raise
        
        # Align length
        if len(signal_denoised) > len(signal):
            signal_denoised = signal_denoised[:len(signal)]
        elif len(signal_denoised) < len(signal):
            signal_denoised = np.concatenate([
                signal_denoised,
                np.zeros(len(signal) - len(signal_denoised))
            ])
        
        logger.info(f"Denoising complete: input_len={len(signal)}, output_len={len(signal_denoised)}")
        
        return signal_denoised
    
    def denoise_dict(self, decomposition: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """Denoise a decomposition dictionary (scale-wise)."""
        if not decomposition:
            raise ValueError("Decomposition cannot be empty")
        
        denoised = WaveletUtils.denoise_coefficients(
            decomposition,
            denoise_approximation=self.denoise_approximation,
            robust=self.robust
        )
        
        logger.info(f"Denoised decomposition: {len(denoised)} levels")
        
        return denoised


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    print("="*80)
    print("WaveletDenoiser: Dead Branch Fixed")
    print("="*80)
    
    np.random.seed(42)
    
    # Test 1: Denoise sine + noise
    print("\n[TEST 1] Denoise Sine + Noise")
    signal_clean = np.sin(2*np.pi*np.arange(500)/50)
    signal_noisy = signal_clean + 0.2*np.random.randn(500)
    
    denoiser = WaveletDenoiser(wavelet='db4', level=4, robust=False)
    signal_denoised = denoiser.denoise(signal_noisy)
    
    mse_noisy = np.mean((signal_clean - signal_noisy)**2)
    mse_denoised = np.mean((signal_clean - signal_denoised)**2)
    
    print(f"✓ Original MSE: {mse_noisy:.6f}")
    print(f"✓ Denoised MSE: {mse_denoised:.6f}")
    assert mse_denoised < mse_noisy
    
    # Test 2: Approximation preserved
    print("\n[TEST 2] Approximation Preserved")
    from src.wavelets.dwt_pipeline import DWTDecomposer
    
    decomposer = DWTDecomposer()
    decomp = decomposer.decompose(signal_noisy)
    
    denoised_dict = denoiser.denoise_dict(decomp)
    
    assert np.allclose(decomp[0], denoised_dict[0]), "Approximation should be unchanged"
    assert not np.allclose(decomp[1], denoised_dict[1]), "Details should be thresholded"
    
    print("✓ Approximation preserved (not thresholded)")
    print("✓ Details thresholded")
    
    print("\n" + "="*80)
    print("✓✓✓ ALL DENOISER TESTS PASSED ✓✓✓")
    print("="*80)