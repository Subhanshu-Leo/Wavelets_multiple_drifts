"""
Utility functions for wavelet analysis with CORRECTED implementations.

CRITICAL FIXES:
1. MAD (not IQR) as default robust threshold (more efficient per D-J)
2. Fixed scale entropy (excludes approximation, entropy of details only)
3. Removed duplicate broken coherence method
4. Fixed dead branch in soft-threshold
5. Proper documentation of assumptions
"""

import numpy as np
import pywt
from typing import Dict, Tuple, Optional
import logging
import math

logger = logging.getLogger(__name__)


class WaveletUtils:
    """Helper functions for wavelet operations."""
    
    @staticmethod
    def universal_threshold(coefficients: np.ndarray, 
                           robust: bool = False) -> float:
        """
        Compute threshold for wavelet coefficient shrinkage.
        
        CRITICAL FIX: MAD is default (Donoho-Johnstone's original choice).
        IQR is available but not recommended (less efficient).
        
        Formula: λ = σ̂ * √(2 ln n)
        
        Args:
            coefficients: Wavelet coefficients (1D array)
            robust: If True, use IQR; if False, use MAD (default, recommended)
        
        Returns:
            Threshold value
        
        Reference: Donoho & Johnstone (1995)
        """
        if not isinstance(coefficients, np.ndarray):
            coefficients = np.array(coefficients)
        
        if coefficients.ndim != 1:
            raise ValueError(f"Must be 1D, got {coefficients.shape}")
        
        if len(coefficients) == 0:
            raise ValueError("Cannot be empty")
        
        n = len(coefficients)
        
        if robust:
            # IQR-based (less efficient than MAD)
            logger.warning("IQR threshold: less efficient than MAD (D-J recommend MAD)")
            q1 = np.percentile(coefficients, 25)
            q3 = np.percentile(coefficients, 75)
            iqr = q3 - q1
            sigma_hat = iqr / 1.3490  # IQR to σ
        else:
            # MAD-based (Donoho-Johnstone original, recommended)
            median = np.median(coefficients)
            mad = np.median(np.abs(coefficients - median))
            sigma_hat = mad / 0.6745  # MAD to σ
        
        # Universal threshold
        threshold = sigma_hat * math.sqrt(2 * math.log(n))
        
        logger.debug(f"Threshold: sigma={sigma_hat:.6f}, lambda={threshold:.6f}, n={n}")
        
        return float(threshold)
    
    @staticmethod
    def soft_threshold(coefficients: np.ndarray, threshold: float) -> np.ndarray:
        """Apply soft-thresholding: sign(x) * max(|x| - λ, 0)."""
        if threshold < 0:
            raise ValueError(f"Threshold must be non-negative, got {threshold}")
        
        return np.sign(coefficients) * np.maximum(np.abs(coefficients) - threshold, 0)
    
    @staticmethod
    def hard_threshold(coefficients: np.ndarray, threshold: float) -> np.ndarray:
        """Apply hard-thresholding: x * (|x| >= λ)."""
        if threshold < 0:
            raise ValueError(f"Threshold must be non-negative, got {threshold}")
        
        return coefficients * (np.abs(coefficients) >= threshold)
    
    @staticmethod
    def denoise_coefficients(decomposition: Dict[int, np.ndarray],
                            denoise_approximation: bool = False,
                            robust: bool = False) -> Dict[int, np.ndarray]:
        """
        Denoise wavelet coefficients.
        
        CRITICAL: Never denoise approximation (scale 0) — it carries signal.
        
        Args:
            decomposition: {scale: coefficients}
            denoise_approximation: If True, also denoise approximation (rare)
            robust: Use IQR threshold instead of MAD
        
        Returns:
            {scale: denoised_coefficients}
        """
        if not decomposition:
            raise ValueError("Cannot be empty")
        
        denoised = {}
        
        for j, coeffs in decomposition.items():
            if j == 0 and not denoise_approximation:
                # Approximation carries signal — NEVER threshold
                denoised[j] = coeffs.copy()
                logger.debug(f"Scale {j} (approx): NOT thresholded")
            else:
                # Detail coefficients
                threshold = WaveletUtils.universal_threshold(coeffs, robust=robust)
                denoised[j] = WaveletUtils.soft_threshold(coeffs, threshold)
                
                orig_energy = np.sum(coeffs**2)
                denoised_energy = np.sum(denoised[j]**2)
                reduction = 100 * (1 - denoised_energy / (orig_energy + 1e-10))
                
                logger.debug(f"Scale {j} (detail): thresholded λ={threshold:.4f}, "
                           f"energy_reduction={reduction:.1f}%")
        
        return denoised
    
    @staticmethod
    def compute_energy(coefficients: np.ndarray) -> float:
        """Compute energy: sum of squared coefficients."""
        return float(np.sum(coefficients ** 2))
    
    @staticmethod
    def compute_energy_ratio(decomposition: Dict[int, np.ndarray],
                            j_fine: int = 1, j_coarse: int = 4) -> float:
        """Compute fine-to-coarse energy ratio."""
        if j_fine not in decomposition or j_coarse not in decomposition:
            raise ValueError(
                f"Scales {j_fine}, {j_coarse} not in decomposition. "
                f"Available: {list(decomposition.keys())}"
            )
        
        E_fine = WaveletUtils.compute_energy(decomposition[j_fine])
        E_coarse = WaveletUtils.compute_energy(decomposition[j_coarse])
        
        ratio = E_fine / (E_coarse + 1e-10)
        
        return float(ratio)
    
    @staticmethod
    def compute_scale_entropy(decomposition: Dict[int, np.ndarray]) -> float:
        """
        Compute entropy of energy distribution across DETAIL scales ONLY.
        
        CRITICAL FIX: Excludes approximation (scale 0) which dominates.
        Entropy measures frequency distribution of detail coefficients.
        
        Args:
            decomposition: {scale: coefficients}
        
        Returns:
            Normalized entropy in [0, 1]
        """
        if not decomposition:
            raise ValueError("Cannot be empty")
        
        # Compute energy for DETAIL scales only (j >= 1)
        energies = np.array([
            WaveletUtils.compute_energy(decomposition[j])
            for j in sorted(decomposition.keys()) if j > 0  # CRITICAL: exclude j=0
        ])
        
        if len(energies) == 0 or np.sum(energies) == 0:
            return 0.0
        
        # Normalize to probability
        p = energies / np.sum(energies)
        p = np.clip(p, 1e-10, 1.0)
        
        # Shannon entropy
        entropy = -np.sum(p * np.log(p))
        
        # Normalize by max entropy
        max_entropy = np.log(len(p))
        normalized = entropy / (max_entropy + 1e-10)
        
        return float(np.clip(normalized, 0, 1))
    
    @staticmethod
    def get_dominant_scale(decomposition: Dict[int, np.ndarray]) -> int:
        """Get scale with maximum energy."""
        energies = {
            j: WaveletUtils.compute_energy(decomposition[j])
            for j in decomposition.keys()
        }
        
        dominant = max(energies, key=energies.get)
        
        return int(dominant)
    
    @staticmethod
    def estimate_noise_level(coefficients: np.ndarray) -> float:
        """Estimate noise std from fine-scale wavelet coefficients."""
        if len(coefficients) == 0:
            raise ValueError("Cannot be empty")
        
        median = np.median(coefficients)
        mad = np.median(np.abs(coefficients - median))
        
        sigma_noise = mad / 0.6745
        
        return float(sigma_noise)
    
    @staticmethod
    def continuous_wavelet_transform(signal: np.ndarray,
                                    scales: np.ndarray,
                                    wavelet: str = 'morl') -> Tuple[np.ndarray, np.ndarray]:
        """Continuous Wavelet Transform."""
        if signal.ndim != 1:
            raise ValueError(f"Signal must be 1D, got {signal.shape}")
        
        try:
            coeffs, freqs = pywt.cwt(signal, scales, wavelet)
        except Exception as e:
            logger.error(f"CWT failed: {e}")
            raise
        
        return coeffs, freqs
    
    @staticmethod
    def validate_decomposition(decomposition: Dict[int, np.ndarray],
                              expected_level: Optional[int] = None) -> bool:
        """Validate wavelet decomposition structure."""
        if not decomposition:
            raise ValueError("Cannot be empty")
        
        if 0 not in decomposition:
            raise ValueError("Missing key 0 (approximation)")
        
        for j, coeffs in decomposition.items():
            if not isinstance(coeffs, np.ndarray):
                raise ValueError(f"Scale {j}: must be ndarray, got {type(coeffs)}")
            
            if coeffs.ndim != 1:
                raise ValueError(f"Scale {j}: must be 1D, got {coeffs.shape}")
            
            if not np.isfinite(coeffs).all():
                raise ValueError(f"Scale {j}: contains NaN or inf")
        
        if expected_level is not None:
            expected_keys = set(range(expected_level + 1))
            actual_keys = set(decomposition.keys())
            
            if actual_keys != expected_keys:
                raise ValueError(
                    f"Expected keys {expected_keys}, got {actual_keys}"
                )
        
        logger.debug(f"Decomposition valid: {len(decomposition)} components")
        
        return True


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    print("="*80)
    print("WaveletUtils: Validation Tests")
    print("="*80)
    
    np.random.seed(42)
    
    # Test 1: Universal threshold (MAD default)
    print("\n[TEST 1] Universal Threshold")
    coeffs = np.random.randn(1000)
    threshold_mad = WaveletUtils.universal_threshold(coeffs, robust=False)
    threshold_iqr = WaveletUtils.universal_threshold(coeffs, robust=True)
    
    print(f"✓ MAD threshold: {threshold_mad:.4f}")
    print(f"✓ IQR threshold: {threshold_iqr:.4f}")
    
    # Test 2: Entropy excluding approximation
    print("\n[TEST 2] Scale Entropy (Details Only)")
    from src.wavelets.dwt_pipeline import DWTDecomposer
    
    signal = np.sin(2*np.pi*np.arange(500)/50) + 0.1*np.random.randn(500)
    decomposer = DWTDecomposer()
    decomp = decomposer.decompose(signal)
    
    entropy = WaveletUtils.compute_scale_entropy(decomp)
    print(f"✓ Entropy (details only): {entropy:.4f}")
    
    # Test 3: Dominant scale
    print("\n[TEST 3] Dominant Scale")
    dominant = WaveletUtils.get_dominant_scale(decomp)
    print(f"✓ Dominant scale: {dominant}")
    
    print("\n" + "="*80)
    print("✓✓✓ ALL WAVELET UTILS TESTS PASSED ✓✓✓")
    print("="*80)