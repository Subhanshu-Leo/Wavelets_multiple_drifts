"""
Discrete Wavelet Transform pipeline with CORRECTED index mapping.
Reference: Percival & Walden (2000)
"""

import numpy as np
import pywt
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class DWTDecomposer:
    """
    Decompose signals into wavelet coefficients with correct scale indexing.
    
    CRITICAL FIX: pywt.wavedec returns [cA_n, cD_n, cD_{n-1}, ..., cD_1]
    We map:
    - decomposition[0] → approximation (coarsest)
    - decomposition[j] → detail at scale j (j=1 is finest, j=level is coarsest detail)
    """
    
    def __init__(self, wavelet: str = 'db4', level: int = 4):
        """
        Initialize DWT decomposer.
        
        Args:
            wavelet: Mother wavelet ('db4', 'sym4', etc.)
            level: Decomposition level
        
        Raises:
            ValueError: If wavelet is not available
        """
        try:
            self.wavelet = wavelet
            self.level = level
            self._filter_length = len(pywt.Wavelet(wavelet).dec_lo)
            self._min_signal_length = self._compute_min_length()
            self._last_signal_length = None
            logger.info(f"Initialized DWTDecomposer: wavelet={wavelet}, level={level}, min_length={self._min_signal_length}")
        except ValueError as e:
            raise ValueError(f"Invalid wavelet '{wavelet}': {e}")
    
    def _compute_min_length(self) -> int:
        """
        Compute minimum signal length for this wavelet and level.
        
        For a decomposition at level J with filter length L:
        min_length ≈ (L - 1) * 2^J + 1
        """
        return (self._filter_length - 1) * (2 ** self.level) + 1
    
    def decompose(self, signal: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Decompose signal into wavelet coefficients.
        
        Args:
            signal: 1D time series
        
        Returns:
            {0: approximation_cA, 1: detail_cD1 (finest), ..., level: detail_cDJ}
        
        Raises:
            ValueError: If signal is too short or not 1D
        """
        # Input validation
        if not isinstance(signal, np.ndarray):
            signal = np.array(signal)
        
        if signal.ndim != 1:
            raise ValueError(f"Signal must be 1D, got shape {signal.shape}")
        
        if len(signal) < self._min_signal_length:
            raise ValueError(
                f"Signal length {len(signal)} too short for decomposition. "
                f"Minimum: {self._min_signal_length}"
            )
        
        if not np.isfinite(signal).all():
            raise ValueError("Signal contains NaN or inf values")
        
        # Store length for reconstruction
        self._last_signal_length = len(signal)
        
        # Decompose
        coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)
        
        # Map coefficients to correct scales
        # coeffs[0] = cA (approximation, coarsest)
        # coeffs[1] = cD_J (coarsest detail)
        # coeffs[J] = cD_1 (finest detail)
        
        decomposition = {
            0: coeffs[0]  # Approximation at coarsest level
        }
        
        # Map details: j=1 is finest (highest freq), j=level is coarsest detail
        for j in range(1, self.level + 1):
            # coeffs[k] corresponds to detail at level (level + 2 - k)
            # So coeffs[self.level + 1 - j] is the detail at level j
            decomposition[j] = coeffs[self.level + 1 - j]
        
        logger.debug(f"Decomposed signal (len={len(signal)}) into {len(decomposition)} components")
        
        return decomposition
    
    def reconstruct(self, decomposition: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Reconstruct signal from wavelet coefficients.
        
        Args:
            decomposition: {scale: coefficients}
        
        Returns:
            Reconstructed signal aligned to original length
        
        Raises:
            ValueError: If decomposition is incomplete or corrupted
        """
        if self._last_signal_length is None:
            raise RuntimeError(
                "reconstruct() called before decompose(). "
                "Call decompose() first to set the reference length."
            )
        
        if 0 not in decomposition:
            raise ValueError("Decomposition missing approximation coefficients (key 0)")
        
        # Reconstruct from decomposition dict
        coeffs = [decomposition[0]]  # Start with approximation
        
        for j in range(self.level, 0, -1):
            if j not in decomposition:
                raise ValueError(f"Decomposition missing detail coefficients at scale {j}")
            coeffs.append(decomposition[j])
        
        # pywt.waverec expects [cA, cD_J, cD_{J-1}, ..., cD_1]
        signal = pywt.waverec(coeffs, self.wavelet)
        
        # Align to original length
        if len(signal) > self._last_signal_length:
            signal = signal[:self._last_signal_length]
        elif len(signal) < self._last_signal_length:
            signal = np.concatenate([
                signal,
                np.zeros(self._last_signal_length - len(signal))
            ])
        
        logger.debug(f"Reconstructed signal to length {len(signal)}")
        return signal
    
    def compute_energy(self, coefficients: np.ndarray) -> float:
        """Compute wavelet energy: sum of squared coefficients."""
        energy = float(np.sum(coefficients ** 2))
        return energy
    
    def reset(self):
        """Reset internal state (useful when processing new streams)."""
        self._last_signal_length = None
        logger.debug("DWTDecomposer reset")


# Example and validation
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Test 1: Basic decomposition/reconstruction
    print("Test 1: Decomposition/Reconstruction")
    signal = np.sin(2*np.pi*np.arange(1000)/100) + 0.1*np.random.randn(1000)
    decomposer = DWTDecomposer(wavelet='db4', level=4)
    
    decomp = decomposer.decompose(signal)
    reconstructed = decomposer.reconstruct(decomp)
    
    mse = np.mean((signal - reconstructed)**2)
    print(f"✓ Reconstruction MSE: {mse:.2e}")
    assert mse < 1e-10, f"Reconstruction error too large: {mse}"
    
    # Test 2: Scale ordering (energy should decrease from coarse to fine for smooth signals)
    print("\nTest 2: Scale Ordering (j=1 is fine, j=4 is coarse)")
    signal_smooth = np.sin(2*np.pi*np.arange(1000)/100)  # Smooth sine
    decomp_smooth = decomposer.decompose(signal_smooth)
    
    print("Scale | Energy | Interpretation")
    print("-" * 50)
    for j in range(5):
        energy = decomposer.compute_energy(decomp_smooth[j])
        if j == 0:
            interp = "Approximation (coarsest)"
        else:
            interp = f"Detail (j={j}, {'finest' if j==1 else 'coarse' if j==4 else 'mid'})"
        print(f"{j:5d} | {energy:10.2f} | {interp}")
    
    # Coarse signal should have most energy in coarse components
    assert decomposer.compute_energy(decomp_smooth[0]) > decomposer.compute_energy(decomp_smooth[1])
    print("✓ Scale ordering correct")
    
    # Test 3: Input validation
    print("\nTest 3: Input Validation")
    try:
        decomposer.decompose(np.array([1, 2, 3]))  # Too short
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Caught short signal: {e}")
    
    print("\n✓ All tests passed!")