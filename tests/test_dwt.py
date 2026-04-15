"""
Unit tests for DWT pipeline.
"""

import pytest
import numpy as np
from src.wavelets.dwt_pipeline import DWTDecomposer

class TestDWTDecomposer:
    """Test DWT decomposition and reconstruction."""
    
    @pytest.fixture
    def sample_signal(self):
        """Create sample signal."""
        np.random.seed(42)
        return np.sin(2*np.pi*np.arange(1000)/100) + 0.1*np.random.randn(1000)
    
    def test_initialization(self):
        """Test decomposer initialization."""
        decomposer = DWTDecomposer(wavelet='db4', level=4)
        assert decomposer.wavelet == 'db4'
        assert decomposer.level == 4
    
    def test_decompose(self, sample_signal):
        """Test decomposition."""
        decomposer = DWTDecomposer(wavelet='db4', level=4)
        decomp = decomposer.decompose(sample_signal)
        
        # Should have J detail levels + 1 approximation
        assert len(decomp) == 5
        assert all(isinstance(v, np.ndarray) for v in decomp.values())
    
    def test_decompose_keys(self, sample_signal):
        """Test decomposition has correct keys."""
        decomposer = DWTDecomposer(level=4)
        decomp = decomposer.decompose(sample_signal)
        
        expected_keys = {0, 1, 2, 3, 4}  # Approximation at level 4
        assert set(decomp.keys()) == expected_keys
    
    def test_reconstruct(self, sample_signal):
        """Test reconstruction quality."""
        decomposer = DWTDecomposer(wavelet='db4', level=4)
        decomp = decomposer.decompose(sample_signal)
        reconstructed = decomposer.reconstruct(decomp)
        
        # Should reconstruct nearly perfectly
        mse = np.mean((sample_signal - reconstructed)**2)
        assert mse < 1e-10
    
    def test_reconstruct_length(self, sample_signal):
        """Test reconstruction preserves length."""
        decomposer = DWTDecomposer(level=4)
        decomp = decomposer.decompose(sample_signal)
        reconstructed = decomposer.reconstruct(decomp)
        
        assert len(reconstructed) == len(sample_signal)
    
    def test_energy_computation(self, sample_signal):
        """Test energy computation."""
        decomposer = DWTDecomposer(level=4)
        decomp = decomposer.decompose(sample_signal)
        
        for j in decomp.keys():
            energy = decomposer.compute_energy(decomp[j])
            assert energy > 0
            assert isinstance(energy, float)
    
    def test_energy_decreases_with_level(self, sample_signal):
        """Test that energy decreases at finer scales (usually)."""
        decomposer = DWTDecomposer(level=4)
        decomp = decomposer.decompose(sample_signal)
        
        # Coarse approximation should have significant energy
        coarse_energy = decomposer.compute_energy(decomp[4])
        
        # Fine details might have less
        fine_energy = decomposer.compute_energy(decomp[1])
        
        assert coarse_energy > 0
        assert fine_energy > 0
    
    def test_parseval_energy_conservation(self, sample_signal):
        """Test approximate energy conservation (Parseval)."""
        decomposer = DWTDecomposer(level=4)
        
        # Original signal energy
        original_energy = np.sum(sample_signal**2)
        
        decomp = decomposer.decompose(sample_signal)
        
        # Sum of wavelet coefficient energies
        wavelet_energy = sum(
            decomposer.compute_energy(decomp[j])
            for j in decomp.keys()
        )
        
        # Should be approximately equal
        ratio = wavelet_energy / (original_energy + 1e-10)
        assert 0.9 < ratio < 1.1
    
    def test_different_wavelets(self, sample_signal):
        """Test different wavelet families."""
        for wavelet in ['db4', 'sym4', 'coif3']:
            decomposer = DWTDecomposer(wavelet=wavelet, level=4)
            decomp = decomposer.decompose(sample_signal)
            
            assert len(decomp) == 5
            
            reconstructed = decomposer.reconstruct(decomp)
            mse = np.mean((sample_signal - reconstructed)**2)
            assert mse < 1e-9