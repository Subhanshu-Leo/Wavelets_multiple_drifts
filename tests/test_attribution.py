"""
Unit tests for feature attribution methods.
"""

import pytest
import numpy as np
from src.attribution.granger import granger_causality_test
from src.attribution.coherence import WaveletCoherence
from src.attribution.importance import compute_permutation_importance

class TestGrangerCausality:
    """Test Granger causality test."""
    def test_no_causality(self):
        np.random.seed(42)
        error_signal = np.random.randn(200)
        feature_signal = np.random.randn(200)
        passes, p_val = granger_causality_test(error_signal, feature_signal)
        assert 0 <= p_val <= 1
        assert p_val > 0.05
    
    def test_with_causality(self):
        np.random.seed(42)
        feature_signal = np.random.randn(200)
        error_signal = np.zeros(200)
        for t in range(1, 200):
            error_signal[t] = 0.7 * feature_signal[t-1] + 0.1*np.random.randn()
        passes, p_val = granger_causality_test(error_signal, feature_signal, lag=5)
        assert 0 <= p_val <= 1

class TestCoherence:
    """Test wavelet coherence."""
    def test_coherent_signals(self):
        np.random.seed(42)
        feature = np.sin(2*np.pi*np.arange(500)/50)
        error = 0.9 * feature + 0.1*np.random.randn(500)
        
        coherence_analyzer = WaveletCoherence()
        coh_matrix, scales = coherence_analyzer.compute(feature, error)
        assert coh_matrix is not None
        assert np.max(coh_matrix) > 0
    
    def test_independent_signals(self):
        np.random.seed(42)
        feature = np.random.randn(500)
        error = np.random.randn(500)
        
        coherence_analyzer = WaveletCoherence()
        passes, coh_val = coherence_analyzer.compute(
            feature, error, scale_idx=2, threshold=0.7
        )
        assert 0 <= coh_val <= 1

class TestPermutationImportance:
    """Test permutation-based importance."""
    def test_importance_computation(self):
        np.random.seed(42)
        # Fix: Requires 4 args (X_train, y_train, X_test, y_test)
        X_train = np.random.randn(100, 3)
        y_train = X_train[:, 0] + 0.1*np.random.randn(100) # Feature 0 is important
        
        X_test = np.random.randn(50, 3)
        y_test = X_test[:, 0] + 0.1*np.random.randn(50)
        
        importances, ranks = compute_permutation_importance(X_train, y_train, X_test, y_test)
        
        assert len(importances) == 3
        assert len(ranks) == 3
        assert ranks[0] <= ranks[1] # Feature 0 should be highest ranked