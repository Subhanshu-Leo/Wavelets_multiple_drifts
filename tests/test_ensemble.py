"""
Unit tests for heterogeneous ensemble.
"""

import pytest
import numpy as np
from src.ensemble.heterogeneous import HeterogeneousEnsemble, MultiResolutionEnsemble

class TestHeterogeneousEnsemble:
    """Test HeterogeneousEnsemble class."""
    
    @pytest.fixture
    def dummy_data(self):
        """Create dummy training data."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        return X, y
    
    def test_initialization(self):
        """Test ensemble initialization."""
        ensemble = HeterogeneousEnsemble()
        assert not ensemble.is_trained
    
    def test_fit(self, dummy_data):
        """Test training."""
        X, y = dummy_data
        ensemble = HeterogeneousEnsemble()
        ensemble.fit(X, y)
        assert ensemble.is_trained
    
    def test_predict(self, dummy_data):
        """Test prediction."""
        X, y = dummy_data
        ensemble = HeterogeneousEnsemble()
        ensemble.fit(X, y)
        y_pred = ensemble.predict(X)
        assert y_pred.shape == (100,)
        assert not np.isnan(y_pred).any()

    def test_prediction_not_trained(self, dummy_data):
        """Test that prediction fails if not trained."""
        X, y = dummy_data
        ensemble = HeterogeneousEnsemble()
        with pytest.raises(RuntimeError):
            ensemble.predict(X)

# [MultiResolutionEnsemble tests remain the same]

class TestMultiResolutionEnsemble:
    """Test MultiResolutionEnsemble class."""
    
    @pytest.fixture
    def multi_scale_data(self):
        """Create multi-scale training data."""
        np.random.seed(42)
        J = 2
        n = 100
        X_dict = {j: np.random.randn(n, 3) for j in range(J+1)}
        y_dict = {j: np.random.randn(n) for j in range(J+1)}
        return X_dict, y_dict, J
    
    def test_initialization(self):
        """Test initialization."""
        ensemble = MultiResolutionEnsemble(J=4)
        assert len(ensemble.learners) == 5  # 0 to 4
    
    def test_fit(self, multi_scale_data):
        """Test training on multi-scale data."""
        X_dict, y_dict, J = multi_scale_data
        ensemble = MultiResolutionEnsemble(J=J)
        
        ensemble.fit(X_dict, y_dict)
        
        assert len(ensemble.initial_weights) > 0
        assert sum(ensemble.initial_weights.values()) == pytest.approx(1.0)
    
    def test_predict(self, multi_scale_data):
        """Test prediction."""
        X_dict, y_dict, J = multi_scale_data
        ensemble = MultiResolutionEnsemble(J=J)
        ensemble.fit(X_dict, y_dict)
        
        y_pred = ensemble.predict(X_dict)
        
        assert y_pred.shape == (100,)
        assert not np.isnan(y_pred).any()
    
    def test_weights_are_normalized(self, multi_scale_data):
        """Test that weights sum to 1."""
        X_dict, y_dict, J = multi_scale_data
        ensemble = MultiResolutionEnsemble(J=J)
        ensemble.fit(X_dict, y_dict)
        
        total = sum(ensemble.initial_weights.values())
        assert total == pytest.approx(1.0)
    
    def test_custom_weights(self, multi_scale_data):
        """Test prediction with custom weights."""
        X_dict, y_dict, J = multi_scale_data
        ensemble = MultiResolutionEnsemble(J=J)
        ensemble.fit(X_dict, y_dict)
        
        custom_weights = {j: 1.0/(J+1) for j in range(J+1)}
        y_pred = ensemble.predict(X_dict, weights=custom_weights)
        
        assert y_pred.shape == (100,)