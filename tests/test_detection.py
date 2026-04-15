"""
Unit tests for drift detection modules.
"""

import pytest
import numpy as np
from src.detection.layer1_hoeffding import HoeffingScreener
from src.detection.layer2_permutation import AdaptivePermutationTest
from src.detection.oob_validation import OOBValidator

class TestHoeffingScreener:
    """Test Hoeffing Screener calculations."""
    
    def test_screener_calibration(self):
        """Test calibration of reference energies."""
        np.random.seed(42)
        ref_energies = np.abs(np.random.randn(50) + 1.0)
        
        screener = HoeffingScreener(alpha=0.05)
        screener.calibrate(ref_energies)
        
        assert screener._mean is not None
        assert screener._std is not None
        assert screener.is_calibrated
    
    def test_should_trigger(self):
        """Test trigger decision."""
        np.random.seed(42)
        ref_energies = np.abs(np.random.randn(50) + 1.0)
        
        screener = HoeffingScreener(alpha=0.05)
        screener.calibrate(ref_energies)
        
        # High error should trigger
        high_error_window = np.abs(np.random.randn(20) + 5.0)
        assert screener.check(high_error_window) == True
        
        # Normal error should not trigger
        normal_error_window = np.abs(np.random.randn(20) + 1.0)
        assert screener.check(normal_error_window) == False

class TestAdaptivePermutation:
    """Test adaptive permutation test."""
    
    def test_run_composite(self):
        """Test composite permutation run on data without drift."""
        np.random.seed(42)
        W_hist = np.random.normal(0, 1, 200)
        W_new = np.random.normal(0, 1, 200)  # Same distribution
        
        p_val, n_perms = AdaptivePermutationTest.run_composite(W_hist, W_new)
        
        assert 0 <= p_val <= 1
        assert n_perms > 0
        assert p_val > 0.05  # No drift expected
        
    def test_with_drift(self):
        """Test on data with drift."""
        np.random.seed(42)
        W_hist = np.random.normal(0, 1, 200)
        W_new = np.random.normal(1.5, 1, 200)  # Clear Shifted distribution
        
        p_val, n_perms = AdaptivePermutationTest.run_composite(W_hist, W_new)
        assert p_val < 0.05  # Drift detected

# [OOB Validator tests remain the same]


class TestOOBValidator:
    """Test out-of-bag validation."""
    
    def test_initialization(self):
        """Test initialization."""
        validator = OOBValidator(val_window_size=50)
        assert validator.val_window_size == 50
        assert len(validator.val_buffer) == 0
    
    def test_add_observation(self):
        """Test adding observations."""
        validator = OOBValidator(val_window_size=50)
        
        X = np.random.randn(5)
        validator.add_observation(X, y_true=1.0, y_pred_train=1.1)
        
        assert len(validator.val_buffer) == 1
        assert len(validator.train_errors) == 1
    
    def test_buffer_size_limit(self):
        """Test that buffer respects size limit."""
        validator = OOBValidator(val_window_size=10)
        
        for i in range(20):
            X = np.random.randn(5)
            validator.add_observation(X, 1.0, 1.0)
        
        assert len(validator.val_buffer) == 10
    
    def test_check_overfitting_no_overfit(self):
        """Test overfitting detection when none present."""
        validator = OOBValidator()
        
        # Good predictions on both sets
        for i in range(20):
            validator.train_errors.append(0.1)
            validator.val_errors.append(0.1)
        
        is_overfit, ratio = validator.check_overfitting()
        
        assert not is_overfit
        assert ratio == pytest.approx(1.0)
    
    def test_check_overfitting_with_overfit(self):
        """Test overfitting detection when overfit present."""
        validator = OOBValidator()
        
        # Good on training, bad on validation
        for i in range(20):
            validator.train_errors.append(0.05)
            validator.val_errors.append(0.15)
        
        is_overfit, ratio = validator.check_overfitting()
        
        assert is_overfit
        assert ratio > 2.0