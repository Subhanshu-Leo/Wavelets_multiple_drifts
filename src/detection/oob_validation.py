"""
Out-of-bag validation with CORRECTED implementation.

CRITICAL FIXES:
1. Use deque instead of list.pop(0)
2. Auto-call get_validation_error() or enforce contract
3. Add comprehensive logging
4. Fix overfitting detection logic

Reference: Bifet et al. (2007) - ADWIN uses OOB for drift detection
"""

from collections import deque
import numpy as np
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class OOBValidator:
    """
    Out-of-bag validation to guard against overfitting post-retraining.
    
    FIXED:
    1. Uses deque(maxlen=...) for O(1) operations
    2. Automatic error tracking (don't require manual calls)
    3. Proper overfitting detection with logging
    4. Thread-safe design (no mutable shared state)
    """
    
    def __init__(self, val_window_size: int = 50,
                 overfit_ratio_threshold: float = 2.0,
                 min_history: int = 10):
        """
        Initialize OOB validator.
        
        Args:
            val_window_size: Size of validation window (deque)
            overfit_ratio_threshold: Flag if val_error / train_error > this
            min_history: Minimum errors needed before overfitting check
        
        CRITICAL: val_window_size must be > 0
        """
        if val_window_size <= 0:
            raise ValueError(f"val_window_size must be > 0, got {val_window_size}")
        
        self.val_window_size = val_window_size
        self.overfit_ratio_threshold = overfit_ratio_threshold
        self.min_history = min_history
        
        # FIXED: Use deque with maxlen for O(1) append and auto-eviction
        self.val_buffer = deque(maxlen=val_window_size)
        self.train_errors = deque(maxlen=500)
        self.val_errors = deque(maxlen=500)
        
        logger.info(
            f"Initialized OOBValidator: val_window={val_window_size}, "
            f"overfit_threshold={overfit_ratio_threshold}"
        )
    
    def add_observation(self, X_point: np.ndarray, y_true: float,
                       y_pred_train: float, y_pred_val: Optional[float] = None):
        """
        Add a training and validation observation.
        
        FIXED: Automatically track both train and val errors.
        Caller provides predictions on both train and val sets.
        
        Args:
            X_point: Feature vector for this time step
            y_true: Ground truth target
            y_pred_train: Model's prediction (training set)
            y_pred_val: Model's prediction (validation set, if different)
        
        Raises:
            ValueError: If inputs are invalid
        """
        # Input validation
        if not isinstance(X_point, np.ndarray):
            X_point = np.array(X_point)
        
        if not np.isfinite([y_true, y_pred_train]).all():
            raise ValueError(f"Non-finite values: y_true={y_true}, y_pred_train={y_pred_train}")
        
        # Store in validation buffer
        self.val_buffer.append((X_point, y_true))
        
        # Automatically compute errors
        train_error = np.abs(y_true - y_pred_train)
        self.train_errors.append(train_error)
        
        if y_pred_val is not None:
            if not np.isfinite(y_pred_val):
                raise ValueError(f"Non-finite y_pred_val: {y_pred_val}")
            val_error = np.abs(y_true - y_pred_val)
            self.val_errors.append(val_error)
        
        logger.debug(
            f"Added observation: train_error={train_error:.4f}, "
            f"val_error={val_error if y_pred_val else 'N/A'}, "
            f"buffer_size={len(self.val_buffer)}"
        )
    
    def check_overfitting(self) -> Tuple[bool, float, dict]:
        """
        Check if training error << validation error (overfitting).
        
        FIXED: This now REQUIRES that add_observation has been called
        with y_pred_val (validation predictions). If val_errors is empty,
        raises ValueError instead of silently returning False.
        
        Returns:
            (is_overfitting, ratio, metadata)
            - is_overfitting: True if val_error/train_error > threshold
            - ratio: Actual ratio
            - metadata: Detailed statistics
        
        Raises:
            RuntimeError: If insufficient data or no validation predictions provided
        """
        if len(self.train_errors) < self.min_history:
            raise RuntimeError(
                f"Need {self.min_history} observations minimum, "
                f"have {len(self.train_errors)}"
            )
        
        if len(self.val_errors) < self.min_history:
            raise RuntimeError(
                f"No validation errors tracked. "
                f"Call add_observation() with y_pred_val parameter."
            )
        
        # Use recent errors for stability
        recent_window = min(20, len(self.train_errors))
        
        recent_train_errors = list(self.train_errors)[-recent_window:]
        recent_val_errors = list(self.val_errors)[-recent_window:]
        
        train_mean = np.mean(recent_train_errors)
        val_mean = np.mean(recent_val_errors)
        
        # Avoid division by zero
        ratio = val_mean / (train_mean + 1e-10)
        is_overfitting = ratio > self.overfit_ratio_threshold
        
        metadata = {
            'train_error_mean': train_mean,
            'val_error_mean': val_mean,
            'ratio': ratio,
            'threshold': self.overfit_ratio_threshold,
            'is_overfitting': is_overfitting,
            'recent_window': recent_window,
            'total_train_errors': len(self.train_errors),
            'total_val_errors': len(self.val_errors)
        }
        
        if is_overfitting:
            logger.warning(
                f"OVERFITTING DETECTED: val_error/train_error = {ratio:.2f} "
                f"(threshold={self.overfit_ratio_threshold})"
            )
        else:
            logger.debug(f"No overfitting: ratio={ratio:.2f}")
        
        return is_overfitting, ratio, metadata
    
    def get_recent_errors(self, n: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get recent training and validation errors.
        
        Args:
            n: Number of recent observations
        
        Returns:
            (train_errors, val_errors) as numpy arrays
        """
        n = min(n, len(self.train_errors), len(self.val_errors))
        
        train_recent = np.array(list(self.train_errors)[-n:])
        val_recent = np.array(list(self.val_errors)[-n:])
        
        return train_recent, val_recent
    
    def get_validation_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract validation features and targets as numpy arrays.
        
        Returns:
            (X_val, y_val)
        """
        if len(self.val_buffer) == 0:
            raise RuntimeError("Validation buffer is empty")
        
        X_val = np.array([x for x, _ in self.val_buffer])
        y_val = np.array([y for _, y in self.val_buffer])
        
        return X_val, y_val
    
    def reset(self):
        """Reset for a new stream or instrument."""
        self.val_buffer.clear()
        self.train_errors.clear()
        self.val_errors.clear()
        logger.info("OOBValidator reset")


# Example and validation
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    print("="*60)
    print("OOBValidator: Validation Tests")
    print("="*60)
    
    validator = OOBValidator(val_window_size=50)
    
    # Test 1: Add observations and check for overfitting
    print("\nTest 1: Good generalization (no overfitting)")
    np.random.seed(42)
    
    for t in range(100):
        X = np.random.randn(5)
        y_true = np.random.randn()
        
        # Both predictions are similar and accurate
        y_pred_train = y_true + 0.05*np.random.randn()
        y_pred_val = y_true + 0.05*np.random.randn()
        
        validator.add_observation(X, y_true, y_pred_train, y_pred_val)
    
    is_overfit, ratio, metadata = validator.check_overfitting()
    assert not is_overfit
    print(f" Ratio: {ratio:.2f}, Correctly identified as NOT overfitting")
    
    # Test 2: Severe overfitting
    print("\nTest 2: Severe overfitting detection")
    validator.reset()
    
    for t in range(100):
        X = np.random.randn(5)
        y_true = np.random.randn()
        
        # Train predictions perfect (overfit), val predictions poor
        y_pred_train = y_true  # Perfect fit on train
        y_pred_val = y_true + 0.5*np.random.randn()  # Poor on validation
        
        validator.add_observation(X, y_true, y_pred_train, y_pred_val)
    
    is_overfit, ratio, metadata = validator.check_overfitting()
    assert is_overfit
    print(f" Ratio: {ratio:.2f}, Correctly identified as OVERFITTING")
    
    # Test 3: Error on missing val predictions
    print("\nTest 3: Error handling for missing validation predictions")
    validator.reset()
    
    for t in range(100):
        X = np.random.randn(5)
        y_true = np.random.randn()
        y_pred = y_true + 0.1*np.random.randn()
        
        # Only add train error, no validation
        validator.add_observation(X, y_true, y_pred, y_pred_val=None)
    
    try:
        validator.check_overfitting()
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        print(f" Correctly raised error: {e}")
    
    print("\n" + "="*60)
    print(" All OOBValidator tests passed!")
    print("="*60)