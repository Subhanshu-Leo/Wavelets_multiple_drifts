"""
Staged retraining validation with CORRECTED import order and deque usage.

CRITICAL FIXES:
1. Move imports to top
2. Use deque instead of list.pop(0)
3. Add logging
"""

from typing import Tuple
from collections import deque
import numpy as np
import logging

logger = logging.getLogger(__name__)

class StagedRetrainingValidator:
    """Validate retrained learners before deployment."""
    
    def __init__(self, val_window_size: int = 50,
                 improvement_threshold: float = 0.10):
        """
        Initialize validator.
        
        Args:
            val_window_size: Validation window size
            improvement_threshold: Min improvement to deploy
        """
        self.val_window_size = val_window_size
        self.improvement_threshold = improvement_threshold
        
        # FIXED: Use deque with maxlen instead of list.pop(0)
        self.val_buffer = deque(maxlen=val_window_size)
        self.train_errors = deque(maxlen=500)  # Keep last 500 for analysis
    
    def add_val_observation(self, X_point: np.ndarray, 
                           y_true: float):
        """Buffer validation observation."""
        self.val_buffer.append((X_point, y_true))
        logger.debug(f"Added validation observation (buffer size: {len(self.val_buffer)})")
    
    def add_train_error(self, error: float):
        """Track training error."""
        self.train_errors.append(error)
    
    def validate_retrained_learner(self, learner_old,
                                   learner_new) -> Tuple[str, float, dict]:
        """
        Compare old vs new learner on validation data.
        
        Args:
            learner_old: Previous learner
            learner_new: Retrained learner
        
        Returns:
            (decision, improvement, metrics)
            - decision: 'deploy', 'inconclusive', 'reject'
            - improvement: Relative improvement
            - metrics: Detailed metrics dict
        """
        if len(self.val_buffer) < self.val_window_size // 2:
            logger.warning(f"Insufficient validation data ({len(self.val_buffer)} < {self.val_window_size//2})")
            return 'inconclusive', np.nan, {'reason': 'insufficient_data'}
        
        # Extract validation data
        X_val = np.array([x for x, _ in self.val_buffer])
        y_val = np.array([y for _, y in self.val_buffer])
        
        # Get predictions
        try:
            pred_old = learner_old.predict(X_val)
            pred_new = learner_new.predict(X_val)
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return 'inconclusive', np.nan, {'reason': 'prediction_error', 'error': str(e)}
        
        # Compute MSE
        mse_old = np.mean((y_val - pred_old) ** 2)
        mse_new = np.mean((y_val - pred_new) ** 2)
        
        # Improvement
        improvement = (mse_old - mse_new) / (mse_old + 1e-10)
        
        metrics = {
            'mse_old': mse_old,
            'mse_new': mse_new,
            'improvement': improvement,
            'val_size': len(self.val_buffer),
            'improvement_threshold': self.improvement_threshold
        }
        
        # Decision rule
        if improvement > self.improvement_threshold:
            decision = 'deploy'
            logger.info(f"Decision: DEPLOY (improvement={improvement:.2%})")
        elif improvement < -0.05:  # New is worse
            decision = 'reject'
            logger.warning(f"Decision: REJECT (degradation={-improvement:.2%})")
        else:
            decision = 'inconclusive'
            logger.info(f"Decision: INCONCLUSIVE (improvement={improvement:.2%})")
        
        return decision, improvement, metrics