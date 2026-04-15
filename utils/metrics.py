"""
Evaluation metrics for drift detection.
"""

import numpy as np
from scipy import stats

class DriftMetrics:
    """Compute drift detection evaluation metrics."""
    
    @staticmethod
    def detection_delay(true_drift_time: int,
                       detected_drift_time: int) -> int:
        """
        Compute detection delay (in steps).
        
        Args:
            true_drift_time: When drift actually occurred
            detected_drift_time: When drift was detected
        
        Returns:
            Delay in steps
        """
        if detected_drift_time is None:
            return float('inf')
        
        return max(0, detected_drift_time - true_drift_time)
    
    @staticmethod
    def false_alarm_rate(n_alarms: int,
                        n_stationary_steps: int) -> float:
        """
        Compute false alarm rate on stationary data.
        
        Args:
            n_alarms: Number of alarms on stationary data
            n_stationary_steps: Length of stationary data
        
        Returns:
            FAR (0-1)
        """
        return n_alarms / (n_stationary_steps + 1e-10)
    
    @staticmethod
    def prequential_error(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         metric: str = 'rmse') -> float:
        """
        Compute prequential (online) error.
        
        Args:
            y_true: True values
            y_pred: Predictions
            metric: 'rmse', 'mae', 'mse'
        
        Returns:
            Error value
        """
        errors = np.abs(y_true - y_pred)
        
        if metric == 'rmse':
            return np.sqrt(np.mean(errors ** 2))
        elif metric == 'mae':
            return np.mean(errors)
        elif metric == 'mse':
            return np.mean(errors ** 2)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    @staticmethod
    def recall_precision(true_drifts: np.ndarray,
                        detected_drifts: np.ndarray,
                        window: int = 20) -> tuple:
        """
        Compute recall and precision for synthetic data.
        
        A detection is a TP if within 'window' steps of true drift.
        
        Args:
            true_drifts: Indices of true drifts
            detected_drifts: Indices of detected drifts
            window: Time window for TP definition
        
        Returns:
            (recall, precision)
        """
        if len(true_drifts) == 0:
            return 0.0, 1.0 if len(detected_drifts) == 0 else 0.0
        
        if len(detected_drifts) == 0:
            return 0.0, 0.0
        
        # Count TPs
        tp = 0
        for true_t in true_drifts:
            if any(abs(detected_t - true_t) <= window
                   for detected_t in detected_drifts):
                tp += 1
        
        recall = tp / len(true_drifts)
        precision = tp / len(detected_drifts)
        
        return recall, precision
    
    @staticmethod
    def wilcoxon_test(errors1: np.ndarray,
                     errors2: np.ndarray,
                     alpha: float = 0.05) -> tuple:
        """
        Wilcoxon signed-rank test comparing two error series.
        
        Args:
            errors1: Error series 1
            errors2: Error series 2
            alpha: Significance level
        
        Returns:
            (is_significant, p_value, statistic)
        
        Reference: Demšar (2006)
        """
        stat, p_value = stats.wilcoxon(errors1, errors2)
        
        is_significant = p_value < alpha
        
        return is_significant, p_value, stat


# Example
if __name__ == '__main__':
    y_true = np.sin(2*np.pi*np.arange(500)/50)
    y_pred = y_true + 0.1*np.random.randn(500)
    
    rmse = DriftMetrics.prequential_error(y_true, y_pred, metric='rmse')
    print(f"RMSE: {rmse:.4f}")
    
    true_drifts = np.array([1000, 2500])
    detected_drifts = np.array([1015, 2510])
    
    recall, prec = DriftMetrics.recall_precision(true_drifts, detected_drifts)
    print(f"Recall: {recall:.2f}, Precision: {prec:.2f}")