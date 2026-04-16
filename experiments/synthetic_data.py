"""
Synthetic data generator for testing concept drift detection.
Includes sudden mean shifts, variance shifts (volatility), gradual, and incremental drifts.
"""

import numpy as np
from typing import Tuple, List

class SyntheticDriftGenerator:
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(self.seed)

    def generate(self, drift_type: str = 'sudden', n: int = 5000, n_features: int = 5) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Generate synthetic streaming data with specific drift types.
        
        Args:
            drift_type: 'sudden', 'variance', 'gradual', 'incremental', or 'recurring'
            n: Total stream length
            n_features: Number of input features
            
        Returns:
            X (features), y (target), true_drifts (list of drift indices)
        """
        np.random.seed(self.seed)
        X = np.random.randn(n, n_features)
        y = np.zeros(n)
        
        # Base relationship: y is sum of features + noise
        for t in range(n):
            y[t] = np.sum(X[t]) + 0.1 * np.random.randn()

        true_drifts = []

        if drift_type == 'sudden':
            # Single massive mean shift at t=2500
            drift_t = n // 2
            true_drifts = [drift_t]
            for t in range(drift_t, n):
                y[t] += 5.0  # +5.0 mean shift
                
        elif drift_type == 'variance':
            # Volatility explosion at t=2500 (mean stays same, variance 10x)
            drift_t = n // 2
            true_drifts = [drift_t]
            for t in range(drift_t, n):
                y[t] = np.sum(X[t]) + 1.0 * np.random.randn() # Base noise was 0.1, now 1.0
                
        elif drift_type == 'gradual':
            # Slowly shifts from base concept to new concept between t=2000 and t=3000
            start_drift = int(n * 0.4)
            end_drift = int(n * 0.6)
            true_drifts = [start_drift]
            
            for t in range(start_drift, end_drift):
                progress = (t - start_drift) / (end_drift - start_drift)
                y[t] += progress * 4.0  # Gradually shifts up to +4.0
            for t in range(end_drift, n):
                y[t] += 4.0
                
        elif drift_type == 'incremental':
            # Small incremental step changes
            step_size = n // 5
            for i in range(1, 4):
                drift_t = i * step_size
                true_drifts.append(drift_t)
                for t in range(drift_t, drift_t + step_size if i < 3 else n):
                    y[t] += (i * 1.5)  # Steps of 1.5, 3.0, 4.5
                    
        elif drift_type == 'recurring':
            # Flips back and forth between Concept A and Concept B
            step_size = n // 4
            for i in range(1, 4):
                drift_t = i * step_size
                true_drifts.append(drift_t)
                if i % 2 != 0:
                    for t in range(drift_t, min(drift_t + step_size, n)):
                        y[t] += 3.0  # Concept B
                else:
                    for t in range(drift_t, min(drift_t + step_size, n)):
                        pass # Back to Concept A (base)

        return X, y, true_drifts