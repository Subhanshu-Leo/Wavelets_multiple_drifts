"""
Synthetic data generator for testing concept drift detection.
"""

import numpy as np
from typing import Tuple, List

class SyntheticDriftGenerator:
    def __init__(self, seed: int = 42):
        self.seed = seed

    def generate(self, drift_type: str = 'sudden', n: int = 5000, n_features: int = 5) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        np.random.seed(self.seed)
        X = np.random.randn(n, n_features)
        y = np.zeros(n)
        
        # Base relationship (Concept A)
        for t in range(n):
            y[t] = np.sum(X[t]) + 0.1 * np.random.randn()

        true_drifts = []

        if drift_type == 'sudden':
            drift_t = n // 2
            true_drifts = [drift_t]
            for t in range(drift_t, n):
                y[t] += 5.0
                
        elif drift_type == 'variance':
            drift_t = n // 2
            true_drifts = [drift_t]
            for t in range(drift_t, n):
                y[t] = np.sum(X[t]) + 1.0 * np.random.randn()
                
        elif drift_type == 'gradual':
            start_drift = int(n * 0.4)
            end_drift = int(n * 0.6)
            true_drifts = [start_drift]
            
            for t in range(start_drift, end_drift):
                progress = (t - start_drift) / (end_drift - start_drift)
                y[t] += progress * 4.0
            for t in range(end_drift, n):
                y[t] += 4.0
                
        elif drift_type == 'incremental':
            step_size = n // 5
            for i in range(1, 4):
                drift_t = i * step_size
                true_drifts.append(drift_t)
                # FIXED: End boundary ensures we don't overlap additions
                end_t = (i + 1) * step_size if i < 3 else n
                for t in range(drift_t, end_t):
                    y[t] += (i * 1.5)  # Clean steps of 1.5, 3.0, 4.5
                    
        elif drift_type == 'recurring':
            step_size = n // 4
            for i in range(1, 4):
                drift_t = i * step_size
                true_drifts.append(drift_t)
                if i % 2 != 0:
                    for t in range(drift_t, min(drift_t + step_size, n)):
                        y[t] += 3.0  # Concept B
                # FIXED: removed the confusing 'else: pass'. 
                # Concept A naturally resumes because we only add to the base y array.

        return X, y, true_drifts