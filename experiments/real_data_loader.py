"""
Load and prepare real-world financial data for experiments.
"""

import numpy as np
import pandas as pd
from utils.io import FinancialDataLoader, ElectricityDataLoader
from typing import Tuple

class RealDataExperiment:
    """Manage real-world data for drift detection experiments."""
    
    @staticmethod
    def prepare_sp500(filepath: str = None,
                     train_ratio: float = 0.2) -> Tuple[dict, dict, dict]:
        """
        Prepare S&P 500 data for streaming experiment.
        
        Args:
            filepath: Path to CSV data
            train_ratio: Fraction for warm-up
        
        Returns:
            (X_warm, y_warm), (X_eval, y_eval), metadata
        """
        print("Loading S&P 500 data...")
        X, y = FinancialDataLoader.prepare_sp500_data(filepath)
        
        n = len(X)
        n_train = int(train_ratio * n)
        
        X_warm = X[:n_train]
        y_warm = y[:n_train]
        
        X_eval = X[n_train:]
        y_eval = y[n_train:]
        
        metadata = {
            'source': 'S&P 500',
            'total_length': n,
            'warm_up_size': n_train,
            'eval_size': len(X_eval),
            'n_features': X.shape[1],
            'date_range': '2015-2023'
        }
        
        print(f"✓ Loaded {n} samples")
        print(f"✓ Warm-up: {n_train}, Evaluation: {len(X_eval)}")
        print(f"✓ Features: {X.shape[1]}")
        
        return (X_warm, y_warm), (X_eval, y_eval), metadata
    
    @staticmethod
    def prepare_electricity(filepath: str,
                          train_ratio: float = 0.2) -> Tuple[dict, dict, dict]:
        """
        Prepare Electricity Load data for streaming experiment.
        
        Args:
            filepath: Path to CSV data
            train_ratio: Fraction for warm-up
        
        Returns:
            (X_warm, y_warm), (X_eval, y_eval), metadata
        """
        print("Loading Electricity Load data...")
        X, y = ElectricityDataLoader.load_electricity(filepath)
        
        n = len(X)
        n_train = int(train_ratio * n)
        
        X_warm = X[:n_train]
        y_warm = y[:n_train]
        
        X_eval = X[n_train:]
        y_eval = y[n_train:]
        
        metadata = {
            'source': 'UCI Electricity',
            'total_length': n,
            'warm_up_size': n_train,
            'eval_size': len(X_eval),
            'n_features': X.shape[1],
            'frequency': 'hourly'
        }
        
        print(f"✓ Loaded {n} samples")
        print(f"✓ Warm-up: {n_train}, Evaluation: {len(X_eval)}")
        print(f"✓ Features: {X.shape[1]}")
        
        return (X_warm, y_warm), (X_eval, y_eval), metadata
    
    @staticmethod
    def create_streaming_batches(X: np.ndarray, y: np.ndarray,
                                batch_size: int = 1) -> list:
        """
        Convert data into streaming batches.
        
        Args:
            X: Feature matrix
            y: Target vector
            batch_size: Samples per batch (usually 1 for streaming)
        
        Returns:
            List of (X_batch, y_batch) tuples
        """
        batches = []
        
        for i in range(0, len(X) - batch_size + 1, batch_size):
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            batches.append((X_batch, y_batch))
        
        return batches


# Example
if __name__ == '__main__':
    # Example: prepare S&P 500 (if file available)
    try:
        (X_warm, y_warm), (X_eval, y_eval), metadata = \
            RealDataExperiment.prepare_sp500(filepath='data/sp500.csv')
        
        print("\nMetadata:")
        for key, val in metadata.items():
            print(f"  {key}: {val}")
    
    except Exception as e:
        print(f"Could not load S&P 500 data: {e}")
        print("(Make sure 'data/sp500.csv' exists or download from Yahoo Finance)")