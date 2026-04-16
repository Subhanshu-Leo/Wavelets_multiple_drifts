"""
Missing value handling with CORRECTED interpolation logic.

CRITICAL FIX: Use proper contextual interpolation, not global smoothing.
"""

import numpy as np
from scipy.interpolate import interp1d
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

class MissingValueHandler:
    """Handle missing values with context-aware interpolation."""
    
    def __init__(self, gap_threshold: int = 5, 
                 small_gap_method: str = 'linear',
                 large_gap_method: str = 'forward_fill'):
        """
        Initialize handler.
        
        Args:
            gap_threshold: Gap size threshold
            small_gap_method: 'linear', 'spline'
            large_gap_method: 'forward_fill', 'skip'
        """
        self.gap_threshold = gap_threshold
        self.small_gap_method = small_gap_method
        self.large_gap_method = large_gap_method
    
    def impute(self, X: np.ndarray, 
              missing_mask: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Impute missing values.
        
        FIXED: Use local context, not global smoothing.
        
        Args:
            X: Signal with NaN/inf
            missing_mask: Boolean array (True = missing)
        
        Returns:
            (X_imputed, metadata)
        """
        X_work = X.copy()
        imputed_indices = np.where(missing_mask)[0]
        
        if len(imputed_indices) == 0:
            return X, {'n_imputed': 0, 'method': 'none'}
        
        # Identify gaps
        gaps = self._identify_gaps(imputed_indices)
        
        metadata = {
            'n_imputed': len(imputed_indices),
            'n_gaps': len(gaps),
            'gaps': gaps,
            'methods_used': []
        }
        
        # Handle each gap
        for gap_start, gap_end in gaps:
            gap_size = gap_end - gap_start + 1
            
            if gap_size < self.gap_threshold:
                # Small gap: local linear interpolation
                X_work = self._interpolate_local(X_work, gap_start, gap_end)
                metadata['methods_used'].append(f"linear({gap_size})")
                logger.debug(f"Imputed gap [{gap_start}, {gap_end}] (size={gap_size}) using linear")
            else:
                # Large gap: forward fill with warning
                X_work = self._forward_fill(X_work, gap_start, gap_end)
                metadata['methods_used'].append(f"forward_fill({gap_size})")
                logger.warning(f"Imputed gap [{gap_start}, {gap_end}] (size={gap_size}) using forward_fill")
        
        logger.info(f"Imputed {len(imputed_indices)} missing values: {metadata}")
        
        return X_work, metadata
    
    @staticmethod
    def _identify_gaps(missing_indices: np.ndarray) -> list:
        """Identify contiguous gap regions."""
        if len(missing_indices) == 0:
            return []
        
        gaps = []
        gap_start = missing_indices[0]
        gap_end = missing_indices[0]
        
        for i in range(1, len(missing_indices)):
            if missing_indices[i] == gap_end + 1:
                gap_end = missing_indices[i]
            else:
                gaps.append((gap_start, gap_end))
                gap_start = missing_indices[i]
                gap_end = missing_indices[i]
        
        gaps.append((gap_start, gap_end))
        return gaps
    
    def _interpolate_local(self, X: np.ndarray, 
                          gap_start: int, gap_end: int) -> np.ndarray:
        """
        Interpolate using neighboring values (LOCAL context).
        """
        X_work = X.copy()
        
        # Find available points around gap
        before_idx = np.where(~np.isnan(X[:gap_start]))[0]
        after_idx = np.where(~np.isnan(X[gap_end+1:]))[0] + (gap_end + 1)
        
        if len(before_idx) == 0 or len(after_idx) == 0:
            # Not enough context: forward fill
            return self._forward_fill(X_work, gap_start, gap_end)
        
        # Interpolate between last before and first after
        x_before = before_idx[-1]
        x_after = after_idx[0]
        
        y_before = X[x_before]
        y_after = X[x_after]
        
        # Linear interpolation
        interp_indices = np.arange(gap_start, gap_end + 1)
        interp_values = y_before + (y_after - y_before) * (interp_indices - x_before) / (x_after - x_before)
        
        X_work[gap_start:gap_end+1] = interp_values
        
        return X_work
    
    @staticmethod
    def _forward_fill(X: np.ndarray, 
                     gap_start: int, gap_end: int) -> np.ndarray:
        """Forward-fill large gaps."""
        X_work = X.copy()
        
        before_idx = np.where(~np.isnan(X[:gap_start]))[0]
        if len(before_idx) > 0:
            last_value = X[before_idx[-1]]
            X_work[gap_start:gap_end+1] = last_value
        
        return X_work


# Example
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    X = np.sin(2*np.pi*np.arange(500)/50)
    
    # Create missing values
    missing_mask = np.zeros(500, dtype=bool)
    missing_mask[[50, 51, 52, 200, 201, 350]] = True  # Mix of small and large gaps
    X_missing = X.copy()
    X_missing[missing_mask] = np.nan
    
    handler = MissingValueHandler(gap_threshold=5)
    X_imputed, metadata = handler.impute(X_missing, missing_mask)
    
    print(f" Imputed {metadata['n_imputed']} values")
    print(f"  Methods: {metadata['methods_used']}")
    
    mse = np.mean((X_imputed[missing_mask] - X[missing_mask])**2)
    print(f"  MSE at missing locations: {mse:.4f}")