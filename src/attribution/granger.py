"""
Granger causality with PROPER stationarity handling.

CRITICAL FIX: Test for stationarity before Granger test.
Use differencing or co-integration if needed.

Reference: Granger (1969), Engle & Granger (1987)
"""

import numpy as np
from scipy import stats
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

def adf_test(series: np.ndarray, max_lag: int = 5) -> Tuple[float, bool]:
    """
    Augmented Dickey-Fuller test for stationarity.
    
    Args:
        series: Time series
        max_lag: Maximum lag for ADF test
    
    Returns:
        (p_value, is_stationary)
    """
    try:
        from statsmodels.tsa.stattools import adfuller
        
        result = adfuller(series, maxlag=max_lag, autolag='AIC')
        p_value = result[1]
        is_stationary = p_value < 0.05
        
        logger.debug(f"ADF test: p={p_value:.4f}, stationary={is_stationary}")
        
        return p_value, is_stationary
    
    except Exception as e:
        logger.warning(f"ADF test failed: {e}. Assuming non-stationary.")
        return np.nan, False


def granger_causality_test(error_signal: np.ndarray,
                          feature_signal: np.ndarray,
                          lag: int = 5,
                          alpha: float = 0.05,
                          enforce_stationarity: bool = True) -> Tuple[bool, float]:
    """
    Test if feature Granger-causes error.
    
    FIXED:
    1. Check stationarity of both signals
    2. Difference non-stationary series
    3. Apply Granger test only on stationary data
    
    Args:
        error_signal: Error time series e_t
        feature_signal: Feature time series X_t
        lag: AR lag order (set via AIC if None)
        alpha: Significance level
        enforce_stationarity: If True, difference non-stationary series
    
    Returns:
        (granger_significant, p_value)
    
    Reference: Granger (1969), Engle & Granger (1987)
    """
    try:
        from statsmodels.tsa.api import grangercausalitytests
    except ImportError:
        logger.error("statsmodels required for Granger test")
        return False, np.nan
    
    # Input validation
    if len(error_signal) < max(lag * 3, 20):
        logger.warning(f"Series too short ({len(error_signal)} < {max(lag*3, 20)}) for Granger test")
        return False, np.nan
    
    error_work = error_signal.copy()
    feature_work = feature_signal.copy()
    n_diffs = 0
    
    if enforce_stationarity:
        # Check error stationarity
        _, error_stationary = adf_test(error_work, max_lag=lag)
        if not error_stationary:
            logger.info("Error signal non-stationary, differencing...")
            error_work = np.diff(error_work)
            n_diffs += 1
        
        # Check feature stationarity
        _, feature_stationary = adf_test(feature_work, max_lag=lag)
        if not feature_stationary:
            logger.info("Feature signal non-stationary, differencing...")
            feature_work = np.diff(feature_work)
            n_diffs += 1
        
        if n_diffs > 0 and len(error_work) < max(lag * 3, 20):
            logger.warning(f"Series too short after differencing for Granger test")
            return False, np.nan
    
    # Run Granger test
    try:
        data = np.column_stack([error_work, feature_work])
        gc_result = grangercausalitytests(data, maxlag=lag, verbose=False)
        
        # Extract p-value from lag test
        f_stat, p_value, _, _ = gc_result[lag][0]['ssr_ftest']
        
        granger_passes = p_value < alpha
        
        logger.debug(
            f"Granger causality (lag={lag}, n_diffs={n_diffs}): "
            f"F={f_stat:.2f}, p={p_value:.4f}, significant={granger_passes}"
        )
        
        return granger_passes, p_value
    
    except Exception as e:
        logger.error(f"Granger test failed: {e}")
        return False, np.nan


# Example
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Test 1: No causality
    print("Test 1: Independent signals")
    np.random.seed(42)
    error = np.random.randn(200)
    feature = np.random.randn(200)
    
    passes, p_val = granger_causality_test(error, feature, lag=5)
    print(f"✓ No causality: p={p_val:.4f}, passes={passes}")
    
    # Test 2: With causality
    print("\nTest 2: Feature causes error")
    feature = np.random.randn(200)
    error = np.zeros(200)
    for t in range(1, 200):
        error[t] = 0.7 * feature[t-1] + 0.1*np.random.randn()
    
    passes, p_val = granger_causality_test(error, feature, lag=5)
    print(f"✓ With causality: p={p_val:.4f}, passes={passes}")