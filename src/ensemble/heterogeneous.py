"""
Heterogeneous ensemble with CORRECTED learned weighting and random_state handling.

CRITICAL FIXES:
1. Per-scale learned weights (not fixed equal 1/3)
2. random_state passed as parameter (not hardcoded)
3. Proper initialization
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
import logging
from typing import Dict,Optional

logger = logging.getLogger(__name__)


class HeterogeneousEnsemble:
    """
    Ensemble of three diverse learners with learned combination.
    
    FIXED: Weights are learned per-scale via validation error,
    not fixed equal averages.
    """
    
    def __init__(self, random_state: Optional[int] = None):
        """
        Initialize heterogeneous ensemble.
        
        Args:
            random_state: Random seed for reproducibility
                         (None = different each instance)
        """
        self.random_state = random_state
        
        # Three diverse learners
        self.rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=random_state
        )
        
        self.gb = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=random_state
        )
        
        self.ridge = Ridge(alpha=1.0)
        
        # Learned weights (not fixed 1/3)
        self.weights = None
        self.is_trained = False
        
        logger.debug(f"Initialized HeterogeneousEnsemble (random_state={random_state})")
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            val_fraction: float = 0.2) -> None:
        """
        Train all three learners and learn optimal weights.
        
        CRITICAL FIX: Weights are learned from validation performance,
        not fixed to 1/3.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training targets (n_samples,)
            val_fraction: Fraction for validation set
        """
        if len(X) < 20:
            raise ValueError(f"Need at least 20 samples, got {len(X)}")
        
        # Split train/val
        n_val = max(10, int(len(X) * val_fraction))
        n_train = len(X) - n_val
        
        X_train, X_val = X[:n_train], X[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]
        
        # Train all three learners
        logger.debug("Training heterogeneous base learners...")
        self.rf.fit(X_train, y_train)
        self.gb.fit(X_train, y_train)
        self.ridge.fit(X_train, y_train)
        
        # Learn weights from validation error
        pred_rf_val = self.rf.predict(X_val)
        pred_gb_val = self.gb.predict(X_val)
        pred_ridge_val = self.ridge.predict(X_val)
        
        mse_rf = np.mean((y_val - pred_rf_val) ** 2)
        mse_gb = np.mean((y_val - pred_gb_val) ** 2)
        mse_ridge = np.mean((y_val - pred_ridge_val) ** 2)
        
        # CRITICAL FIX: Weights proportional to inverse MSE
        weight_rf = 1.0 / (mse_rf + 1e-10)
        weight_gb = 1.0 / (mse_gb + 1e-10)
        weight_ridge = 1.0 / (mse_ridge + 1e-10)
        
        # Normalize
        total = weight_rf + weight_gb + weight_ridge
        self.weights = {
            'rf': weight_rf / total,
            'gb': weight_gb / total,
            'ridge': weight_ridge / total
        }
        
        logger.info(
            f"Learned weights: RF={self.weights['rf']:.3f}, "
            f"GB={self.weights['gb']:.3f}, Ridge={self.weights['ridge']:.3f}"
        )
        
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using weighted average of three learners.
        
        CRITICAL FIX: Uses learned weights, not fixed 1/3.
        """
        if not self.is_trained:
            raise RuntimeError("Ensemble not trained. Call fit() first.")
        
        pred_rf = self.rf.predict(X)
        pred_gb = self.gb.predict(X)
        pred_ridge = self.ridge.predict(X)
        
        # CRITICAL FIX: Use learned weights
        prediction = (
            self.weights['rf'] * pred_rf +
            self.weights['gb'] * pred_gb +
            self.weights['ridge'] * pred_ridge
        )
        
        return prediction


class MultiResolutionEnsemble:
    """
    Multi-resolution ensemble with heterogeneous base learners.
    
    FIXED: Explicit scale validation, per-scale learned weights.
    """
    
    def __init__(self, J: int = 4, require_all_scales: bool = True,
                 random_state: Optional[int] = None):
        """
        Initialize multi-resolution ensemble.
        
        Args:
            J: Number of decomposition levels
            require_all_scales: If True, raise error if any scale missing
            random_state: Random seed
        """
        self.J = J
        self.require_all_scales = require_all_scales
        self.random_state = random_state
        
        # Per-scale heterogeneous ensemble
        self.learners = {
            j: HeterogeneousEnsemble(random_state=random_state)
            for j in range(J + 1)
        }
        
        self.initial_weights = {}
        self.is_trained = False
        
        logger.info(
            f"Initialized MultiResolutionEnsemble: J={J}, "
            f"require_all_scales={require_all_scales}, random_state={random_state}"
        )
    
    def fit(self, X_dict: Dict[int, np.ndarray],
           y_dict: Dict[int, np.ndarray],
           val_size: int = 100) -> None:
        """
        Train ensemble on multi-scale data.
        
        FIXED: Validates all scales present upfront.
        """
        expected_scales = set(range(self.J + 1))
        actual_scales = set(X_dict.keys()) & set(y_dict.keys())
        missing_scales = expected_scales - actual_scales
        
        if missing_scales:
            if self.require_all_scales:
                raise ValueError(f"Missing scales (require_all_scales=True): {sorted(missing_scales)}")
            else:
                logger.warning(f"Missing scales: {sorted(missing_scales)}")
        
        # Train per-scale
        self.initial_weights = {}
        
        for j in sorted(actual_scales):
            X = X_dict[j]
            y = y_dict[j]
            
            logger.debug(f"Training scale {j}: X{X.shape}, y{y.shape}")
            
            self.learners[j].fit(X, y, val_fraction=min(0.2, val_size / len(X)))
            
            # Compute weight from validation
            X_val = X[-val_size:] if len(X) > val_size else X
            y_val = y[-val_size:] if len(y) > val_size else y
            
            y_pred_val = self.learners[j].predict(X_val)
            mse_val = np.mean((y_val - y_pred_val) ** 2)
            
            self.initial_weights[j] = 1.0 / (mse_val + 1e-10)
        
        # Normalize weights
        total = sum(self.initial_weights.values())
        self.initial_weights = {j: w / total for j, w in self.initial_weights.items()}
        
        self.is_trained = True
        
        logger.info(
            f"Ensemble trained: {len(self.initial_weights)} scales, "
            f"weights={self.initial_weights}"
        )
    
    def predict(self, X_dict: Dict[int, np.ndarray],
               weights: Optional[Dict[int, float]] = None) -> np.ndarray:
        """Make ensemble prediction."""
        if not self.is_trained:
            raise RuntimeError("Ensemble not trained. Call fit() first.")
        
        if weights is None:
            weights = self.initial_weights
        
        # Get the target length (minimum across all scales)
        lengths = [X_dict[j].shape[0] for j in X_dict.keys() if j in X_dict]
        
        if not lengths:
            raise ValueError("X_dict is empty or has no valid scales")
        
        target_len = min(lengths)
        
        # REMOVED: Don't warn on shape mismatches - they're expected with DWT
        # (Different scales have different lengths after decomposition)
        
        weighted_pred = np.zeros(target_len)
        
        for j in sorted(weights.keys()):
            if j not in X_dict:
                continue
            
            if j not in self.learners:
                continue
            
            X_j = X_dict[j]
            
            # Truncate or pad to target length
            if X_j.shape[0] > target_len:
                X_j = X_j[:target_len, :]
            elif X_j.shape[0] < target_len:
                pad_rows = target_len - X_j.shape[0]
                X_j = np.vstack([X_j, np.tile(X_j[-1, :], (pad_rows, 1))])
            
            try:
                pred_j = self.learners[j].predict(X_j)
                
                if len(pred_j) != target_len:
                    pred_j = pred_j[:target_len] if len(pred_j) > target_len else \
                            np.pad(pred_j, (0, target_len - len(pred_j)), mode='edge')
                
                weighted_pred += weights[j] * pred_j
                
            except Exception as e:
                logger.error(f"Prediction failed at scale {j}: {e}")
                continue
        
        return weighted_pred


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    from typing import Dict
    
    print("="*80)
    print("Heterogeneous Ensemble: Learned Weights")
    print("="*80)
    
    np.random.seed(42)
    
    # Test 1: Heterogeneous ensemble with learned weights
    print("\n[TEST 1] Learned Weights (Not Fixed 1/3)")
    X_train = np.random.randn(100, 5)
    y_train = np.sum(X_train, axis=1) + 0.1*np.random.randn(100)
    
    ensemble = HeterogeneousEnsemble(random_state=42)
    ensemble.fit(X_train, y_train)
    
    print(f" Learned weights: {ensemble.weights}")
    assert abs(sum(ensemble.weights.values()) - 1.0) < 1e-6, "Weights should sum to 1"
    assert not all(abs(w - 1/3) < 0.01 for w in ensemble.weights.values()), \
        "Weights should NOT all be 1/3 (learned, not fixed)"
    
    # Test 2: Prediction works
    print("\n[TEST 2] Predictions")
    X_test = np.random.randn(10, 5)
    y_pred = ensemble.predict(X_test)
    
    assert y_pred.shape == (10,)
    print(f" Predictions: shape={y_pred.shape}")
    
    # Test 3: Random state reproducibility
    print("\n[TEST 3] Random State Reproducibility")
    
    ensemble1 = HeterogeneousEnsemble(random_state=42)
    ensemble1.fit(X_train, y_train)
    pred1 = ensemble1.predict(X_test)
    
    ensemble2 = HeterogeneousEnsemble(random_state=42)
    ensemble2.fit(X_train, y_train)
    pred2 = ensemble2.predict(X_test)
    
    assert np.allclose(pred1, pred2), "Same random_state should produce same predictions"
    print(" Reproducibility verified")
    
    # Test 4: Multi-resolution ensemble
    print("\n[TEST 4] Multi-Resolution Ensemble")
    J = 2
    X_dict = {j: np.random.randn(100, 5) for j in range(J+1)}
    y_dict = {j: np.random.randn(100) for j in range(J+1)}
    
    multi_ensemble = MultiResolutionEnsemble(J=J, random_state=42)
    multi_ensemble.fit(X_dict, y_dict)
    
    X_test_dict = {j: np.random.randn(10, 5) for j in range(J+1)}
    y_pred = multi_ensemble.predict(X_test_dict)
    
    assert y_pred.shape == (10,)
    print(f" Multi-resolution predictions: shape={y_pred.shape}")
    
    print("\n" + "="*80)
    print(" ALL ENSEMBLE TESTS PASSED ")
    print("="*80)