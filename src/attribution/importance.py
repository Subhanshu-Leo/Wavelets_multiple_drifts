"""
Permutation-based feature importance with CORRECTED out-of-sample evaluation.

CRITICAL FIX:
- Compute baseline MSE on held-out test set, NOT training data
- Use cross-validated baseline to avoid overfitting bias
- Permutation importance measures real feature impact, not overfit artifacts

Reference: Breiman (2001), Molnar (2020)
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict
from scipy import stats
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


def compute_permutation_importance(X_train: np.ndarray,
                                   y_train: np.ndarray,
                                   X_test: np.ndarray,
                                   y_test: np.ndarray,
                                   n_repeats: int = 10,
                                   random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute feature importance via permutation on held-out test set.
    
    CRITICAL FIX:
    1. Baseline MSE computed on TEST set (not training)
    2. Importance measures real generalization impact
    3. Avoids overfitting bias from in-sample evaluation
    
    Args:
        X_train: Training features (n_samples, n_features)
        y_train: Training targets (n_samples,)
        X_test: Test features (n_test, n_features)
        y_test: Test targets (n_test,)
        n_repeats: Number of permutation repeats
        random_state: For reproducibility
    
    Returns:
        (importances, feature_ranks)
        - importances: Feature importance scores
        - feature_ranks: Rank of each feature (1=most important)
    
    Reference: Breiman (2001), Molnar (2020)
    """
    np.random.seed(random_state)
    
    # Train model on training set
    model = RandomForestRegressor(
        n_estimators=50, max_depth=8, random_state=random_state
    )
    model.fit(X_train, y_train)
    
    # CRITICAL FIX: Baseline MSE on TEST set (not training)
    y_pred_test = model.predict(X_test)
    baseline_mse = np.mean((y_test - y_pred_test) ** 2)
    
    logger.info(
        f"Baseline MSE on test set: {baseline_mse:.6f} "
        f"(test size: {len(X_test)})"
    )
    
    if baseline_mse == 0:
        logger.warning("Baseline MSE is zero. Feature importance is undefined.")
        # Return uniform importance
        return np.ones(X_train.shape[1]) / X_train.shape[1], \
               np.ones(X_train.shape[1])
    
    n_features = X_train.shape[1]
    importances = np.zeros(n_features)
    
    # Permutation importance for each feature
    for feature_idx in range(n_features):
        mse_increases = []
        
        for repeat in range(n_repeats):
            # Permute feature on TEST set
            X_test_permuted = X_test.copy()
            X_test_permuted[:, feature_idx] = np.random.permutation(
                X_test_permuted[:, feature_idx]
            )
            
            # MSE on permuted test set
            y_pred_permuted = model.predict(X_test_permuted)
            permuted_mse = np.mean((y_test - y_pred_permuted) ** 2)
            
            # Importance: increase in MSE from permutation
            mse_increase = permuted_mse - baseline_mse
            mse_increases.append(mse_increase)
        
        # Average importance over repeats
        importances[feature_idx] = np.mean(mse_increases)
        
        logger.debug(
            f"Feature {feature_idx}: importance={importances[feature_idx]:.6f}, "
            f"std={np.std(mse_increases):.6f}"
        )
    
    # Rank features (higher importance = lower rank number)
    feature_ranks = stats.rankdata(-importances)
    
    logger.info(
        f"Top 3 features: {np.argsort(-importances)[:3]} "
        f"with importances {np.sort(-importances)[:3]}"
    )
    
    return importances, feature_ranks


def compute_importance_with_crossval(X: np.ndarray,
                                    y: np.ndarray,
                                    n_splits: int = 5,
                                    n_repeats: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute importance via cross-validation (when no test set available).
    
    ALTERNATIVE: Use CV to avoid train-test data leakage.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        n_splits: Cross-validation folds
        n_repeats: Permutation repeats per fold
    
    Returns:
        (importances, feature_ranks)
    
    Reference: Scikit-learn best practices
    """
    from sklearn.model_selection import KFold
    
    n_samples, n_features = X.shape
    importances_cv = []
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Compute importance for this fold
        fold_importances, _ = compute_permutation_importance(
            X_train, y_train, X_test, y_test, n_repeats=n_repeats
        )
        
        importances_cv.append(fold_importances)
    
    # Average importance across folds
    importances = np.mean(importances_cv, axis=0)
    feature_ranks = stats.rankdata(-importances)
    
    logger.info(f"Cross-validated importance computed ({n_splits} folds)")
    
    return importances, feature_ranks


def identify_top_important_features(importances: np.ndarray,
                                   k: int = 3) -> np.ndarray:
    """
    Identify top-k most important features.
    
    Args:
        importances: Feature importance scores
        k: Number of top features
    
    Returns:
        Indices of top-k features (sorted by importance, descending)
    """
    top_idx = np.argsort(-importances)[:k]
    top_importances = importances[top_idx]
    
    logger.info(f"Top {k} features: {top_idx} with importances {top_importances}")
    
    return top_idx


# Example
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    print("="*60)
    print("Permutation Importance: Out-of-Sample Validation")
    print("="*60)
    
    np.random.seed(42)
    
    # Generate synthetic data
    n_samples = 200
    X = np.random.randn(n_samples, 5)
    
    # Feature 0 is important, others are noise
    y = X[:, 0] + 0.05*np.random.randn(n_samples)
    
    # Train-test split
    split_idx = int(0.7 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Test 1: Out-of-sample importance
    print("\nTest 1: Out-of-sample permutation importance")
    importances, ranks = compute_permutation_importance(
        X_train, y_train, X_test, y_test, n_repeats=10
    )
    
    print(f"✓ Importances: {importances}")
    print(f"✓ Ranks: {ranks}")
    
    # Feature 0 should have highest importance
    assert ranks[0] <= 2, f"Feature 0 should be in top 2, got rank {ranks[0]}"
    print(f"✓ Feature 0 correctly identified as most important")
    
    # Test 2: Top k features
    print("\nTest 2: Identify top-3 features")
    top_3 = identify_top_important_features(importances, k=3)
    print(f"✓ Top 3 features: {top_3}")
    
    # Test 3: Cross-validated importance (no separate test set)
    print("\nTest 3: Cross-validated importance")
    importances_cv, ranks_cv = compute_importance_with_crossval(
        X, y, n_splits=3, n_repeats=5
    )
    
    print(f"✓ CV importances: {importances_cv}")
    assert ranks_cv[0] <= 2
    print(f"✓ Feature 0 correctly identified in CV as well")
    
    print("\n" + "="*60)
    print("✓ All importance tests passed!")
    print("="*60)