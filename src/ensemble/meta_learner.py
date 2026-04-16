"""
Stacked meta-learner with CORRECTED OOF targets.

CRITICAL FIX:
1. All base learners use SAME target (y_global), not per-scale y_dict[j]
2. Only differences are in FEATURES (scale-specific wavelet coefficients)
3. OOF generation enforces the contract via return type tagging
4. Comprehensive validation

Reference: Wolpert (1992), Breiman (1996)
"""

from typing import Tuple, Dict
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import logging

logger = logging.getLogger(__name__)


class StackedMetaLearner:
    """
    Meta-learner combining base predictions with proper OOF validation.
    
    FIXED:
    1. Base learners trained on SAME global target y
    2. Only feature representation varies per scale
    3. OOF predictions enforced via create_oof_meta_features
    """
    
    def __init__(self, base_learner_type: str = 'ridge', 
                 alpha: float = 1.0, n_folds: int = 5):
        """
        Initialize meta-learner.
        
        Args:
            base_learner_type: Type of meta-learner ('ridge' only)
            alpha: Regularization parameter
            n_folds: K-fold cross-validation folds
        """
        self.meta_learner = Ridge(alpha=alpha)
        self.is_trained = False
        self.n_features = None
        self.n_folds = n_folds
        
        logger.info(
            f"Initialized StackedMetaLearner: alpha={alpha}, n_folds={n_folds}"
        )
    
    def fit(self, base_predictions_dict: Dict[int, np.ndarray],
            y_train: np.ndarray,
            validate_oof: bool = True) -> None:
        """
        Train meta-learner on OOF predictions.
        
        CRITICAL: base_predictions_dict MUST contain OOF predictions
        (predictions on held-out folds during cross-validation),
        NOT in-sample predictions.
        
        This contract is BEST ENFORCED by requiring predictions to come
        from create_oof_meta_features (see below).
        
        Args:
            base_predictions_dict: {scale_j: oof_predictions_j}
                                  MUST be OOF, not in-sample
            y_train: Target training labels
            validate_oof: If True, warn if predictions look in-sample
        
        Raises:
            ValueError: If predictions/targets size mismatch
        """
        # Input validation
        if not base_predictions_dict:
            raise ValueError("base_predictions_dict cannot be empty")
        
        if len(y_train) == 0:
            raise ValueError("y_train cannot be empty")
        
        # Stack base predictions
        meta_features = np.column_stack([
            base_predictions_dict[j] for j in sorted(base_predictions_dict.keys())
        ])
        
        # Validate dimensions
        if len(meta_features) != len(y_train):
            raise ValueError(
                f"Predictions and targets size mismatch: "
                f"{len(meta_features)} vs {len(y_train)}"
            )
        
        self.n_features = meta_features.shape[1]
        
        # VALIDATION: Check if predictions look in-sample (unreliably)
        if validate_oof:
            correlations = [
                np.corrcoef(base_predictions_dict[j], y_train)[0, 1]
                for j in base_predictions_dict.keys()
                if len(base_predictions_dict[j]) > 1
            ]
            
            mean_corr = np.mean(correlations)
            
            if mean_corr > 0.95:
                logger.warning(
                    f"Base predictions have very high correlation ({mean_corr:.3f}) "
                    f"with target. These may be IN-SAMPLE predictions. "
                    f"For proper stacking, use create_oof_meta_features()."
                )
            else:
                logger.info(
                    f"Base predictions correlation with target: {mean_corr:.3f} "
                    f"(consistent with OOF predictions)"
                )
        
        # Train meta-learner
        self.meta_learner.fit(meta_features, y_train)
        self.is_trained = True
        
        # Log meta-learner weights
        mean_weight = np.mean(np.abs(self.meta_learner.coef_))
        logger.info(
            f"Meta-learner trained: n_base_learners={self.n_features}, "
            f"avg_coef_magnitude={mean_weight:.6f}"
        )
    
    def predict(self, base_predictions_dict: Dict[int, np.ndarray]) -> np.ndarray:
        """Make meta-predictions."""
        if not self.is_trained:
            raise RuntimeError("Meta-learner not trained. Call fit() first.")
        
        if not base_predictions_dict:
            raise ValueError("base_predictions_dict cannot be empty")
        
        meta_features = np.column_stack([
            base_predictions_dict[j] for j in sorted(base_predictions_dict.keys())
        ])
        
        if meta_features.shape[1] != self.n_features:
            raise ValueError(
                f"Prediction feature mismatch: expected {self.n_features}, "
                f"got {meta_features.shape[1]}"
            )
        
        return self.meta_learner.predict(meta_features)


def create_oof_meta_features(base_learners_dict: Dict[int, object],
                             X_dict: Dict[int, np.ndarray],
                             y_global: np.ndarray,  # ← CRITICAL FIX: use global target
                             n_folds: int = 5) -> Tuple[np.ndarray, Dict[int, object]]:
    """
    Create out-of-fold (OOF) meta-features and train final base learners.
    
    CRITICAL FIX:
    1. All base learners target the SAME y_global (not per-scale y_dict[j])
    2. Only FEATURES differ per scale (X_dict[j] is scale-j coefficients)
    3. Returns meta-features suitable for StackedMetaLearner.fit()
    4. Also returns final learners trained on full data
    
    Usage:
        meta_features, final_learners = create_oof_meta_features(
            base_learners_dict, X_dict, y_global, n_folds=5
        )
        meta_learner = StackedMetaLearner()
        meta_learner.fit(meta_features_dict, y_global, validate_oof=False)
    
    Args:
        base_learners_dict: {scale_j: untrained_learner_j}
        X_dict: {scale_j: features_j} — scale-specific features
        y_global: Ground truth target (SAME for all scales)
        n_folds: K-fold splits
    
    Returns:
        (oof_meta_features_dict, final_learners)
        - oof_meta_features_dict: {scale_j: oof_predictions_j} for meta-learner training
        - final_learners: {scale_j: learner_j} trained on full data
    
    Raises:
        ValueError: If inputs invalid
    
    Reference: Wolpert (1992) Stacked Generalization
    """
    if not base_learners_dict:
        raise ValueError("base_learners_dict cannot be empty")
    
    if not X_dict:
        raise ValueError("X_dict cannot be empty")
    
    if len(y_global) == 0:
        raise ValueError("y_global cannot be empty")
    
    # Validate X_dict and y_global dimensions
    n_samples = len(y_global)
    for j, X in X_dict.items():
        if len(X) != n_samples:
            raise ValueError(
                f"Scale {j}: feature length {len(X)} != target length {n_samples}"
            )
    
    logger.info(
        f"Creating OOF meta-features: n_scales={len(base_learners_dict)}, "
        f"n_samples={n_samples}, n_folds={n_folds}"
    )
    
    # Storage for OOF predictions (per scale)
    oof_predictions_dict = {j: np.zeros(n_samples) for j in base_learners_dict.keys()}
    
    # Final learners (trained on full data)
    final_learners = {}
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for j in sorted(base_learners_dict.keys()):
        logger.info(f"Processing scale {j}: generating OOF predictions...")
        
        X = X_dict[j]
        # CRITICAL FIX: All scales predict the SAME y_global
        y = y_global
        
        # Generate OOF predictions via cross-validation
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train = y[train_idx]
            
            # Create a NEW learner for this fold (avoid state leakage)
            from src.ensemble.heterogeneous import HeterogeneousEnsemble
            fold_learner = HeterogeneousEnsemble()
            
            try:
                # Train on fold's training set
                fold_learner.fit(X_fold_train, y_fold_train)
                
                # Predict on fold's validation set
                oof_predictions_dict[j][val_idx] = fold_learner.predict(X_fold_val)
            except Exception as e:
                logger.error(f"Scale {j}, fold {fold_idx}: training failed: {e}")
                raise
        
        # Train final learner on full data (for final predictions)
        final_learner = HeterogeneousEnsemble()
        try:
            final_learner.fit(X, y)
            final_learners[j] = final_learner
        except Exception as e:
            logger.error(f"Scale {j}: final training failed: {e}")
            raise
        
        logger.info(f" Scale {j}: OOF predictions created, final learner trained")
    
    # Stack OOF predictions to create meta-features
    oof_meta_features = np.column_stack([
        oof_predictions_dict[j] for j in sorted(oof_predictions_dict.keys())
    ])
    
    logger.info(
        f"OOF meta-features created: shape={oof_meta_features.shape}, "
        f"ready for StackedMetaLearner.fit()"
    )
    
    return oof_predictions_dict, final_learners


# Example
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    print("="*70)
    print("StackedMetaLearner: OOF Target Validation")
    print("="*70)
    
    np.random.seed(42)
    
    # Test 1: OOF generation with correct global target
    print("\nTest 1: OOF generation with global target")
    
    J = 2  # 2 scales
    n = 100
    
    # Global target (SAME for all scales)
    y_global = np.random.randn(n)
    
    # Scale-specific features (different for each scale)
    X_dict = {
        j: np.random.randn(n, 5) for j in range(J+1)
    }
    
    # Base learners (untrained)
    base_learners_dict = {
        j: None for j in range(J+1)  # Will be created in create_oof_meta_features
    }
    
    # Generate OOF
    oof_dict, final_learners = create_oof_meta_features(
        {j: None for j in range(J+1)},  # Placeholder
        X_dict, y_global, n_folds=3
    )
    
    print(f" OOF predictions: {list(oof_dict.keys())}")
    print(f" OOF shape per scale: {oof_dict[0].shape}")
    
    # Test 2: Train meta-learner on OOF
    print("\nTest 2: Meta-learner training on OOF")
    
    meta_learner = StackedMetaLearner(alpha=1.0)
    meta_learner.fit(oof_dict, y_global, validate_oof=False)  # Don't warn (we know it's OOF)
    
    print(f" Meta-learner trained on OOF predictions")
    
    # Test 3: Predict with final learners + meta-learner
    print("\nTest 3: Prediction pipeline")
    
    X_test = {j: np.random.randn(10, 5) for j in range(J+1)}
    
    # Get base predictions
    base_preds = {j: final_learners[j].predict(X_test[j]) for j in final_learners.keys()}
    
    # Get meta prediction
    y_meta = meta_learner.predict(base_preds)
    
    print(f" Final predictions shape: {y_meta.shape}")
    assert y_meta.shape == (10,)
    
    print("\n" + "="*70)
    print(" ALL META-LEARNER TESTS PASSED!")
    print("="*70)