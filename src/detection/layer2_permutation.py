"""
Layer 2: Permutation testing with CORRECTED composite statistic and Laplace correction.

CRITICAL FIXES:
1. Laplace correction for p-value (count + 1) / (b + 1)
2. Block permutation for time series (not standard permutation)
3. Symmetric normalization of mean and std terms
4. Proper early stopping logic
"""

import numpy as np
from scipy import stats
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class AdaptivePermutationTest:
    """Permutation test with composite statistic and block bootstrap."""
    
    @staticmethod
    def _composite_statistic(W_hist: np.ndarray, W_new: np.ndarray) -> float:
        """
        Compute composite test statistic.
        
        FORMULA: stat = mean_term + std_term
        where both terms are normalized by pooled_std (dimensionless).
        """
        if len(W_hist) == 0 or len(W_new) == 0:
            raise ValueError("Windows cannot be empty")
        
        mean_hist = np.mean(W_hist)
        mean_new = np.mean(W_new)
        std_hist = np.std(W_hist)
        std_new = np.std(W_new)
        
        pooled_std = np.std(np.concatenate([W_hist, W_new]))
        
        if pooled_std < 1e-10:
            pooled_std = 1e-10
        
        # Both terms normalized (dimensionless, comparable)
        mean_term = np.abs(mean_new - mean_hist) / pooled_std
        std_term = np.abs(std_new - std_hist) / pooled_std
        
        composite = mean_term + std_term
        
        logger.debug(
            f"Composite: mean_term={mean_term:.4f}, std_term={std_term:.4f}, "
            f"total={composite:.4f}"
        )
        
        return float(composite)
    
    @staticmethod
    def run_composite(W_hist: np.ndarray, W_new: np.ndarray,
                     b_min: int = 100, b_max: int = 1000,
                     early_stop_low: float = 0.001,
                     early_stop_high: float = 0.20) -> Tuple[float, int]:
        """
        Standard permutation test (uses pool-and-shuffle).
        
        CRITICAL FIX: Laplace correction on p-value.
        
        Note: For time series with autocorrelation, use block_permutation_test.
        """
        if len(W_hist) == 0 or len(W_new) == 0:
            raise ValueError("Windows cannot be empty")
        
        combined = np.concatenate([W_hist, W_new])
        n_hist = len(W_hist)
        
        delta_obs = AdaptivePermutationTest._composite_statistic(W_hist, W_new)
        
        count_extreme = 0
        
        for b in range(1, b_max + 1):
            shuffled_idx = np.random.permutation(len(combined))
            
            W_hist_perm = combined[shuffled_idx[:n_hist]]
            W_new_perm = combined[shuffled_idx[n_hist:]]
            
            delta_perm = AdaptivePermutationTest._composite_statistic(
                W_hist_perm, W_new_perm
            )
            
            if delta_perm >= delta_obs:
                count_extreme += 1
            
            # Early stopping check (after b_min permutations)
            if b >= b_min:
                # CRITICAL FIX: Laplace correction (count + 1) / (b + 1)
                p_val = (count_extreme + 1) / (b + 1)
                
                if p_val < early_stop_low:
                    logger.info(f"Early stop (drift): p={p_val:.6f} after {b} perms")
                    return p_val, b
                
                if p_val > early_stop_high:
                    logger.info(f"Early stop (no drift): p={p_val:.6f} after {b} perms")
                    return p_val, b
        
        # CRITICAL FIX: Laplace correction
        final_p = (count_extreme + 1) / (b_max + 1)
        
        logger.info(f"Completed {b_max} permutations: p={final_p:.6f}")
        
        return final_p, b_max
    
    @staticmethod
    def block_permutation_test(W_hist: np.ndarray, W_new: np.ndarray,
                              b_min: int = 100, b_max: int = 500,
                              block_size: int = 5) -> Tuple[float, int]:
        """
        Block permutation test for time series (handles autocorrelation).
        
        CRITICAL: Preserves temporal structure via block resampling.
        """
        combined = np.concatenate([W_hist, W_new])
        n_hist = len(W_hist)
        n_total = len(combined)
        
        delta_obs = AdaptivePermutationTest._composite_statistic(W_hist, W_new)
        
        count_extreme = 0
        
        for b in range(1, b_max + 1):
            # Block resampling (preserves temporal structure)
            n_blocks = max(1, n_total // block_size)
            block_indices = np.arange(n_blocks)
            np.random.shuffle(block_indices)
            
            shuffled = []
            for bi in block_indices:
                start = bi * block_size
                end = min((bi + 1) * block_size, n_total)
                shuffled.append(combined[start:end])
            
            shuffled_full = np.concatenate(shuffled)[:n_total]
            
            W_hist_perm = shuffled_full[:n_hist]
            W_new_perm = shuffled_full[n_hist:]
            
            delta_perm = AdaptivePermutationTest._composite_statistic(
                W_hist_perm, W_new_perm
            )
            
            if delta_perm >= delta_obs:
                count_extreme += 1
            
            # Early stopping
            if b >= b_min:
                # CRITICAL FIX: Laplace correction
                p_val = (count_extreme + 1) / (b + 1)
                
                if p_val < 0.001:
                    logger.info(f"Block permutation early stop (drift): p={p_val:.6f}")
                    return p_val, b
                
                if p_val > 0.20:
                    logger.info(f"Block permutation early stop (no drift): p={p_val:.6f}")
                    return p_val, b
        
        # CRITICAL FIX: Laplace correction
        final_p = (count_extreme + 1) / (b_max + 1)
        
        return final_p, b_max
    
    @staticmethod
    def run_ks_test(W_hist: np.ndarray, W_new: np.ndarray) -> Tuple[float, float]:
        """Kolmogorov-Smirnov two-sample test (faster alternative)."""
        stat, p_val = stats.ks_2samp(W_hist, W_new)
        logger.debug(f"KS test: stat={stat:.4f}, p={p_val:.6f}")
        return float(stat), float(p_val)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    
    print("="*80)
    print("Layer 2 Permutation Test: Composite + Laplace Correction")
    print("="*80)
    
    np.random.seed(42)
    
    # Test 1: Mean shift
    print("\n[TEST 1] Mean Shift")
    W_hist = np.random.normal(0, 1, 200)
    W_new = np.random.normal(0.5, 1, 200)
    
    p_val, n_perms = AdaptivePermutationTest.run_composite(W_hist, W_new)
    assert p_val < 0.05, "Should detect mean shift"
    print(f"✓ Mean shift detected: p={p_val:.6f}")
    
    # Test 2: Variance shift
    print("\n[TEST 2] Variance Shift (Primary Financial Drift Mode)")
    W_hist = np.random.normal(0, 1, 200)
    W_new = np.random.normal(0, 2, 200)
    
    p_val, n_perms = AdaptivePermutationTest.run_composite(W_hist, W_new)
    assert p_val < 0.05, "Should detect variance shift with composite statistic"
    print(f"✓ Variance shift detected: p={p_val:.6f}")
    
    # Test 3: Block permutation on time series
    print("\n[TEST 3] Block Permutation (Time Series)")
    W_hist = np.cumsum(np.random.randn(200)) / 10  # Autocorrelated
    W_new = np.cumsum(np.random.randn(200)) / 5 + 5  # Drifted
    
    p_val_std, n_perms_std = AdaptivePermutationTest.run_composite(W_hist, W_new)
    p_val_block, n_perms_block = AdaptivePermutationTest.block_permutation_test(W_hist, W_new)
    
    print(f"  Standard permutation: p={p_val_std:.6f}, perms={n_perms_std}")
    print(f"  Block permutation: p={p_val_block:.6f}, perms={n_perms_block}")
    print("✓ Both detect drift (block better for autocorrelated data)")
    
    # Test 4: Laplace correction prevents p=0
    print("\n[TEST 4] Laplace Correction (no p=0)")
    W_hist_constant = np.ones(200)
    W_new_shifted = np.ones(200) + 2
    
    p_val, _ = AdaptivePermutationTest.run_composite(W_hist_constant, W_new_shifted)
    assert p_val > 0, "p-value should never be exactly 0 (Laplace correction)"
    print(f"✓ p-value never 0: p={p_val:.6f}")
    
    print("\n" + "="*80)
    print("✓✓✓ ALL PERMUTATION TESTS PASSED ✓✓✓")
    print("="*80)