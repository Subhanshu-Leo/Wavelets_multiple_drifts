"""
Layer 1: Energy drift screening with CORRECTED statistical foundation.

CRITICAL FIXES:
1. Data-driven calibration of Hoeffding bounds (unbounded energy problem)
2. Proper dataclass field exposure prevention
3. TWO-SIDED Page-Hinkley test (detects both increases AND decreases)
4. Comprehensive logging and validation

References:
- Hoeffding, W. (1963). Probability inequalities for sums of bounded random variables.
- Page, E.S. (1954). Continuous inspection schemes.
- Germain et al. (2015). Risk bounds for the majority vote classifier.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np
import logging
import math

logger = logging.getLogger(__name__)


@dataclass
class HoeffingScreener:
    """
    Hoeffding-based drift screening with data-driven calibration.
    
    CRITICAL FIXES:
    1. Private fields use field(init=False, repr=False) to prevent 
       accidental construction-time override
    2. Requires calibration before use (enforced with RuntimeError)
    3. Clear API: input MUST be mean energy difference, not sum
    
    Usage:
        screener = HoeffingScreener(alpha=0.05)
        screener.calibrate(reference_energies)  # Must call first!
        delta_mean = |np.mean(E_new) - np.mean(E_hist)|
        if screener.should_escalate(delta_mean, n_window=200):
            # Escalate to Layer 2 (permutation test)
    """
    
    alpha: float = 0.05  # Type I error rate (user-configurable)
    
    # CRITICAL FIX: Private fields cannot be passed at __init__
    _calibrated_range: Optional[float] = field(default=None, init=False, repr=False)
    _n_obs: int = field(default=0, init=False, repr=False)
    _is_calibrated: bool = field(default=False, init=False, repr=False)
    
    def __post_init__(self):
        """Validate alpha after dataclass initialization."""
        if not (0 < self.alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")
        
        logger.debug(f"Initialized HoeffingScreener with alpha={self.alpha}")
    
    def calibrate(self, reference_energies: np.ndarray) -> None:
        """
        Calibrate Hoeffding bound using reference (historical) energy values.
        
        MUST be called before computing thresholds.
        
        This solves the critical problem: wavelet energy is unbounded, so we
        cannot use raw Hoeffding's inequality. Instead, we calibrate the bound
        on the energy RANGE observed in a stable (drift-free) historical window.
        
        Args:
            reference_energies: Energy values from stable period (1D array)
        
        Returns:
            None
        
        Raises:
            ValueError: If reference is too small or invalid
        
        Example:
            screener = HoeffingScreener(alpha=0.05)
            # historical_energies computed from stable window
            screener.calibrate(historical_energies)
        """
        # Input validation
        if not isinstance(reference_energies, np.ndarray):
            reference_energies = np.array(reference_energies)
        
        if reference_energies.ndim != 1:
            raise ValueError(f"reference_energies must be 1D, got shape {reference_energies.shape}")
        
        if len(reference_energies) < 10:
            raise ValueError(
                f"Need >=10 reference values for robust calibration, got {len(reference_energies)}"
            )
        
        if not np.isfinite(reference_energies).all():
            raise ValueError("reference_energies contains NaN or inf values")
        
        # Use percentile range: 1st to 99th percentile
        # More robust than min-max, especially for heavy-tailed energy distributions
        p01 = np.percentile(reference_energies, 1)
        p99 = np.percentile(reference_energies, 99)
        
        self._calibrated_range = p99 - p01
        self._n_obs = len(reference_energies)
        
        # Guard against zero-range (constant signal)
        if self._calibrated_range == 0:
            logger.warning(
                "Calibration range is zero (constant signal). "
                "Setting to 1.0 to avoid degenerate behavior."
            )
            self._calibrated_range = 1.0
        
        self._is_calibrated = True
        
        logger.info(
            f"HoeffingScreener calibrated successfully:\n"
            f"  Energy range: [{p01:.4f}, {p99:.4f}]\n"
            f"  Calibrated range (R): {self._calibrated_range:.4f}\n"
            f"  Reference observations: {self._n_obs}\n"
            f"  Significance level (α): {self.alpha}"
        )
    
    def threshold(self, n_window: int) -> float:
        """
        Compute Hoeffding threshold for a given window size.
        
        Formula:
            ε = sqrt(R² * ln(2/α) / (2n))
        
        where:
            R = calibrated energy range
            α = significance level
            n = window size
        
        CRITICAL: This threshold is for MEAN energy differences:
            δ = |E_new/n - E_hist/n|
        NOT for sums.
        
        Args:
            n_window: Size of comparison window (number of samples per window)
        
        Returns:
            Threshold epsilon (same units as mean energy difference)
        
        Raises:
            RuntimeError: If not yet calibrated
            ValueError: If n_window is invalid
        """
        if not self._is_calibrated:
            raise RuntimeError(
                "Call calibrate() with reference data before computing threshold. "
                "This is mandatory — the screener has no default bounds."
            )
        
        if n_window <= 0:
            raise ValueError(f"n_window must be > 0, got {n_window}")
        
        # Hoeffding's inequality:
        # P(|E_mean_new - E_mean_hist| >= ε) <= 2*exp(-2*n*ε² / R²)
        #
        # Solving for ε at significance level α:
        # 2*exp(-2*n*ε² / R²) = α
        # exp(-2*n*ε² / R²) = α/2
        # -2*n*ε² / R² = ln(α/2)
        # ε² = -R² * ln(α/2) / (2*n) = R² * ln(2/α) / (2*n)
        # ε = sqrt(R² * ln(2/α) / (2*n))
        
        try:
            epsilon = math.sqrt(
                self._calibrated_range**2 * math.log(2.0 / self.alpha) / (2 * n_window)
            )
        except (ValueError, OverflowError) as e:
            logger.error(f"Threshold computation failed: {e}")
            raise
        
        return epsilon
    
    def should_escalate(self, delta_mean_energy: float, n_window: int) -> bool:
        """
        Decide if observed change is large enough to escalate to Layer 2.
        
        CRITICAL FIX: Input is delta_mean_energy = |mean(E_new) - mean(E_hist)|,
        NOT the sum difference.
        
        If you have window energies E_new = sum(coeffs_new²) and 
        E_hist = sum(coeffs_hist²), the correct input is:
            delta_mean_energy = |E_new/n_window - E_hist/n_window|
        
        NOT:
            delta_mean_energy = |E_new - E_hist|  # ← WRONG!
        
        Args:
            delta_mean_energy: |mean(E_new) - mean(E_hist)|
                              Units: same as individual coefficient energies
            n_window: Window size (number of coefficients)
        
        Returns:
            True if delta_mean_energy > threshold (escalate to Layer 2)
            False if delta_mean_energy <= threshold (skip Layer 2)
        
        Raises:
            ValueError: If inputs invalid
        """
        # Input validation
        if not self._is_calibrated:
            raise RuntimeError(
                "Must call calibrate() before should_escalate(). "
                "Use the pattern: calibrate() once, then should_escalate() repeatedly."
            )
        
        if n_window <= 0:
            raise ValueError(f"n_window must be > 0, got {n_window}")
        
        if delta_mean_energy < 0:
            raise ValueError(
                f"delta_mean_energy must be >= 0, got {delta_mean_energy}. "
                f"Compute as |mean(E_new) - mean(E_hist)|."
            )
        
        if not np.isfinite(delta_mean_energy):
            raise ValueError(f"delta_mean_energy must be finite, got {delta_mean_energy}")
        
        # Compute threshold and make decision
        threshold = self.threshold(n_window)
        escalate = delta_mean_energy > threshold
        
        logger.debug(
            f"HoeffingScreener.should_escalate():\n"
            f"  delta_mean_energy: {delta_mean_energy:.6f}\n"
            f"  threshold: {threshold:.6f}\n"
            f"  escalate: {escalate}"
        )
        
        return escalate


@dataclass
class PageHinkleyTest:
    """
    TWO-SIDED Page-Hinkley test: detects BOTH upward and downward drifts.
    
    CRITICAL FIX: Original one-sided test only detects energy INCREASES.
    Financial volatility can DECREASE (e.g., crisis calms down, market stabilizes).
    This is a real drift mode, and the original code would miss it.
    
    Two-sided maintains separate cumulative sums for increases and decreases.
    
    References:
    - Page, E.S. (1954). Continuous inspection schemes.
    - Lowry & Montgomery (1995). A review of multivariate control charts.
    
    Usage:
        ph = PageHinkleyTest(delta=1.0, lambda_param=5.0)
        ph.calibrate(reference_energies)
        for t, energy in enumerate(stream):
            if ph.update(energy):
                print(f"Drift detected at time {t}")
    """
    
    delta: float = 1.0  # Minimum detectable change (in units of std)
    lambda_param: float = 5.0  # Detection threshold (typically 3-5)
    
    # Reference statistics (set by calibrate)
    _reference_mean: Optional[float] = field(default=None, init=False, repr=False)
    _reference_std: Optional[float] = field(default=None, init=False, repr=False)
    
    # CRITICAL FIX: TWO-SIDED test requires separate accumulators
    # For upward shifts
    _m_up: float = field(default=0.0, init=False, repr=False)
    _min_m_up: float = field(default=0.0, init=False, repr=False)
    
    # For downward shifts
    _m_down: float = field(default=0.0, init=False, repr=False)
    _min_m_down: float = field(default=0.0, init=False, repr=False)
    
    # Tracking
    _n_updates: int = field(default=0, init=False, repr=False)
    _is_calibrated: bool = field(default=False, init=False, repr=False)
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.delta <= 0:
            raise ValueError(f"delta must be > 0, got {self.delta}")
        if self.lambda_param <= 0:
            raise ValueError(f"lambda_param must be > 0, got {self.lambda_param}")
    
    def calibrate(self, reference_energies: np.ndarray) -> None:
        """
        Calibrate on reference window.
        
        Args:
            reference_energies: Energy values from stable period
        
        Raises:
            ValueError: If reference is invalid
        """
        if not isinstance(reference_energies, np.ndarray):
            reference_energies = np.array(reference_energies)
        
        if len(reference_energies) < 10:
            raise ValueError(
                f"Need >=10 reference values, got {len(reference_energies)}"
            )
        
        if not np.isfinite(reference_energies).all():
            raise ValueError("reference_energies contains NaN or inf values")
        
        self._reference_mean = float(np.mean(reference_energies))
        self._reference_std = float(np.std(reference_energies))
        self._is_calibrated = True
        
        logger.info(
            f"PageHinkleyTest (TWO-SIDED) calibrated:\n"
            f"  Mean: {self._reference_mean:.4f}\n"
            f"  Std: {self._reference_std:.4f}\n"
            f"  Delta: {self.delta} std\n"
            f"  Lambda: {self.lambda_param}"
        )
    
    def update(self, energy_value: float) -> bool:
        """
        Update test statistic and check for drift (BOTH directions).
        
        CRITICAL FIX: Checks BOTH:
        - Upward drift: energy increases persistently
        - Downward drift: energy decreases persistently
        
        Args:
            energy_value: New energy value
        
        Returns:
            True if drift detected (either direction)
        
        Raises:
            RuntimeError: If not calibrated
        """
        if not self._is_calibrated:
            raise RuntimeError("Call calibrate() first")
        
        if not np.isfinite(energy_value):
            raise ValueError(f"energy_value must be finite, got {energy_value}")
        
        # Deviation from reference mean
        drift_term = self.delta * self._reference_std / 2.0
        deviation = energy_value - self._reference_mean
        
        # UPWARD shift detector
        self._m_up += deviation - drift_term
        self._min_m_up = min(self._min_m_up, self._m_up)
        stat_up = self._m_up - self._min_m_up
        
        # DOWNWARD shift detector (symmetrically)
        self._m_down -= deviation - drift_term  # Negative so decreases are positive
        self._min_m_down = min(self._min_m_down, self._m_down)
        stat_down = self._m_down - self._min_m_down
        
        # Alarm if either direction exceeds threshold
        threshold = self.lambda_param * self._reference_std
        alarm = (stat_up > threshold) or (stat_down > threshold)
        
        self._n_updates += 1
        
        if alarm:
            alarm_type = "UPWARD" if stat_up > threshold else "DOWNWARD"
            logger.warning(
                f"PageHinkleyTest ALARM ({alarm_type}):\n"
                f"  Time: {self._n_updates}\n"
                f"  Energy value: {energy_value:.4f}\n"
                f"  Upward stat: {stat_up:.4f}\n"
                f"  Downward stat: {stat_down:.4f}\n"
                f"  Threshold: {threshold:.4f}"
            )
        else:
            logger.debug(
                f"PageHinkleyTest update {self._n_updates}: "
                f"energy={energy_value:.4f}, stat_up={stat_up:.4f}, "
                f"stat_down={stat_down:.4f}"
            )
        
        return alarm
    
    def reset(self):
        """Reset for next stream."""
        self._m_up = 0.0
        self._min_m_up = 0.0
        self._m_down = 0.0
        self._min_m_down = 0.0
        self._n_updates = 0
        logger.info("PageHinkleyTest reset")
    
    def get_state(self) -> dict:
        """Get current test state (for monitoring/debugging)."""
        return {
            'n_updates': self._n_updates,
            'm_up': self._m_up,
            'min_m_up': self._min_m_up,
            'm_down': self._m_down,
            'min_m_down': self._min_m_down,
            'stat_up': self._m_up - self._min_m_up,
            'stat_down': self._m_down - self._min_m_down,
        }


# Example and validation
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    print("="*70)
    print("Hoeffding Screener & Page-Hinkley Test: Validation")
    print("="*70)
    
    np.random.seed(42)
    
    # ======================================================================
    # Test 1: HoeffingScreener with proper calibration
    # ======================================================================
    print("\n" + "="*70)
    print("TEST 1: HoeffingScreener with Data-Driven Calibration")
    print("="*70)
    
    # Reference window: stable data
    ref_energies = 100 + 10*np.random.randn(500)
    screener = HoeffingScreener(alpha=0.05)
    screener.calibrate(ref_energies)
    
    # Test 1a: Small change should NOT escalate
    small_change = 0.5  # Mean energy difference < 0.5
    assert not screener.should_escalate(small_change, n_window=200)
    print(f" Test 1a: Small change (δ={small_change}) correctly NOT escalated")
    
    # Test 1b: Large change should escalate
    large_change = 2.0  # Mean energy difference > 2.0
    assert screener.should_escalate(large_change, n_window=200)
    print(f" Test 1b: Large change (δ={large_change}) correctly escalated")
    
    # Test 1c: Error handling for uncalibrated screener
    screener_uncal = HoeffingScreener(alpha=0.05)
    try:
        screener_uncal.should_escalate(1.0, 200)
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        print(" Test 1c: Uncalibrated screener correctly raises error")
    
    # ======================================================================
    # Test 2: Page-Hinkley Test — TWO-SIDED (key fix!)
    # ======================================================================
    print("\n" + "="*70)
    print("TEST 2: Page-Hinkley Test (TWO-SIDED) — Detects UP and DOWN")
    print("="*70)
    
    # Test 2a: Stable phase (no alarm)
    ph = PageHinkleyTest(delta=1.0, lambda_param=5.0)
    ph.calibrate(ref_energies)
    
    alarms = []
    for _ in range(50):
        is_alarm = ph.update(100 + np.random.randn())
        if is_alarm:
            alarms.append('stable')
    
    assert len(alarms) == 0, "Should not alarm during stable phase"
    print(" Test 2a: Stable phase — no false alarms")
    
    # Test 2b: UPWARD shift (energy increases)
    ph.reset()
    ph.calibrate(ref_energies)
    
    alarm_time_up = None
    for t in range(100):
        shifted_value = 115 + np.random.randn()  # Mean shifted UP
        if ph.update(shifted_value):
            alarm_time_up = t
            break
    
    assert alarm_time_up is not None and alarm_time_up < 50
    print(f" Test 2b: UPWARD shift detected at time {alarm_time_up}")
    
    # Test 2c: DOWNWARD shift (energy decreases) — CRITICAL!
    # This is the key fix: original one-sided test would miss this.
    ph.reset()
    ph.calibrate(ref_energies)
    
    alarm_time_down = None
    for t in range(100):
        shifted_value = 85 + np.random.randn()  # Mean shifted DOWN
        if ph.update(shifted_value):
            alarm_time_down = t
            break
    
    assert alarm_time_down is not None and alarm_time_down < 50
    print(f" Test 2c: DOWNWARD shift detected at time {alarm_time_down} (KEY FIX!)")
    
    # ======================================================================
    # Test 3: Integration Test — Screener → Page-Hinkley Pipeline
    # ======================================================================
    print("\n" + "="*70)
    print("TEST 3: Layer 1 → Layer 2 Integration")
    print("="*70)
    
    # Simulate Layer 1 (Hoeffding screener) followed by Layer 2 (Page-Hinkley)
    screener = HoeffingScreener(alpha=0.05)
    ph = PageHinkleyTest(delta=1.0, lambda_param=5.0)
    
    # Calibrate both on reference
    screener.calibrate(ref_energies)
    ph.calibrate(ref_energies)
    
    # Simulate stream with drift
    detected = False
    for t in range(200):
        if t < 100:
            # Stable phase
            energy = 100 + 10*np.random.randn()
        else:
            # Drifted phase (upward shift)
            energy = 120 + 10*np.random.randn()
        
        # Layer 1: Quick screening via Hoeffding
        mean_energy_diff = np.abs(energy - 100) / np.sqrt(100)  # Normalized
        
        if screener.should_escalate(mean_energy_diff, n_window=200):
            # Layer 2: Confirmation via Page-Hinkley
            if ph.update(energy):
                detected = True
                print(f" Drift detected via Layer 1→2 at time {t}")
                break
        else:
            ph.update(energy)  # Still update Ph without escalation
    
    assert detected, "Pipeline should detect drift"
    
    print("\n" + "="*70)
    print(" ALL TESTS PASSED!")
    print("="*70)