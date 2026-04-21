"""
Baseline drift detectors for comparison against WaveletDriftDetectionPipeline.

Five baselines chosen to cover the standard literature:

  1. StaticBaseline   — no drift detection at all; frozen warm-up model
                        Lower bound on all metrics; shows cost of ignoring drift.

  2. CUSUMBaseline    — Page (1954) cumulative-sum control chart.
                        Gold-standard for sudden mean shifts in industrial SPC.

  3. ADWINBaseline    — Bifet & Gavalda (2007) adaptive windowing.
                        Most cited practical baseline; handles gradual drift.

  4. DDMBaseline      — Gama et al. (2004) drift detection method.
                        Error-rate based; canonical ML-stream baseline.

  5. KSWINBaseline    — Kolmogorov-Smirnov windowed test.
                        Distribution-free; sensitive to shape/variance changes,
                        not just mean — a fair test of your composite statistic.

All baselines share the same public interface as WaveletDriftDetectionPipeline:
    baseline.warm_up(X_warm, y_warm)   -> stats dict
    baseline.process_stream(X, y)      -> results dict

results dict keys (identical to pipeline):
    drifts_detected   : list[int]   stream indices where drift was flagged
    predictions       : list[float]
    errors            : list[float]
    predictions_made  : int
    retrainings       : int

Usage:
    from experiments.baselines import build_all_baselines, run_comparison
    results = run_comparison(X_warm, y_warm, X_stream, y_stream, true_drifts)
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

from utils.metrics import DriftMetrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared base class
# ---------------------------------------------------------------------------

class _BaselineBase:
    """
    Common warm-up / stream loop shared by all baselines.

    Subclasses only need to implement:
        _fit_predictor(X, y)          -> fit self._model on warm-up data
        _predict_one(x)               -> return scalar prediction for row x
        _update_detector(error, t)    -> update detector state; return True if drift
        _retrain(X_buf, y_buf)        -> retrain model on recent data (optional)
    """

    name: str = "BaseBaseline"

    def __init__(self):
        self._model = None
        self._is_warm = False
        # Rolling buffer used for optional retraining after detected drift
        self._retrain_buf_X: deque = deque(maxlen=200)
        self._retrain_buf_y: deque = deque(maxlen=200)

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def warm_up(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train predictor and calibrate detector on warm-up data."""
        self._fit_predictor(X, y)
        # Compute warm-up errors to calibrate detector
        preds = np.array([self._predict_one(X[i]) for i in range(len(X))])
        errors = y - preds
        self._calibrate_detector(errors)
        self._is_warm = True
        logger.info(f"{self.name}: warm-up complete on {len(X)} samples")
        return {
            "n_samples": len(X),
            "model": type(self._model).__name__,
            "ensemble_trained": True,
        }

    def process_stream(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Process stream sample-by-sample; return results dict."""
        if not self._is_warm:
            raise RuntimeError("Call warm_up() first.")

        drifts_detected: List[int] = []
        predictions: List[float] = []
        errors: List[float] = []
        retrainings = 0

        # Cooldown: after detecting drift, suppress further alarms for N steps
        cooldown_remaining = 0
        _COOLDOWN = 50

        for t in range(len(X)):
            x_t, y_t = X[t], float(y[t])

            y_pred = self._predict_one(x_t)
            error  = y_t - y_pred

            predictions.append(y_pred)
            errors.append(error)

            # Buffer for retraining
            self._retrain_buf_X.append(x_t)
            self._retrain_buf_y.append(y_t)

            if cooldown_remaining > 0:
                cooldown_remaining -= 1
                continue

            if self._update_detector(error, t):
                drifts_detected.append(t)
                cooldown_remaining = _COOLDOWN

                # Optional retraining on recent buffer
                if len(self._retrain_buf_X) >= 30:
                    Xb = np.array(self._retrain_buf_X)
                    yb = np.array(self._retrain_buf_y)
                    try:
                        self._retrain(Xb, yb)
                        retrainings += 1
                        # Re-calibrate detector on new error distribution
                        preds_buf = np.array([self._predict_one(Xb[i]) for i in range(len(Xb))])
                        self._calibrate_detector(yb - preds_buf)
                    except Exception as e:
                        logger.warning(f"{self.name}: retraining failed: {e}")

        return {
            "drifts_detected":  drifts_detected,
            "predictions":      predictions,
            "errors":           errors,
            "predictions_made": len(predictions),
            "retrainings":      retrainings,
        }

    # ------------------------------------------------------------------ #
    # Overridable hooks                                                    #
    # ------------------------------------------------------------------ #

    def _fit_predictor(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit a Ridge regression predictor (default for all baselines)."""
        self._model = Ridge(alpha=1.0)
        self._model.fit(X, y)

    def _predict_one(self, x: np.ndarray) -> float:
        return float(self._model.predict(x.reshape(1, -1))[0])

    def _calibrate_detector(self, errors: np.ndarray) -> None:
        """Override to initialise detector state from warm-up errors."""
        pass

    def _update_detector(self, error: float, t: int) -> bool:
        """Return True if drift is detected at this step."""
        return False

    def _retrain(self, X: np.ndarray, y: np.ndarray) -> None:
        """Retrain predictor after drift (default: refit Ridge)."""
        self._model = Ridge(alpha=1.0)
        self._model.fit(X, y)


# ---------------------------------------------------------------------------
# 1. Static baseline — frozen model, no detector
# ---------------------------------------------------------------------------

class StaticBaseline(_BaselineBase):
    """
    Frozen warm-up model with no drift detection.

    Establishes the *lower bound*: a pipeline that never adapts.
    Any method that detects and retrains should beat this on recall.
    On stable data, this should have the lowest false-alarm rate.
    """

    name = "Static"

    def _update_detector(self, error: float, t: int) -> bool:
        return False   # never fires

    def _retrain(self, X, y):
        pass           # never retrain


# ---------------------------------------------------------------------------
# 2. CUSUM baseline
# ---------------------------------------------------------------------------

class CUSUMBaseline(_BaselineBase):
    """
    Two-sided CUSUM (Page 1954) on the absolute prediction error.

    Detects persistent mean shifts in the error stream.
    Threshold h is set to k * sigma_warm (calibrated from warm-up).

    Parameters
    ----------
    k : float
        Allowable slack (typical range 0.5–2.0; 1.0 = detect 1-sigma shifts).
    h_sigma : float
        Alarm threshold as multiple of warm-up error std.
        Larger = fewer false alarms, slower detection.
    """

    name = "CUSUM"

    def __init__(self, k: float = 0.5, h_sigma: float = 5.0):
        super().__init__()
        self.k = k
        self.h_sigma = h_sigma
        # State
        self._mu0: float = 0.0
        self._sigma0: float = 1.0
        self._h: float = 5.0
        self._S_pos: float = 0.0   # upper cumsum
        self._S_neg: float = 0.0   # lower cumsum

    def _calibrate_detector(self, errors: np.ndarray) -> None:
        self._mu0    = float(np.mean(errors))
        self._sigma0 = max(float(np.std(errors)), 1e-6)
        self._h      = self.h_sigma * self._sigma0
        self._S_pos  = 0.0
        self._S_neg  = 0.0
        logger.debug(f"CUSUM calibrated: mu={self._mu0:.4f}, sigma={self._sigma0:.4f}, h={self._h:.4f}")

    def _update_detector(self, error: float, t: int) -> bool:
        # Standardise error relative to warm-up distribution
        z = (error - self._mu0) / self._sigma0
        # Two-sided update
        self._S_pos = max(0.0, self._S_pos + z - self.k)
        self._S_neg = max(0.0, self._S_neg - z - self.k)
        if self._S_pos > self._h or self._S_neg > self._h:
            # Reset after alarm
            self._S_pos = 0.0
            self._S_neg = 0.0
            return True
        return False


# ---------------------------------------------------------------------------
# 3. ADWIN baseline
# ---------------------------------------------------------------------------

class ADWINBaseline(_BaselineBase):
    """
    Simplified ADWIN (Bifet & Gavalda 2007).

    Maintains a growing window of errors. At each step it tests every
    possible split of the window into two halves using Hoeffding's bound.
    If the means of the two halves differ significantly, drift is flagged
    and the older half is discarded.

    This implementation is O(n) per step — accurate to the paper's logic
    without the full bucket structure, which is fine for stream lengths
    up to ~10k samples.

    Parameters
    ----------
    delta : float
        Confidence parameter (smaller = fewer false alarms).
        Typical values: 0.002 (conservative) to 0.1 (sensitive).
    min_window : int
        Minimum samples before any test fires.
    """

    name = "ADWIN"

    def __init__(self, delta: float = 0.02, min_window: int = 50):
        super().__init__()
        self.delta = delta
        self.min_window = min_window
        self._window: deque = deque()

    def _calibrate_detector(self, errors: np.ndarray) -> None:
        # Seed window with warm-up errors (last min_window samples)
        self._window = deque(errors[-self.min_window:].tolist())

    def _update_detector(self, error: float, t: int) -> bool:
        self._window.append(error)
        n = len(self._window)
        if n < self.min_window:
            return False

        arr = np.array(self._window)
        mean_total = arr.mean()

        # Test all splits; use stride to keep O(n) amortised
        stride = max(1, n // 40)
        for cut in range(self.min_window // 2, n - self.min_window // 2, stride):
            w0, w1 = arr[:cut], arr[cut:]
            n0, n1 = len(w0), len(w1)
            diff = abs(w0.mean() - w1.mean())
            # Hoeffding bound on the mean difference
            data_range = max(arr.max() - arr.min(), 1e-6)
            bound = data_range * np.sqrt(
                np.log(2.0 / self.delta) / (2.0 * (1.0 / n0 + 1.0 / n1) ** -1)
            )
            # Equivalent cleaner form:
            eps = data_range * np.sqrt(
                (1.0 / (2.0 * n0) + 1.0 / (2.0 * n1)) * np.log(2.0 / self.delta)
            )
            if diff > eps:
                # Discard older half and flag drift
                self._window = deque(arr[cut:].tolist())
                return True
        return False


# ---------------------------------------------------------------------------
# 4. DDM baseline
# ---------------------------------------------------------------------------

class DDMBaseline(_BaselineBase):
    """
    Drift Detection Method (Gama et al. 2004).

    Tracks a binary error stream (error above warm-up threshold = 1, else 0).
    Monitors the running error rate p_t and its standard deviation s_t.
    Drift is flagged when p_t + s_t > p_min + 3 * s_min.

    Parameters
    ----------
    error_threshold : float or None
        Absolute error above which a prediction counts as an "error".
        If None, calibrated from warm-up as mean + std of |errors|.
    warning_level : float
        Factor for warning zone (default 2.0 * s_min).
    drift_level : float
        Factor for drift alarm (default 3.0 * s_min; from original paper).
    min_instances : int
        Minimum samples before monitoring starts.
    """

    name = "DDM"

    def __init__(
        self,
        error_threshold: Optional[float] = None,
        warning_level: float = 2.0,
        drift_level: float = 3.0,
        min_instances: int = 30,
    ):
        super().__init__()
        self._error_threshold = error_threshold
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.min_instances = min_instances
        # Running stats
        self._n: int = 0
        self._p: float = 1.0   # running error rate
        self._s: float = 0.0   # std of error rate
        self._p_min: float = np.inf
        self._s_min: float = np.inf
        self._calibrated_threshold: float = 0.5

    def _calibrate_detector(self, errors: np.ndarray) -> None:
        abs_e = np.abs(errors)
        if self._error_threshold is None:
            # Threshold: predictions within 1 std of warm-up are "correct"
            self._calibrated_threshold = float(abs_e.mean() + abs_e.std())
        else:
            self._calibrated_threshold = self._error_threshold
        # Reset DDM state
        self._n = 0
        self._p = 1.0
        self._s = 0.0
        self._p_min = np.inf
        self._s_min = np.inf
        logger.debug(f"DDM calibrated: error_threshold={self._calibrated_threshold:.4f}")

    def _update_detector(self, error: float, t: int) -> bool:
        # Binarise
        mistake = 1 if abs(error) > self._calibrated_threshold else 0
        self._n += 1
        # Update running error rate (online mean)
        self._p += (mistake - self._p) / self._n
        self._s = np.sqrt(self._p * (1.0 - self._p) / self._n)

        if self._n < self.min_instances:
            return False

        if self._p + self._s <= self._p_min + self._s_min:
            self._p_min = self._p
            self._s_min = self._s

        if self._p + self._s > self._p_min + self.drift_level * self._s_min:
            # Reset on drift
            self._n = 0
            self._p = 1.0
            self._s = 0.0
            self._p_min = np.inf
            self._s_min = np.inf
            return True

        return False


# ---------------------------------------------------------------------------
# 5. KSWIN baseline
# ---------------------------------------------------------------------------

class KSWINBaseline(_BaselineBase):
    """
    Kolmogorov-Smirnov windowed test (Raab et al. 2020).

    Maintains a reference window (oldest half) and a test window (newest half).
    Runs a two-sample KS test every `test_freq` steps.
    Sensitive to ANY distributional change — mean, variance, or shape —
    making it a strong complement to CUSUM (mean-only) and DDM (rate-only).

    Parameters
    ----------
    window_size : int
        Total window length. Reference = first half, test = second half.
    alpha : float
        Significance level for the KS test (p < alpha → drift).
    test_freq : int
        How often to run the KS test (every N samples). Trades speed for
        sensitivity; 1 = every step, 10 = every 10 steps.
    """

    name = "KSWIN"

    def __init__(
        self, window_size: int = 100, alpha: float = 0.05, test_freq: int = 5
    ):
        super().__init__()
        self.window_size = window_size
        self.alpha = alpha
        self.test_freq = test_freq
        self._window: deque = deque(maxlen=window_size)
        self._step = 0

    def _calibrate_detector(self, errors: np.ndarray) -> None:
        # Seed with last window_size warm-up errors
        tail = errors[-self.window_size:] if len(errors) >= self.window_size else errors
        self._window = deque(tail.tolist(), maxlen=self.window_size)
        self._step = 0

    def _update_detector(self, error: float, t: int) -> bool:
        self._window.append(error)
        self._step += 1

        if len(self._window) < self.window_size:
            return False

        # Only run test every test_freq steps
        if self._step % self.test_freq != 0:
            return False

        arr = np.array(self._window)
        mid = self.window_size // 2
        ref_window  = arr[:mid]   # older half
        test_window = arr[mid:]   # recent half

        _, p_val = stats.ks_2samp(ref_window, test_window)

        if p_val < self.alpha:
            # Slide reference to current test window and start fresh
            self._window = deque(test_window.tolist(), maxlen=self.window_size)
            return True

        return False


# ---------------------------------------------------------------------------
# Comparison runner
# ---------------------------------------------------------------------------

def build_all_baselines() -> List[_BaselineBase]:
    """
    Return one instance of every baseline with sensible defaults.

    These defaults are calibrated for the synthetic streams in this project
    (stream length ~1500-4500, 5 features, sudden/gradual/incremental drifts).
    Adjust per-baseline kwargs if you use very different data.
    """
    return [
        StaticBaseline(),
        CUSUMBaseline(k=0.5, h_sigma=5.0),
        ADWINBaseline(delta=0.02, min_window=50),
        DDMBaseline(drift_level=3.0, min_instances=30),
        KSWINBaseline(window_size=100, alpha=0.05, test_freq=5),
    ]


def run_comparison(
    X_warm: np.ndarray,
    y_warm: np.ndarray,
    X_stream: np.ndarray,
    y_stream: np.ndarray,
    true_drift_times: List[int],
    pipeline=None,
    tolerance: int = 100,
    verbose: bool = True,
) -> Dict:
    """
    Warm up and evaluate all baselines (plus optionally your pipeline).

    Parameters
    ----------
    X_warm, y_warm       : warm-up data (same split as used by the pipeline)
    X_stream, y_stream   : evaluation stream
    true_drift_times     : ground-truth drift indices in stream coordinates
    pipeline             : WaveletDriftDetectionPipeline instance (optional).
                           If provided it is run first and included in the table.
    tolerance            : detection window (samples) for TP matching
    verbose              : print a formatted results table

    Returns
    -------
    dict keyed by model name:
        {
          'drifts_detected': [...],
          'metrics': {f1, precision, recall, latency, false_alarms, missed_drifts},
          'rmse': float,
          'mae':  float,
        }
    """
    all_results: Dict[str, Dict] = {}

    def _evaluate(name: str, raw: Dict) -> Dict:
        m = DriftMetrics.compute_all(
            true_drift_times, raw["drifts_detected"], tolerance=tolerance
        )
        errs = np.array(raw["errors"])
        return {
            "drifts_detected": raw["drifts_detected"],
            "retrainings":     raw.get("retrainings", 0),
            "predictions_made": raw.get("predictions_made", len(errs)),
            "metrics": m,
            "rmse":    float(np.sqrt(np.mean(errs ** 2))),
            "mae":     float(np.mean(np.abs(errs))),
        }

    # --- Run pipeline first if provided ---
    if pipeline is not None:
        np.random.seed(42)
        try:
            pipeline.warm_up(X_warm, y_warm)
            raw = pipeline.process_stream(X_stream, y_stream)
            all_results[pipeline.__class__.__name__] = _evaluate(
                pipeline.__class__.__name__, raw
            )
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")

    # --- Run all baselines ---
    import numpy as _np
    for baseline in build_all_baselines():
        _np.random.seed(42)
        try:
            baseline.warm_up(X_warm, y_warm)
            raw = baseline.process_stream(X_stream, y_stream)
            all_results[baseline.name] = _evaluate(baseline.name, raw)
        except Exception as e:
            logger.error(f"{baseline.name} failed: {e}")
            all_results[baseline.name] = {"error": str(e)}

    if verbose:
        _print_table(all_results, true_drift_times)

    return all_results


def _print_table(all_results: Dict, true_drifts: List[int]) -> None:
    """Print a formatted comparison table."""
    print("\n" + "=" * 85)
    print("DRIFT DETECTION COMPARISON")
    print(f"True drifts at: {true_drifts}")
    print("=" * 85)
    header = f"{'Model':<30} {'F1':>6} {'Prec':>6} {'Rec':>6} {'Lat':>7} {'FA':>4} {'RMSE':>8} {'Retrain':>8}"
    print(header)
    print("-" * 85)

    # Sort: pipeline first (if present), then by F1 descending
    pipeline_name = "WaveletDriftDetectionPipeline"
    def sort_key(item):
        name, r = item
        if "error" in r:
            return (2, 0)
        f1 = r["metrics"]["f1"]
        return (0 if name == pipeline_name else 1, -f1)

    for name, r in sorted(all_results.items(), key=sort_key):
        if "error" in r:
            print(f"{'  ' + name:<30} ERROR: {r['error']}")
            continue
        m = r["metrics"]
        marker = " ◀" if name == pipeline_name else ""
        print(
            f"{'  ' + name:<30} "
            f"{m['f1']:>6.3f} "
            f"{m['precision']:>6.3f} "
            f"{m['recall']:>6.3f} "
            f"{m['latency']:>7.1f} "
            f"{m['false_alarms']:>4d} "
            f"{r['rmse']:>8.4f} "
            f"{r['retrainings']:>8d}"
            f"{marker}"
        )
    print("=" * 85)
    print(
        "Columns: F1=harmonic mean, Prec=precision, Rec=recall, Lat=mean detection latency,\n"
        "         FA=false alarms, RMSE=root mean sq error, Retrain=# of model retrains\n"
    )