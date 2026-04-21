"""
Real-world dataset loaders for drift detection experiments.

Two datasets are supported out of the box:

  SP500RealLoader     — Daily S&P 500 index data (Yahoo Finance CSV)
                        Regression: predict next-day log return
                        Known regime changes: COVID crash (2020-02), rate
                        hike cycle (2022-01), tech selloff (2022-04)

  ElectricityLoader   — UCI ELEC2 dataset (45k hourly records)
                        Regression: predict NSW electricity demand
                        Known drift: seasonal shifts, market restructuring

Both produce the same output contract as the synthetic generator:
    X_warm, y_warm    — warm-up arrays (n_warmup, n_features)
    X_stream, y_stream — stream arrays (n_stream, n_features)
    known_drifts       — list of approximate stream indices where regime changes
                         are known to occur (used as ground truth for evaluation)

HOW TO GET THE DATA
-------------------
SP500 (option A – automatic):
    pip install yfinance
    Python: import yfinance as yf
            df = yf.download('^GSPC', start='2015-01-01', end='2024-01-01', auto_adjust=True)
            df.to_csv('data/sp500.csv')

SP500 (option B – manual):
    Go to https://finance.yahoo.com/quote/%5EGSPC/history/
    Download → Historical Data → Max range → Download CSV → save as data/sp500.csv

UCI ELEC2:
    Download from https://www.openml.org/d/151
    Save the CSV as data/electricity.csv
    Expected columns: date, day, period, nswprice, nswdemand,
                      vicprice, vicdemand, transfer, class

USAGE
-----
    from experiments.real_data_loader import SP500RealLoader, ElectricityLoader
    from experiments.real_data_loader import run_real_data_comparison

    # Quick start
    results = run_real_data_comparison('data/sp500.csv', dataset='sp500')

    # Fine-grained control
    loader = SP500RealLoader('data/sp500.csv', warmup_ratio=0.2)
    X_warm, y_warm, X_stream, y_stream, known_drifts, meta = loader.load()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# SP500 loader
# ──────────────────────────────────────────────────────────────────────────────

class SP500RealLoader:
    """
    Load S&P 500 daily data and engineer features for drift detection.

    Feature vector at time t (7 features):
        [ret_{t-1}, ret_{t-2}, ret_{t-3}, ret_{t-4}, ret_{t-5},
         rolling_vol_20, rolling_mom_20]

    Target y[t]: log return at time t (same-period, NOT forward).
    This means the model predicts "given the last 5 returns and recent
    vol/momentum, what is today's return?"  Regime changes are visible
    as sudden jumps in y when the mean or variance shifts.

    Parameters
    ----------
    filepath : str
        Path to Yahoo Finance CSV (columns: Date, Open, High, Low,
        Close, Adj Close or similar).
    warmup_ratio : float
        Fraction of the data to use as warm-up (default 0.20 = ~400 days).
    ar_lags : int
        Number of AR lag features (default 5).
    vol_window : int
        Rolling volatility window in days (default 20).
    mom_window : int
        Rolling momentum window in days (default 20).
    """

    # Known approximate dates of major S&P 500 regime changes
    # These are used to compute stream indices for ground-truth evaluation
    KNOWN_REGIME_CHANGES = {
        '2020-02-19': 'COVID crash begins',
        '2022-01-03': 'Rate hike cycle starts',
        '2022-04-01': 'Tech selloff accelerates',
    }

    def __init__(
        self,
        filepath: str,
        warmup_ratio: float = 0.20,
        ar_lags: int = 5,
        vol_window: int = 20,
        mom_window: int = 20,
    ):
        self.filepath = Path(filepath)
        self.warmup_ratio = warmup_ratio
        self.ar_lags = ar_lags
        self.vol_window = vol_window
        self.mom_window = mom_window

    def load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int], Dict]:
        """
        Load, engineer features, split warm-up / stream, locate drift points.

        Returns
        -------
        X_warm, y_warm, X_stream, y_stream, known_drifts, metadata
        """
        if not self.filepath.exists():
            raise FileNotFoundError(
                f"SP500 CSV not found: {self.filepath}\n"
                "Download from Yahoo Finance: https://finance.yahoo.com/quote/%5EGSPC/history/\n"
                "Or auto-download via: import yfinance as yf; "
                "yf.download('^GSPC', start='2015-01-01').to_csv('data/sp500.csv')"
            )

        # ── Load CSV ──────────────────────────────────────────────────────
        df = pd.read_csv(self.filepath, parse_dates=['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

        # Accept either 'Adj Close' or 'Close'
        price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        prices = df[price_col].values.astype(float)
        dates  = df['Date'].values

        if len(prices) < 200:
            raise ValueError(f"Too few rows ({len(prices)}) — need at least 200 trading days")

        logger.info(f"Loaded {len(prices)} rows from {self.filepath}")

        # ── Log returns ───────────────────────────────────────────────────
        log_returns = np.diff(np.log(np.maximum(prices, 1e-10)))  # shape (n-1,)
        return_dates = dates[1:]

        # ── Feature engineering ───────────────────────────────────────────
        n = len(log_returns)

        # AR lag features
        ar_feats = np.zeros((n, self.ar_lags))
        for lag in range(1, self.ar_lags + 1):
            ar_feats[lag:, lag - 1] = log_returns[: n - lag]

        # Volatility (rolling std) and momentum (rolling sum)
        ret_series = pd.Series(log_returns)
        vol = ret_series.rolling(self.vol_window).std().values
        mom = ret_series.rolling(self.mom_window).sum().values

        features = np.column_stack([ar_feats, vol, mom])  # (n, ar_lags+2)

        # Drop rows with NaN (first ~vol_window rows)
        valid = ~np.isnan(features).any(axis=1)
        X = features[valid]
        y = log_returns[valid]
        valid_dates = return_dates[valid]

        logger.info(f"After feature engineering: {X.shape[0]} rows, {X.shape[1]} features")

        # ── Warm-up / stream split ────────────────────────────────────────
        n_warmup = max(int(self.warmup_ratio * len(X)), 200)
        X_warm,   y_warm   = X[:n_warmup],  y[:n_warmup]
        X_stream, y_stream = X[n_warmup:],  y[n_warmup:]
        stream_dates       = valid_dates[n_warmup:]

        # ── Locate known regime changes in stream coordinates ─────────────
        known_drifts = []
        drift_labels = []
        for date_str, label in self.KNOWN_REGIME_CHANGES.items():
            ts = np.datetime64(date_str)
            idx = np.searchsorted(stream_dates, ts)
            if 0 < idx < len(stream_dates):
                known_drifts.append(int(idx))
                drift_labels.append(f"t={idx}: {label} ({date_str})")
                logger.info(f"Regime change '{label}' → stream index {idx}")

        metadata = {
            'source':            'S&P 500',
            'filepath':          str(self.filepath),
            'n_total':           len(X),
            'n_warmup':          n_warmup,
            'n_stream':          len(X_stream),
            'n_features':        X.shape[1],
            'feature_names':     [f'ret_lag{i}' for i in range(1, self.ar_lags + 1)]
                                 + ['vol_20', 'mom_20'],
            'warmup_date_range': f"{pd.Timestamp(valid_dates[0]).date()} – "
                                 f"{pd.Timestamp(valid_dates[n_warmup-1]).date()}",
            'stream_date_range': f"{pd.Timestamp(stream_dates[0]).date()} – "
                                 f"{pd.Timestamp(stream_dates[-1]).date()}",
            'known_drifts':      known_drifts,
            'drift_labels':      drift_labels,
            'y_mean':            float(y.mean()),
            'y_std':             float(y.std()),
        }

        self._print_summary(metadata)
        return X_warm, y_warm, X_stream, y_stream, known_drifts, metadata

    @staticmethod
    def _print_summary(meta: Dict) -> None:
        print(f"\n{'─'*60}")
        print(f"  Dataset        : {meta['source']}")
        print(f"  Warm-up period : {meta['warmup_date_range']} ({meta['n_warmup']} days)")
        print(f"  Stream period  : {meta['stream_date_range']} ({meta['n_stream']} days)")
        print(f"  Features       : {meta['n_features']}  {meta['feature_names']}")
        print(f"  Known drifts   : {len(meta['known_drifts'])} events")
        for label in meta['drift_labels']:
            print(f"    {label}")
        print(f"{'─'*60}")

    @classmethod
    def download_and_save(cls, save_path: str = 'data/sp500.csv',
                          start: str = '2015-01-01', end: str = '2024-01-01') -> str:
        """
        Auto-download SP500 via yfinance and save to CSV.

        Requires: pip install yfinance
        """
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("Run: pip install yfinance")

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df = yf.download('^GSPC', start=start, end=end,
                         auto_adjust=True, progress=False)
        df.to_csv(save_path)
        logger.info(f"SP500 data saved to {save_path} ({len(df)} rows)")
        print(f"Downloaded {len(df)} rows → {save_path}")
        return save_path


# ──────────────────────────────────────────────────────────────────────────────
# UCI Electricity loader
# ──────────────────────────────────────────────────────────────────────────────

class ElectricityLoader:
    """
    Load UCI ELEC2 dataset (hourly electricity demand, NSW/Victoria).

    Regression target: NSW electricity demand (nswdemand), standardised.

    Features (6):
        day (0–6), period (0–47), nswprice, vicprice, vicdemand, transfer

    Known drifts: seasonal transitions, market restructuring around
    rows 15000 and 27000 in the full 45k-row dataset.

    Parameters
    ----------
    filepath : str
        Path to electricity.csv (from https://www.openml.org/d/151).
    warmup_ratio : float
        Fraction for warm-up (default 0.15 ≈ 6700 rows).
    standardise : bool
        If True, standardise X and y using warm-up statistics only
        (avoids data leakage).
    """

    KNOWN_REGIME_CHANGES = {
        15000: 'Market restructuring / seasonal shift',
        27000: 'Demand pattern change',
    }

    FEATURE_COLS = ['day', 'period', 'nswprice', 'vicprice', 'vicdemand', 'transfer']
    TARGET_COL   = 'nswdemand'

    def __init__(
        self,
        filepath: str,
        warmup_ratio: float = 0.15,
        standardise: bool = True,
    ):
        self.filepath = Path(filepath)
        self.warmup_ratio = warmup_ratio
        self.standardise = standardise

    def load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int], Dict]:
        """
        Load and prepare electricity data.

        Returns
        -------
        X_warm, y_warm, X_stream, y_stream, known_drifts, metadata
        """
        if not self.filepath.exists():
            raise FileNotFoundError(
                f"Electricity CSV not found: {self.filepath}\n"
                "Download from: https://www.openml.org/d/151\n"
                "Save as electricity.csv with columns:\n"
                "  date, day, period, nswprice, nswdemand, vicprice, "
                "vicdemand, transfer, class"
            )

        df = pd.read_csv(self.filepath)

        # Accept both 'class' and 'label' column names (OpenML uses both)
        # Rename to avoid Python reserved word
        if 'class' in df.columns:
            df = df.rename(columns={'class': 'price_direction'})

        # Validate expected columns
        missing_cols = [c for c in self.FEATURE_COLS + [self.TARGET_COL]
                        if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing columns: {missing_cols}\n"
                f"Found: {df.columns.tolist()}"
            )

        # Drop any rows with NaN
        df = df[self.FEATURE_COLS + [self.TARGET_COL]].dropna().reset_index(drop=True)

        X_raw = df[self.FEATURE_COLS].values.astype(float)
        y_raw = df[self.TARGET_COL].values.astype(float)

        logger.info(f"Loaded {len(df)} rows from {self.filepath}")

        # ── Warm-up / stream split ────────────────────────────────────────
        n_warmup = max(int(self.warmup_ratio * len(X_raw)), 200)
        X_warm_raw,   y_warm_raw   = X_raw[:n_warmup],  y_raw[:n_warmup]
        X_stream_raw, y_stream_raw = X_raw[n_warmup:],  y_raw[n_warmup:]

        # ── Standardise using warm-up stats only ─────────────────────────
        if self.standardise:
            X_mean = X_warm_raw.mean(axis=0)
            X_std  = np.maximum(X_warm_raw.std(axis=0), 1e-6)
            y_mean = float(y_warm_raw.mean())
            y_std  = max(float(y_warm_raw.std()), 1e-6)

            X_warm   = (X_warm_raw   - X_mean) / X_std
            X_stream = (X_stream_raw - X_mean) / X_std
            y_warm   = (y_warm_raw   - y_mean) / y_std
            y_stream = (y_stream_raw - y_mean) / y_std
        else:
            X_warm, y_warm     = X_warm_raw,   y_warm_raw
            X_stream, y_stream = X_stream_raw, y_stream_raw

        # ── Known drift stream indices ────────────────────────────────────
        known_drifts = []
        drift_labels = []
        for raw_idx, label in self.KNOWN_REGIME_CHANGES.items():
            stream_idx = raw_idx - n_warmup
            if 0 < stream_idx < len(X_stream):
                known_drifts.append(stream_idx)
                drift_labels.append(f"t={stream_idx}: {label} (row {raw_idx})")

        metadata = {
            'source':        'UCI ELEC2',
            'filepath':      str(self.filepath),
            'n_total':       len(X_raw),
            'n_warmup':      n_warmup,
            'n_stream':      len(X_stream),
            'n_features':    len(self.FEATURE_COLS),
            'feature_names': self.FEATURE_COLS,
            'target':        self.TARGET_COL,
            'standardised':  self.standardise,
            'known_drifts':  known_drifts,
            'drift_labels':  drift_labels,
        }

        self._print_summary(metadata)
        return X_warm, y_warm, X_stream, y_stream, known_drifts, metadata

    @staticmethod
    def _print_summary(meta: Dict) -> None:
        print(f"\n{'─'*60}")
        print(f"  Dataset     : {meta['source']}")
        print(f"  Warm-up     : {meta['n_warmup']} rows")
        print(f"  Stream      : {meta['n_stream']} rows")
        print(f"  Features    : {meta['n_features']}  {meta['feature_names']}")
        print(f"  Target      : {meta['target']} (standardised={meta['standardised']})")
        print(f"  Known drifts: {len(meta['known_drifts'])} events")
        for label in meta['drift_labels']:
            print(f"    {label}")
        print(f"{'─'*60}")


# ──────────────────────────────────────────────────────────────────────────────
# One-call comparison runner
# ──────────────────────────────────────────────────────────────────────────────

def run_real_data_comparison(
    filepath: str,
    dataset: str = 'sp500',
    pipeline=None,
    tolerance: int = 200,
    warmup_ratio: float = 0.20,
    verbose: bool = True,
) -> Dict:
    """
    Load a real dataset and run all baseline comparisons (+ pipeline if given).

    Parameters
    ----------
    filepath    : path to CSV file
    dataset     : 'sp500' or 'electricity'
    pipeline    : WaveletDriftDetectionPipeline instance (optional)
    tolerance   : detection window in samples (wider for real data)
    warmup_ratio: fraction for warm-up
    verbose     : print results table

    Returns
    -------
    dict with per-model results (same format as run_comparison)
    """
    # Load data
    if dataset == 'sp500':
        loader = SP500RealLoader(filepath, warmup_ratio=warmup_ratio)
    elif dataset == 'electricity':
        loader = ElectricityLoader(filepath, warmup_ratio=warmup_ratio)
    else:
        raise ValueError(f"Unknown dataset: '{dataset}'. Use 'sp500' or 'electricity'.")

    X_warm, y_warm, X_stream, y_stream, known_drifts, meta = loader.load()

    print(f"\nRunning comparison on {meta['source']}...")
    print(f"Known drift times (stream coords): {known_drifts}")
    if not known_drifts:
        print("  (no known drifts in the loaded date range — "
              "all detections will be counted as false positives)")

    # Import here to avoid circular deps
    from experiments.baselines import run_comparison

    results = run_comparison(
        X_warm, y_warm,
        X_stream, y_stream,
        true_drift_times=known_drifts,
        pipeline=pipeline,
        tolerance=tolerance,
        verbose=verbose,
    )

    results['__metadata__'] = meta
    return results