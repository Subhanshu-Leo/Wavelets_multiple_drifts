"""
Standalone baseline runner.

Usage:
    python run_baselines.py                          # all drift types, default settings
    python run_baselines.py --drift-type sudden      # one drift type only
    python run_baselines.py --n 3000 --warmup 600    # custom stream length / warmup
    python run_baselines.py --tolerance 150          # wider detection window
    python run_baselines.py --seed 7                 # different random seed
"""

import argparse
import logging
import sys

import numpy as np

sys.path.insert(0, ".")

from experiments.baselines import build_all_baselines, run_comparison
from experiments.synthetic_data import SyntheticDriftGenerator
from utils.metrics import DriftMetrics


def run_one(drift_type: str, n: int, n_warmup: int, tolerance: int, seed: int) -> dict:
    gen = SyntheticDriftGenerator(seed=seed)
    X, y, true_drifts = gen.generate(drift_type=drift_type, n=n)

    X_warm,   y_warm   = X[:n_warmup],  y[:n_warmup]
    X_stream, y_stream = X[n_warmup:],  y[n_warmup:]

    rel_true = [d - n_warmup for d in true_drifts if d > n_warmup]

    print(f"\n{'─'*60}")
    print(f"  Drift type : {drift_type}")
    print(f"  Stream len : {len(X_stream)}  |  Warm-up : {n_warmup}")
    print(f"  True drifts (stream coords) : {rel_true}")
    print(f"{'─'*60}")

    results = run_comparison(
        X_warm, y_warm,
        X_stream, y_stream,
        true_drift_times=rel_true,
        pipeline=None,          # set to your pipeline instance to include it
        tolerance=tolerance,
        verbose=True,
    )
    return results


def main():
    parser = argparse.ArgumentParser(description="Run baseline drift detectors")
    parser.add_argument(
        "--drift-type",
        default="all",
        choices=["all", "sudden", "gradual", "incremental", "recurring"],
        help="Which drift type to test (default: all)",
    )
    parser.add_argument("--n",        type=int, default=3000, help="Total stream length (default 3000)")
    parser.add_argument("--warmup",   type=int, default=500,  help="Warm-up samples (default 500)")
    parser.add_argument("--tolerance",type=int, default=100,  help="Detection tolerance in samples (default 100)")
    parser.add_argument("--seed",     type=int, default=42,   help="Random seed (default 42)")
    parser.add_argument("--verbose",  action="store_true",    help="Show DEBUG logs")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    np.random.seed(args.seed)

    drift_types = (
        ["sudden", "gradual", "incremental", "recurring"]
        if args.drift_type == "all"
        else [args.drift_type]
    )

    all_summaries = {}
    for dtype in drift_types:
        all_summaries[dtype] = run_one(
            drift_type=dtype,
            n=args.n,
            n_warmup=args.warmup,
            tolerance=args.tolerance,
            seed=args.seed,
        )

    # Cross-drift-type summary: average F1 per model
    if len(drift_types) > 1:
        print("\n" + "=" * 60)
        print("  AVERAGE F1 ACROSS ALL DRIFT TYPES")
        print("=" * 60)
        model_names = [b.name for b in build_all_baselines()]
        scores = {name: [] for name in model_names}
        for dtype, res in all_summaries.items():
            for name in model_names:
                if name in res and "metrics" in res[name]:
                    scores[name].append(res[name]["metrics"]["f1"])
        for name in sorted(scores, key=lambda n: -np.mean(scores[n]) if scores[n] else 0):
            vals = scores[name]
            if vals:
                print(f"  {name:<20}  avg F1 = {np.mean(vals):.3f}  "
                      f"(per type: {[f'{v:.2f}' for v in vals]})")
        print("=" * 60)


if __name__ == "__main__":
    main()