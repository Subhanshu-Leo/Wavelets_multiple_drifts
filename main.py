"""
Main entry point for the wavelet drift detection pipeline.
"""

import numpy as np
import yaml
import argparse
from src.pipeline.drift_pipeline import WaveletDriftDetectionPipeline
from experiments.synthetic_data import SyntheticDriftGenerator
from utils.metrics import DriftMetrics

def main(config_path: str = 'config/config.yaml',
        drift_type: str = 'sudden',
        stream_length: int = 5000):
    """Run complete drift detection pipeline."""
    
    print("=" * 60)
    print("Wavelet-Enhanced Concept Drift Detection")
    print("=" * 60)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"\n✓ Loaded config from {config_path}")
    
    # Generate synthetic data
    print(f"\n✓ Generating {drift_type} drift stream (n={stream_length})...")
    gen = SyntheticDriftGenerator()
    X, y, true_drifts = gen.generate(drift_type=drift_type, n=stream_length)
    print(f"  True drifts at: {true_drifts}")
    
    # Split into warm-up and streaming
    n_warmup = config['ensemble']['n_init']
    X_warm, y_warm = X[:n_warmup], y[:n_warmup]
    X_stream, y_stream = X[n_warmup:], y[n_warmup:]
    
    # Initialize pipeline
    print(f"\n✓ Initializing pipeline...")
    pipeline = WaveletDriftDetectionPipeline(config)
    
    # Warm-up
    print(f"\n✓ Warm-up phase (n={n_warmup})...")
    warmup_stats = pipeline.warm_up(X_warm, y_warm)
    print(f"  Warm-up complete: {warmup_stats}")
    
    # Run stream processing
    print(f"\n✓ Running stream processing (n={len(X_stream)})...")
    results = pipeline.process_stream(X_stream, y_stream)
    
    print(f"  Predictions made: {results['predictions_made']}")
    print(f"  Drifts detected at relative t: {results['drifts_detected']}")
    if 'drift_types' in results:
        print(f"  Drift types: {results['drift_types']}")
    print(f"  Retrainings triggered: {results['retrainings']}")
    
    # Evaluate
    print(f"\n✓ Evaluating results...")
    # Adjust true drifts to relative streaming time (subtract warm-up period)
    relative_true_drifts = [d - n_warmup for d in true_drifts if d > n_warmup]
    detected_drifts = results['drifts_detected']
    
    metrics = DriftMetrics.compute_all(
        true_drifts=relative_true_drifts,
        detected_drifts=detected_drifts,
        tolerance=100  # Accept detection within 100 samples
    )
    
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--drift-type', type=str, default='sudden',
                       choices=['sudden', 'gradual', 'incremental', 'recurring'])
    parser.add_argument('--stream-length', type=int, default=5000)
    
    args = parser.parse_args()
    main(args.config, args.drift_type, args.stream_length)