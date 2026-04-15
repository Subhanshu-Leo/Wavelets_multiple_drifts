"""
Simple test to see if drift detection is working at all.
"""

import numpy as np
import logging
from src.pipeline.drift_pipeline import WaveletDriftDetectionPipeline

logging.basicConfig(level=logging.DEBUG)

print("\n" + "="*80)
print("SIMPLE DRIFT DETECTION TEST")
print("="*80 + "\n")

# Config
config = {
    'dwt': {'wavelet': 'db4', 'level': 4},
    'detection': {'alpha': 0.05},
    'cooldown_samples': 50
}

# Initialize & warm up
pipeline = WaveletDriftDetectionPipeline(config)

np.random.seed(42)
X_warm = np.random.randn(500, 5)
y_warm = np.sum(X_warm, axis=1) + 0.1*np.random.randn(500)

print("[1] Warming up...")
pipeline.warm_up(X_warm, y_warm)
print(f"    OK Warm-up complete\n")

# Generate simple drift data
print("[2] Generating drift data...")
X_stream = []
y_stream = []
for t in range(150):
    X_t = np.random.randn(5)
    if t < 75:
        y_t = np.sum(X_t) + 0.1*np.random.randn()
    else:
        # LARGE drift: +5.0
        y_t = np.sum(X_t) + 5.0 + 0.1*np.random.randn()
    X_stream.append(X_t)
    y_stream.append(y_t)

X_stream = np.array(X_stream)
y_stream = np.array(y_stream)

print(f"    OK Data generated\n")

# Process stream
print("[3] Processing stream...")
results = pipeline.process_stream(X_stream, y_stream)

print(f"\n[4] RESULTS:")
print(f"    Predictions: {results['predictions_made']}")
print(f"    Drifts detected: {len(results['drifts_detected'])}")
print(f"    Detection times: {results['drifts_detected']}")
print(f"    Retrainings: {results['retrainings']}")

# Check state
print(f"\n[5] INTERNAL STATE:")
print(f"    _in_cooldown: {pipeline._in_cooldown}")
print(f"    _cooldown_duration: {pipeline._cooldown_duration}")
print(f"    _drift_history: {list(pipeline._drift_history)}")

if len(results['drifts_detected']) == 0:
    print(f"\nXX NO DRIFT DETECTED!")
    print(f"    Expected: 1 detection at t=85")
else:
    print(f"\nOK DRIFT DETECTED!")
    print(f"    Detection latency: {results['drifts_detected'][0] - 75} samples")

print("\n" + "="*80 + "\n")