"""
Run this from your project root:
    python diagnose_13.py

It patches _post_drift_adapt to print exactly what code path runs
and whether the fix is present in the loaded module.
"""
import numpy as np
import inspect
import sys
import importlib

# ── 1. Check what's actually on disk ──────────────────────────────────────────
print("=" * 70)
print("STEP 1: Checking drift_pipeline.py on disk")
print("=" * 70)

with open("src/pipeline/drift_pipeline.py", "r") as f:
    source = f.read()

has_fix = "y_dict reconstruction failed at scale" in source
has_reconstruct_call = "self.decomposer.reconstruct(decomp_j_only)" in source
has_decomp_j_only_dict = "decomp_j_only = {" in source

print(f"  Fix marker present ('y_dict reconstruction failed'):  {has_fix}")
print(f"  reconstruct(decomp_j_only) call present:             {has_reconstruct_call}")
print(f"  decomp_j_only dict construction present:             {has_decomp_j_only_dict}")

# Show the actual y_dict block from disk
start = source.find("# *** THE FIX")
if start == -1:
    start = source.find("y_decomp_full = self.decomposer.decompose(y_train_for_decomp)")
if start != -1:
    print("\n  Relevant block found on disk:")
    snippet = source[start:start+600]
    for line in snippet.splitlines():
        print(f"    {line}")
else:
    print("\n  WARNING: Could not find the y_dict construction block!")
    # Show what IS there around ensemble.fit
    idx = source.find("self.ensemble.fit(X_dict_train")
    if idx != -1:
        chunk = source[max(0, idx-400):idx+100]
        print("  Code before ensemble.fit():")
        for line in chunk.splitlines():
            print(f"    {line}")

# ── 2. Force-reload the module (bypass .pyc cache) ───────────────────────────
print("\n" + "=" * 70)
print("STEP 2: Force-reloading module (bypassing .pyc cache)")
print("=" * 70)

# Remove cached module if already imported
for key in list(sys.modules.keys()):
    if "drift_pipeline" in key or "pipeline" in key:
        del sys.modules[key]

import importlib.util
spec = importlib.util.spec_from_file_location(
    "drift_pipeline_fresh",
    "src/pipeline/drift_pipeline.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Extract the _post_drift_adapt source from the LOADED module
adapt_src = inspect.getsource(mod.WaveletDriftDetectionPipeline._post_drift_adapt)
has_fix_in_loaded = "y_dict reconstruction failed at scale" in adapt_src
has_reconstruct_in_loaded = "self.decomposer.reconstruct(decomp_j_only)" in adapt_src

print(f"  Fix present in LOADED module: {has_fix_in_loaded}")
print(f"  reconstruct call in LOADED:   {has_reconstruct_in_loaded}")

if not has_fix_in_loaded:
    print("\n  LOADED _post_drift_adapt y_dict block:")
    idx = adapt_src.find("y_decomp")
    if idx != -1:
        print(adapt_src[idx:idx+400])

# ── 3. Run the actual pipeline and intercept the failing call ─────────────────
print("\n" + "=" * 70)
print("STEP 3: Running pipeline with instrumented _post_drift_adapt")
print("=" * 70)

import yaml, logging
logging.disable(logging.CRITICAL)  # silence logs for clarity

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

print(f"  cooldown_samples from config: {config.get('detection', {}).get('cooldown_samples', 'NOT SET')}")

Pipeline = mod.WaveletDriftDetectionPipeline
pipeline = Pipeline(config)

print(f"  pipeline._cooldown_duration: {pipeline._cooldown_duration}")
print(f"  pipeline._min_signal_length: {pipeline._min_signal_length}")

np.random.seed(42)
X_warm = np.random.randn(500, 5)
y_warm = np.sum(X_warm, axis=1) + 0.1 * np.random.randn(500)
pipeline.warm_up(X_warm, y_warm)

# Patch _post_drift_adapt to intercept and diagnose
orig_adapt = pipeline._post_drift_adapt

def instrumented_adapt(X_new, y_new, drift_type='mean'):
    print(f"\n  --- _post_drift_adapt called ---")
    print(f"  X_new.shape: {X_new.shape}  (cooldown buffer size)")
    print(f"  y_new.shape: {y_new.shape}")

    J = pipeline.config.dwt_level
    split = len(X_new) // 2
    X_train = X_new[:split]
    y_train = y_new[:split]
    print(f"  split={split}, X_train len={len(X_train)}, min_signal_length={pipeline._min_signal_length}")

    if len(X_train) < pipeline._min_signal_length:
        pad_len = pipeline._min_signal_length - len(X_train)
        y_buf = np.array(list(pipeline._y_buffer))
        if len(y_buf) >= pad_len:
            y_pad = y_buf[-pad_len:]
        else:
            shortage = pad_len - len(y_buf)
            y_pad = np.concatenate([np.full(shortage, y_train[0]), y_buf])
        y_train_for_decomp = np.concatenate([y_pad, y_train])
    else:
        y_train_for_decomp = y_train

    print(f"  y_train_for_decomp len: {len(y_train_for_decomp)}")

    # Show what y_dict will look like under BOTH old and new code
    import pywt
    coeffs_full = pywt.wavedec(y_train_for_decomp, pipeline.config.wavelet, level=J)
    raw_lens = [len(c) for c in coeffs_full]
    print(f"\n  Raw coeff lengths: {raw_lens}")
    print(f"  OLD y_dict min_len (raw coeffs): {min(raw_lens)}  <-- causes 'got N samples' error")

    # Check what the fix produces
    y_dict_new = {}
    for j in range(J + 1):
        decomp_j_only = {
            k: (coeffs_full[0] if k == 0 and j == 0
                else np.zeros_like(coeffs_full[0]) if k == 0
                else coeffs_full[J + 1 - k] if k == j
                else np.zeros_like(coeffs_full[J + 1 - k]))
            for k in range(J + 1)
        }
        rec_coeffs = [decomp_j_only[0]] + [decomp_j_only[jj] for jj in range(J, 0, -1)]
        rec = pywt.waverec(rec_coeffs, pipeline.config.wavelet)
        y_dict_new[j] = rec[:len(y_train_for_decomp)]

    new_lens = [len(y_dict_new[j]) for j in range(J + 1)]
    print(f"  NEW y_dict lengths (reconstructed): {new_lens}")
    print(f"  NEW min_len: {min(new_lens)}  <-- what HeterogeneousEnsemble will see")

    # Now actually call the original (to see if it uses old or new path)
    print(f"\n  Calling actual _post_drift_adapt...")
    try:
        orig_adapt(X_new, y_new, drift_type)
        print(f"  _post_drift_adapt: SUCCESS")
    except Exception as e:
        print(f"  _post_drift_adapt: FAILED with: {e}")

pipeline._post_drift_adapt = instrumented_adapt

# Generate stream with a big drift to trigger retraining
X_stream = np.random.randn(400, 5)
y_stream = np.concatenate([
    np.sum(X_stream[:100], axis=1) + 0.1 * np.random.randn(100),
    np.sum(X_stream[100:], axis=1) + 8.0 + 0.1 * np.random.randn(300),
])

results = pipeline.process_stream(X_stream, y_stream)
print(f"\n  drifts_detected: {results['drifts_detected']}")
print(f"  retrainings:     {results['retrainings']}")

print("\n" + "=" * 70)
print("DIAGNOSIS COMPLETE")
print("=" * 70)