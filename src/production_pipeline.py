"""
Production Pipeline - Complete workflow combining Options 3 & 5

Train → Save → Load → Infer → Evaluate
"""

import numpy as np
import logging
import json
import os
from datetime import datetime
from typing import Dict, Tuple

from src.pipeline.drift_pipeline import WaveletDriftDetectionPipeline
from src.evaluation import DriftDetectionEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_and_save_pipeline(X_warm: np.ndarray, y_warm: np.ndarray, 
                           config: Dict, model_name: str = 'drift_detector_v1') -> str:
    """
    STEP 1: Train pipeline on warm-up data and save it.
    
    Args:
        X_warm: Warm-up features (n_samples, n_features)
        y_warm: Warm-up targets (n_samples,)
        config: Configuration dict
        model_name: Name for saved model (without .pkl)
    
    Returns:
        filepath: Path to saved model
    """
    logger.info("\n" + "="*80)
    logger.info("STEP 1: TRAINING PIPELINE")
    logger.info("="*80)
    
    # Initialize
    pipeline = WaveletDriftDetectionPipeline(config)
    
    # Warm-up
    warm_stats = pipeline.warm_up(X_warm, y_warm)
    logger.info(f"✓ Warm-up complete")
    for key, val in warm_stats.items():
        logger.info(f"  {key}: {val}")
    
    # Save
    filepath = f'{model_name}.pkl'
    pipeline.save(filepath)
    
    logger.info(f"✓ Pipeline ready for inference")
    
    return filepath


def load_and_infer_pipeline(model_path: str, X_stream: np.ndarray, 
                           y_stream: np.ndarray) -> Dict:
    """
    STEP 2: Load saved pipeline and make predictions on stream.
    
    Args:
        model_path: Path to saved .pkl file
        X_stream: Stream features
        y_stream: Stream targets
    
    Returns:
        results: Pipeline results
    """
    logger.info("\n" + "="*80)
    logger.info("STEP 2: LOADING & INFERENCE")
    logger.info("="*80)
    
    # Load
    pipeline = WaveletDriftDetectionPipeline.load(model_path)
    logger.info(f"✓ Pipeline loaded from {model_path}")
    
    # Infer
    results = pipeline.process_stream(X_stream, y_stream)
    
    logger.info(f"✓ Stream processed:")
    logger.info(f"  Predictions: {results['predictions_made']}")
    logger.info(f"  Drifts detected: {len(results['drifts_detected'])}")
    if results['drifts_detected']:
        logger.info(f"  Detection times: {results['drifts_detected']}")
    
    return results


def evaluate_detections(results: Dict, true_drift_times: list, 
                       output_file: str = 'evaluation_metrics.json') -> Dict:
    """
    STEP 3: Evaluate pipeline performance with ground truth.
    
    Args:
        results: Pipeline results
        true_drift_times: Ground truth drift times
        output_file: Where to save evaluation report
    
    Returns:
        metrics: Evaluation metrics
    """
    logger.info("\n" + "="*80)
    logger.info("STEP 3: EVALUATION")
    logger.info("="*80)
    
    evaluator = DriftDetectionEvaluator(tolerance=10)
    metrics = evaluator.evaluate(results, true_drift_times)
    
    # Print summary table
    print(evaluator.summary_table())
    
    # Save metrics
    evaluator.save_metrics(output_file)
    
    logger.info(f"✓ Report saved to {output_file}")
    
    return metrics


def run_complete_pipeline(X_warm: np.ndarray, y_warm: np.ndarray,
                         X_stream: np.ndarray, y_stream: np.ndarray,
                         true_drift_times: list,
                         config: Dict,
                         experiment_name: str = 'drift_detection_exp') -> Tuple[Dict, Dict]:
    """
    Run COMPLETE pipeline: Train → Save → Load → Infer → Evaluate.
    
    This is the full workflow combining Options 3 & 5.
    
    Args:
        X_warm: Warm-up features
        y_warm: Warm-up targets
        X_stream: Stream features
        y_stream: Stream targets
        true_drift_times: Ground truth drift times (for evaluation)
        config: Pipeline configuration
        experiment_name: Name for this experiment
    
    Returns:
        (results, metrics): Pipeline results and evaluation metrics
    
    Example:
        >>> config = {
        ...     'dwt': {'wavelet': 'db4', 'level': 4},
        ...     'detection': {'alpha': 0.05}
        ... }
        >>> results, metrics = run_complete_pipeline(
        ...     X_warm, y_warm, X_stream, y_stream,
        ...     true_drift_times=[75, 150],
        ...     config=config,
        ...     experiment_name='financial_drift_v1'
        ... )
        >>> print(f"F1-Score: {metrics['f1_score']:.3f}")
        >>> print(f"Detections: {results['drifts_detected']}")
    """
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = f'experiments/{experiment_name}_{timestamp}'
    
    os.makedirs(experiment_dir, exist_ok=True)
    
    logger.info("\n" + "="*80)
    logger.info(f"COMPLETE PIPELINE: {experiment_name}")
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Output directory: {experiment_dir}")
    logger.info("="*80)
    
    # ===== OPTION 3A: Train & Save =====
    model_path = f'{experiment_dir}/pipeline_model'
    train_and_save_pipeline(X_warm, y_warm, config, model_path)
    
    # ===== OPTION 3B: Load & Infer =====
    results = load_and_infer_pipeline(f'{model_path}.pkl', X_stream, y_stream)
    
    # ===== OPTION 5: Evaluate =====
    metrics = evaluate_detections(
        results, 
        true_drift_times,
        output_file=f'{experiment_dir}/evaluation_metrics.json'
    )
    
    # ===== Save results summary =====
    summary = {
        'experiment': experiment_name,
        'timestamp': timestamp,
        'model_path': f'{model_path}.pkl',
        'n_warm_up_samples': len(X_warm),
        'n_stream_samples': len(X_stream),
        'true_drift_times': true_drift_times,
        'detected_drift_times': results['drifts_detected'],
        'key_metrics': {
            'f1_score': float(metrics['f1_score']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'accuracy': float(metrics['accuracy']),
            'mean_latency': metrics['mean_latency_samples']
        }
    }
    
    with open(f'{experiment_dir}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n✓ EXPERIMENT COMPLETE")
    logger.info(f"✓ All outputs saved to {experiment_dir}/")
    
    return results, metrics


if __name__ == '__main__':
    """Complete end-to-end example."""
    
    np.random.seed(42)
    
    # Configuration
    config = {
        'dwt': {'wavelet': 'db4', 'level': 4},
        'detection': {'alpha': 0.05},
        'permutation': {'b_min': 100, 'b_max': 500}
    }
    
    # ===== Generate Synthetic Data =====
    print("\n" + "="*80)
    print("Generating synthetic data with known drift at t=75...")
    print("="*80)
    
    X_warm = np.random.randn(500, 5)
    y_warm = np.sum(X_warm, axis=1) + 0.1*np.random.randn(500)
    
    # Stream with mean shift at t=75
    X_stream = []
    y_stream = []
    for t in range(150):
        X_t = np.random.randn(5)
        if t < 75:
            y_t = np.sum(X_t) + 0.1*np.random.randn()  # Stable
        else:
            y_t = np.sum(X_t) + 2.0 + 0.1*np.random.randn()  # Mean shift (+2.0)
        X_stream.append(X_t)
        y_stream.append(y_t)
    
    X_stream = np.array(X_stream)
    y_stream = np.array(y_stream)
    
    print(f"✓ Data generated: 500 warm-up samples, 150 stream samples")
    print(f"✓ True drift at t=75 (target mean increases by 2.0)")
    
    # ===== Run Complete Pipeline =====
    results, metrics = run_complete_pipeline(
        X_warm, y_warm,
        X_stream, y_stream,
        true_drift_times=[75],
        config=config,
        experiment_name='mean_shift_detection'
    )
    
    # ===== Print Key Results =====
    print(f"\n{'='*80}")
    print("KEY RESULTS")
    print(f"{'='*80}")
    print(f"F1-Score:              {metrics['f1_score']:.4f}")
    print(f"Precision:             {metrics['precision']:.4f}")
    print(f"Recall:                {metrics['recall']:.4f}")
    print(f"Accuracy:              {metrics['accuracy']:.4f}")
    if metrics['mean_latency_samples'] is not None:
        print(f"Mean Detection Latency: {metrics['mean_latency_samples']:.1f} samples")
    print(f"Detections:            {results['drifts_detected']}")
    print(f"{'='*80}\n")