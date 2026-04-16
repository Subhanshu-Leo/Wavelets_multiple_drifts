"""
Drift Detection Evaluation Module (Option 5)

Evaluate pipeline performance on synthetic or real data with known drift times.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class DriftDetectionEvaluator:
    """Evaluate drift detection pipeline performance."""
    
    def __init__(self, tolerance: int = 10):
        """
        Initialize evaluator.
        
        Args:
            tolerance: Detection window (detect within ±N samples of true drift)
        """
        self.tolerance = tolerance
        self.metrics = {}
    
    def evaluate(self, results: Dict, true_drift_times: List[int]) -> Dict:
        """
        Evaluate detection quality against ground truth.
        
        Args:
            results: Pipeline output from process_stream()
            true_drift_times: List of true drift times (ground truth)
        
        Returns:
            Comprehensive metrics dict
        """
        detected = set(results['drifts_detected'])
        true_drifts = set(true_drift_times)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"DRIFT DETECTION EVALUATION")
        logger.info(f"{'='*80}")
        logger.info(f"True drift times: {sorted(true_drifts)}")
        logger.info(f"Detected times: {sorted(detected)}")
        logger.info(f"Tolerance: ±{self.tolerance} samples")
        
        # ===== TRUE POSITIVES, FALSE POSITIVES, FALSE NEGATIVES =====
        
        correctly_detected = []
        missed_drifts = []
        
        # For each true drift, check if detected within tolerance
        for true_drift in true_drifts:
            detected_nearby = [
                d for d in detected 
                if abs(d - true_drift) <= self.tolerance
            ]
            
            if detected_nearby:
                correctly_detected.append((true_drift, detected_nearby[0]))
                logger.info(f" Drift at t={true_drift} detected at t={detected_nearby[0]}")
            else:
                missed_drifts.append(true_drift)
                logger.warning(f"✗ Drift at t={true_drift} MISSED")
        
        # False positives: detections not near any true drift
        false_positives = []
        for det in detected:
            if not any(abs(det - td) <= self.tolerance for td in true_drifts):
                false_positives.append(det)
                logger.warning(f"✗ False positive at t={det}")
        
        # ===== METRICS =====
        
        n_tp = len(correctly_detected)
        n_fp = len(false_positives)
        n_fn = len(missed_drifts)
        n_tn = results['predictions_made'] - n_tp - n_fp - n_fn
        
        # Precision, Recall, F1
        precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0.0
        recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Specificity, Accuracy
        specificity = n_tn / (n_tn + n_fp) if (n_tn + n_fp) > 0 else 0.0
        accuracy = (n_tp + n_tn) / results['predictions_made']
        
        # False positive rate, False negative rate
        fpr = n_fp / (n_fp + n_tn) if (n_fp + n_tn) > 0 else 0.0
        fnr = n_fn / (n_fn + n_tp) if (n_fn + n_tp) > 0 else 0.0
        
        # Detection latency (samples after drift before detection)
        detection_latencies = [
            det - true_drift 
            for true_drift, det in correctly_detected
        ]
        mean_latency = np.mean(detection_latencies) if detection_latencies else np.nan
        max_latency = np.max(detection_latencies) if detection_latencies else np.nan
        
        self.metrics = {
            # Confusion matrix
            'true_positives': n_tp,
            'false_positives': n_fp,
            'false_negatives': n_fn,
            'true_negatives': n_tn,
            
            # Classification metrics
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'specificity': specificity,
            'accuracy': accuracy,
            'fpr': fpr,
            'fnr': fnr,
            
            # Detection latency
            'mean_latency_samples': mean_latency,
            'max_latency_samples': max_latency,
            'latencies': detection_latencies,
            
            # Counts
            'n_true_drifts': len(true_drifts),
            'n_detected_drifts': len(detected),
            'n_predictions': results['predictions_made'],
            
            # Details
            'correctly_detected': correctly_detected,
            'missed_drifts': missed_drifts,
            'false_positives': false_positives,
        }
        
        self._log_metrics()
        
        return self.metrics
    
    def _log_metrics(self):
        """Log formatted metrics."""
        m = self.metrics
        
        logger.info(f"\n{'─'*80}")
        logger.info(f"CLASSIFICATION METRICS")
        logger.info(f"{'─'*80}")
        logger.info(f"Precision:  {m['precision']:.4f}  (TP / (TP + FP))")
        logger.info(f"Recall:     {m['recall']:.4f}  (TP / (TP + FN))")
        logger.info(f"F1-Score:   {m['f1_score']:.4f}  (harmonic mean)")
        logger.info(f"Accuracy:   {m['accuracy']:.4f}  ((TP + TN) / Total)")
        logger.info(f"Specificity:{m['specificity']:.4f}  (TN / (TN + FP))")
        
        logger.info(f"\n{'─'*80}")
        logger.info(f"ERROR RATES")
        logger.info(f"{'─'*80}")
        logger.info(f"False Positive Rate: {m['fpr']:.4f}  (FP / (FP + TN))")
        logger.info(f"False Negative Rate: {m['fnr']:.4f}  (FN / (FN + TP))")
        
        logger.info(f"\n{'─'*80}")
        logger.info(f"DETECTION LATENCY (samples after drift)")
        logger.info(f"{'─'*80}")
        logger.info(f"Mean Latency: {m['mean_latency_samples']:.1f} samples")
        logger.info(f"Max Latency:  {m['max_latency_samples']:.1f} samples")
        
        logger.info(f"\n{'─'*80}")
        logger.info(f"CONFUSION MATRIX")
        logger.info(f"{'─'*80}")
        logger.info(f"TP (Correctly Detected): {m['true_positives']}")
        logger.info(f"FP (False Alarms):       {m['false_positives']}")
        logger.info(f"FN (Missed Drifts):      {m['false_negatives']}")
        logger.info(f"TN (Correct Negatives):  {m['true_negatives']}")
    
    def save_metrics(self, filepath: str) -> None:
        """Save metrics to JSON file."""
        # Convert numpy types to native Python types
        metrics_serializable = {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in self.metrics.items()
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics_serializable, f, indent=2, default=str)
        
        logger.info(f" Metrics saved to {filepath}")
    
    def summary_table(self) -> str:
        """Return formatted summary table."""
        m = self.metrics
        
        latency_str = f"{m['mean_latency_samples']:.1f}" if m['mean_latency_samples'] is not None else "N/A"
        n_fp = len(m['false_positives']) if isinstance(m['false_positives'], list) else m['false_positives']
        
        table = f"""
╔══════════════════════════════════════════════════════════╗
║         DRIFT DETECTION PERFORMANCE SUMMARY              ║
╠══════════════════════════════════════════════════════════╣
║ Metric              │ Value        │ Interpretation      ║
├─────────────────────┼──────────────┼─────────────────────┤
║ F1-Score            │ {m['f1_score']:8.4f}    │ Overall balance     ║
║ Precision           │ {m['precision']:8.4f}    │ False alarm rate    ║
║ Recall              │ {m['recall']:8.4f}    │ Miss rate           ║
║ Accuracy            │ {m['accuracy']:8.4f}    │ Overall correctness ║
├─────────────────────┼──────────────┼─────────────────────┤
║ Mean Latency        │ {latency_str:>8}    │ Detection speed     ║
║ True Positives      │ {m['true_positives']:8d}    │ Correct detections  ║
║ False Positives     │ {n_fp:8d}    │ False alarms        ║
║ False Negatives     │ {m['false_negatives']:8d}    │ Missed drifts       ║
╚══════════════════════════════════════════════════════════╝
"""
        return table


# Add numpy import at top of file
import numpy as np


if __name__ == '__main__':
    """Example usage of evaluator."""
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*80)
    print("DRIFT DETECTION EVALUATOR - EXAMPLE")
    print("="*80)
    
    # Simulate pipeline results
    results = {
        'predictions_made': 200,
        'drifts_detected': [95, 96, 97, 150, 151, 160]  # Some false positives
    }
    
    # Ground truth
    true_drift_times = [90, 150]  # Two true drifts
    
    # Evaluate
    evaluator = DriftDetectionEvaluator(tolerance=5)
    metrics = evaluator.evaluate(results, true_drift_times)
    
    print(evaluator.summary_table())
    
    # Save metrics
    evaluator.save_metrics('drift_detection_metrics.json')