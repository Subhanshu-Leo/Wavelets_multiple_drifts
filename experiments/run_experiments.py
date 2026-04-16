"""
Main experiment runner for drift detection evaluation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import yaml
from tqdm import tqdm

from experiments.synthetic_data import SyntheticDriftGenerator
from experiments.real_data_loader import RealDataExperiment
from src.pipeline.drift_pipeline import WaveletDriftDetectionPipeline
from src.detection.layer2_permutation import AdaptivePermutationTest
from utils.metrics import DriftMetrics
from utils.io import ResultsIO

class ExperimentRunner:
    """Run complete drift detection experiments."""
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """Initialize with config."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def run_synthetic_experiment(self, drift_type: str = 'sudden',
                                output_dir: str = 'results/synthetic/') -> dict:
        """
        Run experiment on synthetic data.
        
        Args:
            drift_type: 'sudden', 'gradual', 'incremental', 'recurring'
            output_dir: Results directory
        
        Returns:
            results dictionary
        """
        print(f"\n{'='*60}")
        print(f"Synthetic Drift Experiment: {drift_type}")
        print(f"{'='*60}")
        
        # Generate data
        print("1. Generating synthetic data...")
        gen = SyntheticDriftGenerator(seed=42)
        X, y, true_drifts = gen.generate(drift_type=drift_type, n=5000)
        print(f" True drifts at: {true_drifts}")
        
        # Initialize pipeline
        print("2. Initializing pipeline...")
        pipeline = WaveletDriftDetectionPipeline(self.config)
        
        # Warm-up
        print("3. Warm-up phase...")
        n_init = self.config['ensemble']['n_init']
        # TODO: Implement warm-up in pipeline
        
        # Run stream
        print("4. Processing stream...")
        detected_drifts = []
        predictions = []
        errors = []
        
        for t in tqdm(range(n_init, len(y)-1)):
            # Simplified: just compute error
            y_pred = 0.5 * y[t-1] + 0.1 * X[t, 0] + 0.05 * X[t, 1]
            y_true = y[t]
            
            error = np.abs(y_true - y_pred)
            predictions.append(y_pred)
            errors.append(error)
        
        # Evaluate
        print("5. Evaluating results...")
        if len(true_drifts) > 0:
            detection_delay = [
                min([detected_drifts[i] - true_drifts[j] 
                    for i, j in zip(range(len(detected_drifts)), 
                                    range(len(true_drifts)))])
                if detected_drifts else float('inf')
            ]
        else:
            detection_delay = []
        
        # Metrics
        metrics = {
            'drift_type': drift_type,
            'true_drifts': list(true_drifts),
            'detected_drifts': detected_drifts,
            'detection_delay': np.mean(detection_delay) if detection_delay else np.nan,
            'false_alarm_rate': len(detected_drifts) / len(y),
            'prequential_rmse': np.sqrt(np.mean(np.array(errors)**2)),
            'prequential_mae': np.mean(errors)
        }
        
        print("\nMetrics:")
        for key, val in metrics.items():
            if not isinstance(val, (list, dict)):
                print(f"  {key}: {val}")
        
        # Save results
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        result_file = Path(output_dir) / f'{drift_type}_results.json'
        ResultsIO.save_results(metrics, str(result_file))
        print(f" Results saved to {result_file}")
        
        return metrics
    
    def run_all_synthetic_experiments(self, output_dir: str = 'results/synthetic/') -> pd.DataFrame:
        """Run experiments on all drift types."""
        results = []
        
        for drift_type in ['sudden', 'gradual', 'incremental', 'recurring']:
            metrics = self.run_synthetic_experiment(drift_type, output_dir)
            results.append(metrics)
        
        # Summary table
        summary_df = pd.DataFrame(results)
        summary_file = Path(output_dir) / 'summary.csv'
        summary_df.to_csv(summary_file, index=False)
        print(f"\n Summary saved to {summary_file}")
        
        return summary_df
    
    def run_real_data_experiment(self, dataset: str = 'sp500',
                                filepath: str = None,
                                output_dir: str = 'results/real/') -> dict:
        """
        Run experiment on real-world data.
        
        Args:
            dataset: 'sp500' or 'electricity'
            filepath: Path to data file
            output_dir: Results directory
        
        Returns:
            results dictionary
        """
        print(f"\n{'='*60}")
        print(f"Real Data Experiment: {dataset}")
        print(f"{'='*60}")
        
        # Load data
        print("1. Loading data...")
        if dataset == 'sp500':
            (X_warm, y_warm), (X_eval, y_eval), metadata = \
                RealDataExperiment.prepare_sp500(filepath)
        elif dataset == 'electricity':
            (X_warm, y_warm), (X_eval, y_eval), metadata = \
                RealDataExperiment.prepare_electricity(filepath)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        # Pipeline
        print("2. Initializing pipeline...")
        pipeline = WaveletDriftDetectionPipeline(self.config)
        
        # Warm-up
        print("3. Warm-up phase...")
        # TODO: Implement warm-up
        
        # Stream
        print("4. Processing stream...")
        predictions = []
        errors = []
        
        for t in tqdm(range(len(X_eval))):
            y_pred = np.mean(y_warm)  # Baseline: use warm-up mean
            y_true = y_eval[t]
            
            predictions.append(y_pred)
            errors.append(np.abs(y_true - y_pred))
        
        # Metrics
        metrics = {
            'dataset': dataset,
            'prequential_rmse': np.sqrt(np.mean(np.array(errors)**2)),
            'prequential_mae': np.mean(errors),
            'eval_samples': len(X_eval),
            **metadata
        }
        
        print("\nMetrics:")
        for key, val in metrics.items():
            if not isinstance(val, dict):
                print(f"  {key}: {val}")
        
        # Save
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        result_file = Path(output_dir) / f'{dataset}_results.json'
        ResultsIO.save_results(metrics, str(result_file))
        print(f" Results saved to {result_file}")
        
        return metrics


# Example
if __name__ == '__main__':
    runner = ExperimentRunner('config/config.yaml')
    
    # Run synthetic experiments
    print("Starting Synthetic Experiments...")
    synthetic_results = runner.run_all_synthetic_experiments()
    print("\nSynthetic Results Summary:")
    print(synthetic_results[['drift_type', 'detection_delay', 'prequential_mae']])
    
    # Run real data experiments (if data available)
    try:
        print("\n\nStarting Real Data Experiments...")
        # Uncomment if data is available
        # real_results = runner.run_real_data_experiment(dataset='sp500', filepath='data/sp500.csv')
    except Exception as e:
        print(f"Could not run real data experiments: {e}")