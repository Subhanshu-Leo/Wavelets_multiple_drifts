"""
Visualization utilities for drift detection results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class DriftVisualizer:
    """Visualize drift detection results."""
    
    @staticmethod
    def plot_signal_with_drifts(signal: np.ndarray,
                               drift_times: list,
                               figsize: tuple = (12, 4)):
        """Plot signal with detected drifts marked."""
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(signal, 'b-', alpha=0.7, label='Signal')
        
        for drift_t in drift_times:
            ax.axvline(drift_t, color='r', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title('Signal with Detected Drifts')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, ax
    
    @staticmethod
    def plot_wavelet_energies(energies_hist: np.ndarray,
                             energies_new: np.ndarray,
                             scale_labels: list = None,
                             figsize: tuple = (10, 4)):
        """Plot per-scale wavelet energies (historical vs new)."""
        n_scales = len(energies_hist)
        
        if scale_labels is None:
            scale_labels = [f'Scale {j}' for j in range(n_scales)]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(n_scales)
        width = 0.35
        
        ax.bar(x - width/2, energies_hist, width, label='Historical', alpha=0.7)
        ax.bar(x + width/2, energies_new, width, label='New', alpha=0.7)
        
        ax.set_xlabel('Scale')
        ax.set_ylabel('Energy')
        ax.set_title('Wavelet Energy Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(scale_labels)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig, ax
    
    @staticmethod
    def plot_detection_statistics(delta_j_history: np.ndarray,
                                 threshold: float,
                                 drift_times: list = None,
                                 figsize: tuple = (12, 4)):
        """Plot drift statistic δ_j over time."""
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(delta_j_history, 'b-', alpha=0.7, label='δ_j')
        ax.axhline(threshold, color='r', linestyle='--', label=f'Threshold={threshold:.2f}')
        
        if drift_times:
            for drift_t in drift_times:
                ax.axvline(drift_t, color='g', linestyle=':', alpha=0.5)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Drift Statistic δ_j')
        ax.set_title('Drift Detection Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, ax
    
    @staticmethod
    def plot_prequential_error(errors: np.ndarray,
                              window: int = 50,
                              figsize: tuple = (12, 4)):
        """Plot prequential error with moving average."""
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(errors, 'b-', alpha=0.3, label='Error')
        
        # Moving average
        moving_avg = np.convolve(errors, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(errors)), moving_avg, 'r-', linewidth=2,
               label=f'MA(window={window})')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Error')
        ax.set_title('Prequential Error')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, ax


# Example
if __name__ == '__main__':
    signal = np.sin(2*np.pi*np.arange(1000)/100)
    signal[500:] = np.sin(2*np.pi*np.arange(500)/50)  # Change
    
    viz = DriftVisualizer()
    fig, ax = viz.plot_signal_with_drifts(signal, [500])
    plt.show()