"""
Input/output utilities for data loading and saving.
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from typing import Dict, Tuple, Any

class DataIO:
    """Handle data loading and saving operations."""
    
    @staticmethod
    def load_csv(filepath: str, **kwargs) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            filepath: Path to CSV file
            **kwargs: Additional pandas.read_csv arguments
        
        Returns:
            DataFrame
        """
        return pd.read_csv(filepath, **kwargs)
    
    @staticmethod
    def save_csv(data: pd.DataFrame, filepath: str, **kwargs):
        """
        Save data to CSV file.
        
        Args:
            data: DataFrame to save
            filepath: Output path
            **kwargs: Additional pandas.to_csv arguments
        """
        data.to_csv(filepath, **kwargs)
        print(f"✓ Saved to {filepath}")
    
    @staticmethod
    def load_npy(filepath: str) -> np.ndarray:
        """Load numpy array."""
        return np.load(filepath)
    
    @staticmethod
    def save_npy(array: np.ndarray, filepath: str):
        """Save numpy array."""
        np.save(filepath, array)
        print(f"✓ Saved to {filepath}")
    
    @staticmethod
    def load_pickle(filepath: str) -> Any:
        """Load pickled object."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def save_pickle(obj: Any, filepath: str):
        """Save object as pickle."""
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
        print(f"✓ Saved to {filepath}")
    
    @staticmethod
    def load_json(filepath: str) -> Dict:
        """Load JSON file."""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def save_json(data: Dict, filepath: str, indent: int = 2):
        """Save dictionary as JSON."""
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=indent)
        print(f"✓ Saved to {filepath}")


class FinancialDataLoader:
    """Load and preprocess real-world financial data."""
    
    @staticmethod
    def load_sp500(filepath: str = None, start_date: str = '2015-01-01',
                   end_date: str = '2023-12-31') -> Tuple[np.ndarray, pd.DatetimeIndex]:
        """
        Load S&P 500 data.
        
        Args:
            filepath: Path to CSV (if None, download from Yahoo)
            start_date: Start date
            end_date: End date
        
        Returns:
            (prices, dates)
        """
        if filepath is not None:
            df = pd.read_csv(filepath, parse_dates=['Date'])
        else:
            # Try to download from Yahoo Finance
            try:
                import yfinance as yf
                sp500 = yf.download('^GSPC', start=start_date, end=end_date)
                df = sp500.reset_index()
            except:
                raise RuntimeError("yfinance not installed. Provide filepath instead.")
        
        df = df.sort_values('Date')
        prices = df['Adj Close'].values
        dates = df['Date'].values
        
        return prices, dates
    
    @staticmethod
    def compute_log_returns(prices: np.ndarray) -> np.ndarray:
        """
        Compute log returns from prices.
        
        Args:
            prices: Price series
        
        Returns:
            Log returns
        """
        return np.diff(np.log(prices))
    
    @staticmethod
    def demeaning(returns: np.ndarray, window: int = 250) -> np.ndarray:
        """
        Demean returns using rolling window.
        
        Args:
            returns: Return series
            window: Rolling window size
        
        Returns:
            Demeaned returns
        """
        rolling_mean = pd.Series(returns).rolling(window=window).mean()
        return returns - rolling_mean.values
    
    @staticmethod
    def create_features(returns: np.ndarray, ar_lags: int = 5,
                       vol_window: int = 20,
                       momentum_window: int = 20) -> np.ndarray:
        """
        Create feature matrix from returns.
        
        Args:
            returns: Log returns
            ar_lags: Number of AR lags
            vol_window: Volatility window
            momentum_window: Momentum window
        
        Returns:
            Feature matrix (n_samples, n_features)
        """
        n = len(returns)
        
        # AR lags
        ar_features = np.zeros((n, ar_lags))
        for lag in range(1, ar_lags + 1):
            ar_features[lag:, lag-1] = returns[:-lag]
        
        # Volatility (rolling std)
        vol = pd.Series(returns).rolling(window=vol_window).std().values
        
        # Momentum (cumulative return)
        momentum = pd.Series(returns).rolling(window=momentum_window).sum().values
        
        # Concatenate
        features = np.column_stack([ar_features, vol, momentum])
        
        # Remove NaN rows
        mask = ~np.isnan(features).any(axis=1)
        features = features[mask]
        
        return features
    
    @staticmethod
    def prepare_sp500_data(filepath: str = None,
                          ar_lags: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full S&P 500 preparation pipeline.
        
        Args:
            filepath: Path to data
            ar_lags: Number of AR lags
        
        Returns:
            (X, y) where X is features, y is next-period returns
        """
        # Load
        prices, dates = FinancialDataLoader.load_sp500(filepath)
        
        # Log returns
        returns = FinancialDataLoader.compute_log_returns(prices)
        
        # Demean
        returns_demeaned = FinancialDataLoader.demeaning(returns)
        
        # Features
        X = FinancialDataLoader.create_features(returns_demeaned, ar_lags=ar_lags)
        
        # Target: next-period return
        y = returns_demeaned[ar_lags:-1][:len(X)]
        
        return X, y


class ElectricityDataLoader:
    """Load electricity load dataset (UCI)."""
    
    @staticmethod
    def load_electricity(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load UCI electricity dataset.
        
        Args:
            filepath: Path to electricity.csv
        
        Returns:
            (X, y) - X: (n, 8), y: (n,)
        """
        df = pd.read_csv(filepath)
        
        # Last column is target (load)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        # Normalize to zero mean, unit variance
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X = (X - X_mean) / (X_std + 1e-10)
        
        y_mean = np.mean(y)
        y_std = np.std(y)
        y = (y - y_mean) / (y_std + 1e-10)
        
        return X, y


class ResultsIO:
    """Save and load experiment results."""
    
    @staticmethod
    def save_results(results: Dict[str, Any],
                    filepath: str,
                    format: str = 'json'):
        """
        Save experiment results.
        
        Args:
            results: Results dictionary
            filepath: Output path
            format: 'json' or 'pickle'
        """
        if format == 'json':
            # Convert numpy arrays to lists for JSON serialization
            results_json = {}
            for key, val in results.items():
                if isinstance(val, np.ndarray):
                    results_json[key] = val.tolist()
                elif isinstance(val, (list, dict)):
                    results_json[key] = val
                else:
                    results_json[key] = float(val)
            
            DataIO.save_json(results_json, filepath)
        
        elif format == 'pickle':
            DataIO.save_pickle(results, filepath)
    
    @staticmethod
    def load_results(filepath: str,
                    format: str = 'json') -> Dict[str, Any]:
        """Load results."""
        if format == 'json':
            return DataIO.load_json(filepath)
        elif format == 'pickle':
            return DataIO.load_pickle(filepath)
    
    @staticmethod
    def save_metrics_table(metrics: Dict[str, Dict[str, float]],
                          filepath: str):
        """
        Save metrics as CSV table.
        
        Args:
            metrics: {method_name: {metric_name: value}}
            filepath: Output CSV path
        """
        df = pd.DataFrame(metrics).T
        DataIO.save_csv(df, filepath, index_label='Method')


# Example
if __name__ == '__main__':
    # Test CSV I/O
    data = pd.DataFrame({
        'a': np.random.randn(10),
        'b': np.random.randn(10)
    })
    DataIO.save_csv(data, '/tmp/test.csv')
    data_loaded = DataIO.load_csv('/tmp/test.csv', index_col=0)
    print(f"✓ CSV I/O works: {data_loaded.shape}")
    
    # Test pickle I/O
    obj = {'key': [1, 2, 3]}
    DataIO.save_pickle(obj, '/tmp/test.pkl')
    obj_loaded = DataIO.load_pickle('/tmp/test.pkl')
    print(f"✓ Pickle I/O works: {obj_loaded}")