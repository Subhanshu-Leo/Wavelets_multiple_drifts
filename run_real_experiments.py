import yaml
from experiments.real_data_loader import run_real_data_comparison
from src.pipeline.drift_pipeline import WaveletDriftDetectionPipeline

def load_yaml_config(filepath):
    """Helper function to load the config.yaml file"""
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)

def main():
    print("Loading configuration...")
    # 1. Load your config file
    config = load_yaml_config('config/config.yaml')
    
    # 2. Initialize your custom pipeline with the loaded config
    my_pipeline = WaveletDriftDetectionPipeline(config=config)
    
    # 3. Run the ultimate showdown on the S&P 500 data!
    print("Starting real data experiments...")
    results = run_real_data_comparison(
        filepath='data/sp500.csv',
        dataset='sp500',
        pipeline=my_pipeline,
        tolerance=200,      # Real data is messy, so we use a wider tolerance window
        warmup_ratio=0.20   # Use the first 20% of the data to warm up the models
    )

if __name__ == "__main__":
    main()