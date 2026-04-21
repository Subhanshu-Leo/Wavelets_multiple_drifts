import os
import yfinance as yf

# Create a data folder if it doesn't exist
os.makedirs('data', exist_ok=True)

print("Downloading S&P 500 data...")
# Download S&P 500 index from 2015 to 2024
df = yf.download('^GSPC', start='2015-01-01', end='2024-01-01')
df.to_csv('data/sp500.csv')
print("Saved to data/sp500.csv!")