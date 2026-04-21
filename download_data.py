import os
import yfinance as yf
import pandas as pd

# Create a data folder if it doesn't exist
os.makedirs('data', exist_ok=True)

print("Downloading S&P 500 data...")
# Download S&P 500 index
df = yf.download('^GSPC', start='2015-01-01', end='2024-01-01')

# FIX: yfinance now uses double-headers. We need to flatten them.
if hasattr(df.columns, 'droplevel'):
    try:
        # Drop the second header row (which contains the ticker name)
        df.columns = df.columns.droplevel(1)
    except Exception:
        pass

# Move 'Date' from the index into a regular column
df = df.reset_index()

# Save as a clean CSV
df.to_csv('data/sp500.csv', index=False)
print("Saved cleanly to data/sp500.csv!")