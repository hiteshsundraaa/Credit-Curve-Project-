import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_market_history(output_path="data/market_history.csv"):
    """
    Generates a synthetic multi-asset dataset.
    Nasdaq: Geometric Brownian Motion (GBM)
    VIX: Ornstein-Uhlenbeck (OU) process (Inversely correlated to Nasdaq)
    Spreads: Mean-reverting process tied to VIX
    """
    np.random.seed(42)
    n_days = 252
    dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(n_days)]
    
    # 1. Simulate Nasdaq-100 (GBM)
    nasdaq = [18000.0]
    dt = 1/252
    mu, sigma_q = 0.1, 0.2 
    for _ in range(n_days - 1):
        rtn = (mu - 0.5 * sigma_q**2) * dt + sigma_q * np.sqrt(dt) * np.random.normal()
        nasdaq.append(nasdaq[-1] * np.exp(rtn))
    
    # 2. Derive VIX (Inversely correlated to Nasdaq returns)
    nasdaq_pct = pd.Series(nasdaq).pct_change().fillna(0)
    # VIX spikes when Nasdaq drops
    vix = 20 - (nasdaq_pct * 500) + np.random.normal(0, 1.5, n_days)
    vix = np.clip(vix, 10, 80) 
    
    # 3. Generate 10Y Spreads (Correlated to VIX)
    spreads = 100 + (np.array(vix) - 20) * 2.5 + np.random.normal(0, 4, n_days)
    
    df = pd.DataFrame({
    "Date": dates,
    "Nasdaq_100": nasdaq,  # <--- Make sure this matches EXACTLY
    "VIX": vix,
    "Spread_10Y": spreads
})
    df.to_csv(output_path, index=False)
    return df

if __name__ == "__main__":
    # This allows you to run 'python src/data_generator.py' to test it
    generate_market_history()