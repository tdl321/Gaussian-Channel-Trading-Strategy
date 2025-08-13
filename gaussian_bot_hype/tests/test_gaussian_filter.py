#!/usr/bin/env python3
"""
Simple test for Gaussian Channel Filter
"""

import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path to import strategy modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy.gaussian_filter import GaussianChannelFilter


def test_gaussian_filter():
    """Test that Gaussian filter works with live trading parameters"""
    
    # Create test data (similar to what we'd get from Hyperliquid)
    dates = pd.date_range('2024-01-01', periods=200, freq='D')
    np.random.seed(42)  # For reproducible results
    
    # Generate realistic price data
    base_price = 45000
    prices = []
    for i in range(200):
        # Random walk with some trend
        if i == 0:
            price = base_price
        else:
            change = np.random.normal(0, 0.02)  # 2% daily volatility
            price = prices[-1] * (1 + change)
        prices.append(price)
    
    # Create OHLCV data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Simple OHLC from close price
        high = close * (1 + abs(np.random.normal(0, 0.01)))
        low = close * (1 - abs(np.random.normal(0, 0.01)))
        open_price = close * (1 + np.random.normal(0, 0.005))
        volume = np.random.uniform(1000, 10000)
        
        data.append({
            'Date': date,
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    
    # Calculate HLC3 and True Range (same as in strategy)
    df['hlc3'] = (df['High'] + df['Low'] + df['Close']) / 3
    
    df['tr1'] = df['High'] - df['Low']
    df['tr2'] = abs(df['High'] - df['Close'].shift(1))
    df['tr3'] = abs(df['Low'] - df['Close'].shift(1))
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Initialize Gaussian filter with live trading parameters
    gaussian_filter = GaussianChannelFilter(
        poles=6,        # Live trading parameter
        period=144,     # Live trading parameter
        multiplier=1.414  # Live trading parameter
    )
    
    # Apply filter
    filt_result, hband_result, lband_result = gaussian_filter.apply_filter(
        df['hlc3'], 
        df['true_range']
    )
    
    # Check that we get results
    assert not filt_result.isna().all(), "Filter should produce non-NaN results"
    assert not hband_result.isna().all(), "Upper band should produce non-NaN results"
    assert not lband_result.isna().all(), "Lower band should produce non-NaN results"
    
    # Check that bands are reasonable
    valid_mask = ~filt_result.isna()
    if valid_mask.sum() > 0:
        assert (hband_result[valid_mask] > lband_result[valid_mask]).all(), "Upper band should be above lower band"
        assert (hband_result[valid_mask] > filt_result[valid_mask]).all(), "Upper band should be above filter"
        assert (lband_result[valid_mask] < filt_result[valid_mask]).all(), "Lower band should be below filter"
    
    print("✅ Gaussian filter test passed!")
    print(f"   - Generated {len(df)} bars of test data")
    print(f"   - Filter parameters: {gaussian_filter.poles} poles, {gaussian_filter.period} period, {gaussian_filter.multiplier} multiplier")
    print(f"   - Valid filter values: {valid_mask.sum()}/{len(df)}")
    
    return True


if __name__ == "__main__":
    test_gaussian_filter() 