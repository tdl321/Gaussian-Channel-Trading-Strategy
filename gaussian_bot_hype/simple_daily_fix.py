#!/usr/bin/env python3
"""
Simple fix: Just resample data to daily bars - no strategy changes
"""

import sys
import os
import pandas as pd
from datetime import datetime

# Add current directory to path
sys.path.append('.')

def resample_to_daily():
    """Simply resample 1-minute data to daily bars"""
    print("📊 Loading and resampling data to daily bars...")
    
    csv_path = os.path.join('data', 'historical_data.csv')
    if not os.path.exists(csv_path):
        print(f"❌ Data file not found: {csv_path}")
        return
    
    # Load data
    data = pd.read_csv(csv_path)
    
    # Convert timestamp to datetime
    data['date'] = pd.to_datetime(data['unix'], unit='ms')
    data.set_index('date', inplace=True)
    
    # Rename columns
    data = data.rename(columns={
        'open': 'Open',
        'high': 'High', 
        'low': 'Low',
        'close': 'Close',
        'Volume BTC': 'Volume'
    })
    
    # Sort by date
    data = data.sort_index()
    
    # Resample to daily OHLCV
    daily_data = data.resample('D').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    
    print(f"✅ Resampled {len(data)} 1-minute bars to {len(daily_data)} daily bars")
    print(f"📅 Daily data range: {daily_data.index[0]} to {daily_data.index[-1]}")
    
    # Save daily data
    output_path = os.path.join('data', 'historical_data_daily.csv')
    daily_data.to_csv(output_path)
    print(f"✅ Saved daily data to: {output_path}")
    
    return daily_data

if __name__ == "__main__":
    print("🔄 SIMPLE DAILY DATA FIX")
    print("=" * 40)
    
    daily_data = resample_to_daily()
    
    print("\n" + "=" * 40)
    print("DONE - Use historical_data_daily.csv for backtesting")
    print("=" * 40) 