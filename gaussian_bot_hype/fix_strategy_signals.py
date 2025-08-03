#!/usr/bin/env python3
"""
Fix for Gaussian Channel Strategy - Resample to daily and fix signal logic
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock

# Add current directory to path
sys.path.append('.')

from strategy.gaussian_filter import GaussianChannelFilter
from strategy.signals import SignalGenerator

def create_test_config():
    """Create a test configuration"""
    config = Mock()
    
    # Strategy parameters (matching Pine Script defaults)
    config.POLES = 6
    config.PERIOD = 144
    config.MULTIPLIER = 1.414
    config.ATR_SPACING = 0.4
    config.MAX_PYRAMIDS = 5
    config.POSITION_SIZE_PCT = 1.0
    
    # Mock API keys
    config.HYPERLIQUID_API_KEY = "test_key"
    config.HYPERLIQUID_SECRET_KEY = "test_secret"
    
    return config

def resample_to_daily(data):
    """Resample 1-minute data to daily bars"""
    print("🔄 Resampling 1-minute data to daily bars...")
    
    # Ensure we have datetime index
    if not isinstance(data.index, pd.DatetimeIndex):
        data['date'] = pd.to_datetime(data['unix'], unit='ms')
        data.set_index('date', inplace=True)
    
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
    
    return daily_data

def load_and_resample_data():
    """Load and resample data to daily timeframe"""
    print("📊 Loading and resampling data...")
    
    csv_path = os.path.join('data', 'historical_data.csv')
    if not os.path.exists(csv_path):
        print(f"❌ Data file not found: {csv_path}")
        return None
    
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
    
    # Resample to daily
    daily_data = resample_to_daily(data)
    
    return daily_data

def fix_signal_generation_logic():
    """Test fixed signal generation logic"""
    print("\n🔧 Testing Fixed Signal Generation Logic...")
    
    # Load and resample data
    data = load_and_resample_data()
    if data is None:
        return
    
    # Initialize components
    config = create_test_config()
    gaussian_filter = GaussianChannelFilter(
        poles=config.POLES,
        period=config.PERIOD,
        multiplier=config.MULTIPLIER
    )
    
    # Calculate indicators
    data['hlc3'] = (data['High'] + data['Low'] + data['Close']) / 3
    
    # True Range calculation
    data['tr1'] = data['High'] - data['Low']
    data['tr2'] = abs(data['High'] - data['Close'].shift(1))
    data['tr3'] = abs(data['Low'] - data['Close'].shift(1))
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Apply Gaussian filter
    filt, hband, lband = gaussian_filter.apply_filter(data['hlc3'], data['true_range'])
    data['filt'] = filt
    data['hband'] = hband
    data['lband'] = lband
    
    # FIXED SIGNAL LOGIC - More conservative approach
    # Green channel condition (filter rising)
    data['green_channel'] = (data['filt'] > data['filt'].shift(1)).fillna(False)
    
    # Entry conditions - More conservative
    # Only enter if price closes significantly above the band
    band_threshold = 0.001  # 0.1% above band minimum
    
    data['green_entry'] = (
        data['green_channel'] & 
        (data['Close'] > data['hband'] * (1 + band_threshold)) &
        (data['Close'] > data['Close'].shift(1))  # Price must be rising
    )
    
    data['red_entry'] = (
        ~data['green_channel'] & 
        (data['Close'] > data['hband'] * (1 + band_threshold)) &
        (data['Close'] > data['Close'].shift(1))  # Price must be rising
    )
    
    # Exit condition - More conservative
    data['exit_signal'] = (
        (data['Close'] < data['hband']) |
        (data['Close'] < data['Close'].shift(1) * 0.995)  # 0.5% stop loss
    )
    
    # Add minimum holding period
    data['days_since_entry'] = 0
    entry_dates = data[data['green_entry'] | data['red_entry']].index
    
    for i, entry_date in enumerate(entry_dates):
        if i < len(entry_dates) - 1:
            next_entry = entry_dates[i + 1]
            days_between = (next_entry - entry_date).days
            if days_between < 7:  # Minimum 7 days between entries
                # Remove the second entry
                data.loc[next_entry, 'green_entry'] = False
                data.loc[next_entry, 'red_entry'] = False
    
    # Analyze results
    total_entries = data['green_entry'].sum() + data['red_entry'].sum()
    total_exits = data['exit_signal'].sum()
    
    print(f"\n📊 Fixed Signal Analysis:")
    print(f"Total entries: {total_entries}")
    print(f"Total exits: {total_exits}")
    print(f"Entry rate: {total_entries / len(data):.4f}")
    
    # Monthly analysis
    data['month'] = data.index.to_period('M')
    monthly_signals = data.groupby('month').agg({
        'green_entry': 'sum',
        'red_entry': 'sum',
        'exit_signal': 'sum'
    }).reset_index()
    
    monthly_signals['total_entries'] = monthly_signals['green_entry'] + monthly_signals['red_entry']
    
    avg_signals_per_month = monthly_signals['total_entries'].mean()
    max_signals_per_month = monthly_signals['total_entries'].max()
    
    print(f"Average signals per month: {avg_signals_per_month:.2f}")
    print(f"Max signals per month: {max_signals_per_month}")
    
    # Show recent months
    recent_months = monthly_signals.tail(12)
    print(f"\n📊 Recent 12 months (Fixed Logic):")
    for _, row in recent_months.iterrows():
        print(f"   {row['month']}: {row['total_entries']} signals")
    
    return data, monthly_signals

def compare_before_after():
    """Compare before and after fixes"""
    print("\n🔄 Comparing Before vs After Fixes...")
    
    # Get fixed results
    fixed_data, fixed_monthly = fix_signal_generation_logic()
    
    if fixed_monthly is not None:
        fixed_avg = fixed_monthly['total_entries'].mean()
        fixed_max = fixed_monthly['total_entries'].max()
        
        print(f"\n📈 IMPROVEMENT SUMMARY:")
        print(f"Before: 18.47 signals/month average, 504 max")
        print(f"After:  {fixed_avg:.2f} signals/month average, {fixed_max} max")
        
        improvement = 18.47 / fixed_avg if fixed_avg > 0 else float('inf')
        print(f"Improvement: {improvement:.1f}x reduction in signals")
        
        if fixed_avg < 1.0:
            print("✅ EXCELLENT: Signal count now in reasonable range!")
        elif fixed_avg < 3.0:
            print("⚠️  BETTER: Signal count improved but still high")
        else:
            print("❌ NEEDS MORE WORK: Signal count still too high")

def save_fixed_data():
    """Save the resampled daily data"""
    print("\n💾 Saving resampled daily data...")
    
    data = load_and_resample_data()
    if data is not None:
        output_path = os.path.join('data', 'historical_data_daily.csv')
        data.to_csv(output_path)
        print(f"✅ Saved daily data to: {output_path}")
        print(f"📊 {len(data)} daily bars saved")

if __name__ == "__main__":
    print("🔧 GAUSSIAN CHANNEL STRATEGY FIXES")
    print("=" * 50)
    
    # Run fixes
    compare_before_after()
    save_fixed_data()
    
    print("\n" + "=" * 50)
    print("FIXES COMPLETE")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Use the resampled daily data for backtesting")
    print("2. Update your strategy to use the fixed signal logic")
    print("3. Test with the new parameters") 