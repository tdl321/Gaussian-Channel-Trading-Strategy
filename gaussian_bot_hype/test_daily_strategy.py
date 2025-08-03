#!/usr/bin/env python3
"""
Test script for daily timeframe Gaussian Channel strategy
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
    """Create a test configuration for daily strategy"""
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

def load_daily_data():
    """Load and prepare daily data"""
    print("📊 Loading daily historical data...")
    
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
    
    # Sort by date (oldest first)
    data = data.sort_index()
    
    # Verify this is daily data
    time_diff = data.index[1] - data.index[0]
    print(f"Time difference between bars: {time_diff}")
    
    if time_diff.days == 1:
        print("✅ Confirmed: This is daily data")
    else:
        print(f"⚠️  WARNING: Expected daily data, got {time_diff}")
    
    print(f"✅ Loaded {len(data)} daily bars from {data.index[0]} to {data.index[-1]}")
    print(f"📅 Total time period: {(data.index[-1] - data.index[0]).days} days")
    
    return data

def test_daily_strategy_signals():
    """Test strategy signals on daily data"""
    print("\n🎯 Testing Daily Strategy Signals...")
    
    # Load data
    data = load_daily_data()
    if data is None:
        return
    
    # Initialize components
    config = create_test_config()
    gaussian_filter = GaussianChannelFilter(
        poles=config.POLES,
        period=config.PERIOD,
        multiplier=config.MULTIPLIER
    )
    signal_generator = SignalGenerator(gaussian_filter, config)
    
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
    
    # Prepare signals
    data_with_signals = signal_generator.prepare_signals(data)
    
    # Analyze signals
    signal_summary = signal_generator.get_signal_summary(data_with_signals)
    print(f"Signal Summary: {signal_summary}")
    
    # Monthly analysis
    monthly_signals = analyze_monthly_signals(data_with_signals)
    
    return data_with_signals, monthly_signals

def analyze_monthly_signals(data):
    """Analyze signals by month"""
    print("\n📅 Monthly Signal Analysis...")
    
    # Add month column
    data['month'] = data.index.to_period('M')
    
    # Group by month
    monthly_stats = []
    
    for month, group in data.groupby('month'):
        green_entries = group['green_entry'].sum() if 'green_entry' in group.columns else 0
        red_entries = group['red_entry'].sum() if 'red_entry' in group.columns else 0
        exits = group['exit_signal'].sum() if 'exit_signal' in group.columns else 0
        total_entries = green_entries + red_entries
        
        monthly_stats.append({
            'month': month,
            'green_entries': green_entries,
            'red_entries': red_entries,
            'total_entries': total_entries,
            'exits': exits,
            'trading_days': len(group)
        })
    
    # Convert to DataFrame
    monthly_df = pd.DataFrame(monthly_stats)
    
    # Calculate statistics
    avg_signals_per_month = monthly_df['total_entries'].mean()
    max_signals_per_month = monthly_df['total_entries'].max()
    min_signals_per_month = monthly_df['total_entries'].min()
    
    print(f"Average signals per month: {avg_signals_per_month:.2f}")
    print(f"Max signals per month: {max_signals_per_month}")
    print(f"Min signals per month: {min_signals_per_month}")
    
    # Show months with high signal counts
    high_signal_months = monthly_df[monthly_df['total_entries'] > 5]
    if len(high_signal_months) > 0:
        print(f"\n⚠️  Months with >5 signals:")
        for _, row in high_signal_months.iterrows():
            print(f"   {row['month']}: {row['total_entries']} signals")
    
    # Show recent months
    recent_months = monthly_df.tail(12)
    print(f"\n📊 Recent 12 months:")
    for _, row in recent_months.iterrows():
        print(f"   {row['month']}: {row['total_entries']} signals")
    
    return monthly_df

def compare_with_pine_script_expectations():
    """Compare results with expected Pine Script behavior"""
    print("\n🔄 Comparing with Pine Script Expectations...")
    
    # Expected behavior for daily timeframe
    expected_signals_per_month = 0.5  # 5-20 signals per year = ~0.5 per month
    expected_max_signals_per_month = 3  # Rare spikes
    
    print(f"Expected signals per month: {expected_signals_per_month}")
    print(f"Expected max signals per month: {expected_max_signals_per_month}")
    
    # Get actual results
    data, monthly_signals = test_daily_strategy_signals()
    
    if monthly_signals is not None:
        actual_avg = monthly_signals['total_entries'].mean()
        actual_max = monthly_signals['total_entries'].max()
        
        print(f"\nActual results:")
        print(f"Average signals per month: {actual_avg:.2f}")
        print(f"Max signals per month: {actual_max}")
        
        # Calculate discrepancy
        discrepancy = actual_avg / expected_signals_per_month
        print(f"\nDiscrepancy factor: {discrepancy:.1f}x more signals than expected")
        
        if discrepancy > 10:
            print("🚨 CRITICAL: Strategy generating way too many signals!")
            print("This explains the poor backtest performance.")
        elif discrepancy > 3:
            print("⚠️  WARNING: Strategy generating more signals than expected")
        else:
            print("✅ Signal count looks reasonable")
    
    return data, monthly_signals

def identify_strategy_issues():
    """Identify specific issues in the strategy logic"""
    print("\n🔍 Identifying Strategy Issues...")
    
    data, monthly_signals = test_daily_strategy_signals()
    if data is None:
        return
    
    issues = []
    
    # Issue 1: Check if signals are clustered
    if 'green_entry' in data.columns:
        signal_dates = data[data['green_entry'] | data['red_entry']].index
        if len(signal_dates) > 0:
            # Check for signal clustering
            signal_intervals = []
            for i in range(1, len(signal_dates)):
                interval = (signal_dates[i] - signal_dates[i-1]).days
                signal_intervals.append(interval)
            
            if signal_intervals:
                avg_interval = np.mean(signal_intervals)
                min_interval = min(signal_intervals)
                
                print(f"Average days between signals: {avg_interval:.1f}")
                print(f"Minimum days between signals: {min_interval}")
                
                if min_interval < 7:
                    issues.append(f"Signals too close together (min {min_interval} days)")
                if avg_interval < 14:
                    issues.append(f"Signals too frequent (avg {avg_interval:.1f} days)")
    
    # Issue 2: Check filter stability
    if 'filt' in data.columns:
        filt_changes = data['filt'].diff().abs()
        large_changes = filt_changes[filt_changes > filt_changes.quantile(0.95)]
        
        print(f"Filter stability - 95th percentile change: {large_changes.iloc[0]:.2f}")
        
        if len(large_changes) > len(data) * 0.1:  # More than 10% large changes
            issues.append("Filter too unstable")
    
    # Issue 3: Check band width consistency
    if 'hband' in data.columns and 'lband' in data.columns:
        band_width = data['hband'] - data['lband']
        band_width_ratio = band_width / data['Close']
        
        print(f"Band width ratio - Mean: {band_width_ratio.mean():.4f}")
        print(f"Band width ratio - Std: {band_width_ratio.std():.4f}")
        
        if band_width_ratio.std() > band_width_ratio.mean():
            issues.append("Band width too volatile")
    
    if issues:
        print("\n🚨 IDENTIFIED ISSUES:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
    else:
        print("\n✅ No obvious strategy issues identified")
    
    return issues

if __name__ == "__main__":
    print("🔍 DAILY GAUSSIAN CHANNEL STRATEGY ANALYSIS")
    print("=" * 60)
    
    # Run analysis
    compare_with_pine_script_expectations()
    issues = identify_strategy_issues()
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    
    if issues:
        print(f"\nFound {len(issues)} strategy issues.")
        print("Recommendations:")
        print("1. Review signal generation logic")
        print("2. Check filter parameter sensitivity")
        print("3. Validate entry/exit conditions")
    else:
        print("\nStrategy logic appears correct.")
        print("Check if data quality or parameter tuning is needed.") 