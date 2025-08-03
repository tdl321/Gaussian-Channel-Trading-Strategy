#!/usr/bin/env python3
"""
Test script to compare Pine Script logic with Python implementation
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
    """Create a test configuration without API keys"""
    config = Mock()
    
    # Strategy parameters (matching Pine Script defaults)
    config.POLES = 6
    config.PERIOD = 144
    config.MULTIPLIER = 1.414
    config.ATR_SPACING = 0.4
    config.MAX_PYRAMIDS = 5
    config.POSITION_SIZE_PCT = 1.0
    
    # Mock API keys to avoid validation errors
    config.HYPERLIQUID_API_KEY = "test_key"
    config.HYPERLIQUID_SECRET_KEY = "test_secret"
    
    return config

def load_and_prepare_data():
    """Load and prepare data for testing"""
    print("📊 Loading historical data...")
    
    # Load data from CSV
    csv_path = os.path.join('data', 'historical_data.csv')
    if not os.path.exists(csv_path):
        print(f"❌ Data file not found: {csv_path}")
        return None
    
    # Load data
    data = pd.read_csv(csv_path)
    
    # Convert timestamp to datetime
    data['date'] = pd.to_datetime(data['unix'], unit='ms')
    data.set_index('date', inplace=True)
    
    # Rename columns to match expected format
    data = data.rename(columns={
        'open': 'Open',
        'high': 'High', 
        'low': 'Low',
        'close': 'Close',
        'Volume BTC': 'Volume'
    })
    
    # Sort by date (oldest first)
    data = data.sort_index()
    
    print(f"✅ Loaded {len(data)} bars from {data.index[0]} to {data.index[-1]}")
    return data

def test_gaussian_filter_consistency():
    """Test if Gaussian filter produces consistent results"""
    print("\n🔍 Testing Gaussian Filter Consistency...")
    
    # Load data
    data = load_and_prepare_data()
    if data is None:
        return
    
    # Initialize filter with Pine Script parameters
    config = create_test_config()
    gaussian_filter = GaussianChannelFilter(
        poles=config.POLES,
        period=config.PERIOD,
        multiplier=config.MULTIPLIER
    )
    
    # Calculate HLC3 and True Range (matching Pine Script)
    data['hlc3'] = (data['High'] + data['Low'] + data['Close']) / 3
    
    # True Range calculation (matching Pine Script ta.tr(true))
    data['tr1'] = data['High'] - data['Low']
    data['tr2'] = abs(data['High'] - data['Close'].shift(1))
    data['tr3'] = abs(data['Low'] - data['Close'].shift(1))
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Apply Gaussian filter
    filt, hband, lband = gaussian_filter.apply_filter(data['hlc3'], data['true_range'])
    
    # Add to dataframe
    data['filt'] = filt
    data['hband'] = hband
    data['lband'] = lband
    
    # Check for NaN values
    nan_count = data[['filt', 'hband', 'lband']].isna().sum()
    print(f"NaN values in filter outputs: {nan_count.to_dict()}")
    
    # Check filter stability
    filt_changes = data['filt'].diff().abs()
    print(f"Filter stability - Max change: {filt_changes.max():.6f}")
    print(f"Filter stability - Mean change: {filt_changes.mean():.6f}")
    
    # Check band relationships
    band_width = data['hband'] - data['lband']
    print(f"Band width - Min: {band_width.min():.2f}, Max: {band_width.max():.2f}")
    
    return data

def test_signal_generation():
    """Test signal generation logic"""
    print("\n🎯 Testing Signal Generation...")
    
    # Get data with filters
    data = test_gaussian_filter_consistency()
    if data is None:
        return
    
    # Initialize signal generator
    config = create_test_config()
    gaussian_filter = GaussianChannelFilter(
        poles=config.POLES,
        period=config.PERIOD,
        multiplier=config.MULTIPLIER
    )
    signal_generator = SignalGenerator(gaussian_filter, config)
    
    # Prepare signals
    data_with_signals = signal_generator.prepare_signals(data)
    
    # Analyze signals
    signal_summary = signal_generator.get_signal_summary(data_with_signals)
    print(f"Signal Summary: {signal_summary}")
    
    # Check signal distribution
    if 'green_entry' in data_with_signals.columns:
        green_entries = data_with_signals['green_entry'].sum()
        red_entries = data_with_signals['red_entry'].sum()
        exits = data_with_signals['exit_signal'].sum()
        
        print(f"Green entries: {green_entries}")
        print(f"Red entries: {red_entries}")
        print(f"Exits: {exits}")
        print(f"Total bars: {len(data_with_signals)}")
        print(f"Entry rate: {(green_entries + red_entries) / len(data_with_signals):.4f}")
    
    return data_with_signals

def test_pine_script_logic_comparison():
    """Compare Python implementation with Pine Script logic"""
    print("\n🔄 Comparing with Pine Script Logic...")
    
    # Get data with signals
    data = test_signal_generation()
    if data is None:
        return
    
    # Pine Script uses previous bar data for plotting but current bar for entries
    # Let's check if our implementation matches this logic
    
    # Pine Script logic:
    # 1. For plotting: uses previous bar's hlc3 and true range
    # 2. For entries: uses current bar's hlc3 and true range (0-bar delay)
    
    # Check if we have sufficient data for analysis
    recent_data = data.tail(100)  # Last 100 bars
    
    print("\nRecent Signal Analysis (Last 100 bars):")
    if 'green_entry' in recent_data.columns:
        recent_green = recent_data['green_entry'].sum()
        recent_red = recent_data['red_entry'].sum()
        recent_exits = recent_data['exit_signal'].sum()
        
        print(f"Recent green entries: {recent_green}")
        print(f"Recent red entries: {recent_red}")
        print(f"Recent exits: {recent_exits}")
        
        # Check for signal clustering
        if recent_green > 0 or recent_red > 0:
            print("⚠️  WARNING: Signals detected in recent data - check if this matches Pine Script")
        else:
            print("✅ No recent signals - this might indicate a discrepancy with Pine Script")
    
    # Check filter values
    recent_filt = recent_data['filt'].dropna()
    if len(recent_filt) > 0:
        print(f"\nRecent filter values:")
        print(f"Min: {recent_filt.min():.2f}")
        print(f"Max: {recent_filt.max():.2f}")
        print(f"Current: {recent_filt.iloc[-1]:.2f}")
        
        # Check if filter is trending
        filt_trend = recent_filt.iloc[-1] > recent_filt.iloc[-2] if len(recent_filt) >= 2 else False
        print(f"Filter trending up: {filt_trend}")
    
    return data

def identify_potential_issues():
    """Identify potential issues in the implementation"""
    print("\n🚨 Identifying Potential Issues...")
    
    issues = []
    
    # Issue 1: Data format mismatch
    print("1. Checking data format...")
    csv_path = os.path.join('data', 'historical_data.csv')
    if os.path.exists(csv_path):
        sample_data = pd.read_csv(csv_path, nrows=5)
        print(f"   CSV columns: {list(sample_data.columns)}")
        if 'unix' in sample_data.columns:
            print("   ✅ Unix timestamp format detected")
        else:
            issues.append("Data format mismatch - expected unix timestamps")
    
    # Issue 2: Filter initialization
    print("2. Checking filter initialization...")
    config = create_test_config()
    print(f"   Poles: {config.POLES}")
    print(f"   Period: {config.PERIOD}")
    print(f"   Multiplier: {config.MULTIPLIER}")
    
    # Issue 3: Signal logic
    print("3. Checking signal logic...")
    print("   Pine Script uses current bar data for entries (0-bar delay)")
    print("   Python implementation should match this exactly")
    
    # Issue 4: Data sufficiency
    print("4. Checking data sufficiency...")
    data = load_and_prepare_data()
    if data is not None:
        min_required = config.PERIOD + 25
        print(f"   Required bars: {min_required}")
        print(f"   Available bars: {len(data)}")
        if len(data) < min_required:
            issues.append(f"Insufficient data: {len(data)} < {min_required}")
        else:
            print("   ✅ Sufficient data available")
    
    # Issue 5: Parameter consistency
    print("5. Checking parameter consistency...")
    pine_params = {
        'poles': 6,
        'period': 144,
        'multiplier': 1.414
    }
    python_params = {
        'poles': config.POLES,
        'period': config.PERIOD,
        'multiplier': config.MULTIPLIER
    }
    
    for param, pine_val in pine_params.items():
        python_val = python_params[param]
        if pine_val != python_val:
            issues.append(f"Parameter mismatch: {param} = {python_val} (Pine: {pine_val})")
        else:
            print(f"   ✅ {param}: {python_val}")
    
    if issues:
        print("\n🚨 IDENTIFIED ISSUES:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
    else:
        print("\n✅ No obvious issues identified")
    
    return issues

if __name__ == "__main__":
    print("🔍 GAUSSIAN CHANNEL STRATEGY ANALYSIS")
    print("=" * 50)
    
    # Run all tests
    test_gaussian_filter_consistency()
    test_signal_generation()
    test_pine_script_logic_comparison()
    issues = identify_potential_issues()
    
    print("\n" + "=" * 50)
    print("ANALYSIS COMPLETE")
    print("=" * 50)
    
    if issues:
        print(f"\nFound {len(issues)} potential issues that could explain performance discrepancy.")
        print("Recommendations:")
        print("1. Verify parameter consistency with Pine Script")
        print("2. Check data format and preprocessing")
        print("3. Validate signal generation logic")
        print("4. Test with exact Pine Script parameters")
    else:
        print("\nNo obvious issues found. Consider:")
        print("1. Running backtest with exact same data as Pine Script")
        print("2. Comparing individual bar calculations")
        print("3. Checking for timing differences in signal generation") 