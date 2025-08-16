#!/usr/bin/env python3
"""
Test for Signal Generation with Gaussian Channel Strategy
"""

import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path to import strategy modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy.gaussian_filter import GaussianChannelFilter
from strategy.signals import SignalGenerator


def test_signal_generation():
    """Test that signal generation works with live trading parameters"""
    
    # Create test data with clear trend for signal testing
    dates = pd.date_range('2024-01-01', periods=300, freq='D')
    np.random.seed(42)  # For reproducible results
    
    # Generate price data with a clear uptrend then downtrend
    base_price = 45000
    prices = []
    
    # First 100 days: uptrend
    for i in range(100):
        if i == 0:
            price = base_price
        else:
            change = np.random.normal(0.005, 0.02)  # 0.5% daily uptrend
            price = prices[-1] * (1 + change)
        prices.append(price)
    
    # Next 100 days: downtrend
    for i in range(100):
        change = np.random.normal(-0.003, 0.02)  # 0.3% daily downtrend
        price = prices[-1] * (1 + change)
        prices.append(price)
    
    # Last 100 days: sideways with volatility
    for i in range(100):
        change = np.random.normal(0, 0.03)  # High volatility, no trend
        price = prices[-1] * (1 + change)
        prices.append(price)
    
    # Create OHLCV data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Realistic OHLC from close price
        volatility = abs(np.random.normal(0, 0.015))
        high = close * (1 + volatility)
        low = close * (1 - volatility)
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
    
    # Initialize components with live trading parameters
    gaussian_filter = GaussianChannelFilter(
        poles=6,        # Original parameter
        period=144,     # Original parameter
        multiplier=1.414  # Original parameter
    )
    
    signal_generator = SignalGenerator(
        gaussian_filter=gaussian_filter,
        config_params={
            'POLES': 6,
            'PERIOD': 144,
            'MULTIPLIER': 1.414
        }
    )
    
    # Generate signals
    df_with_signals = signal_generator.prepare_signals(df)
    
    # Check that signals were generated
    assert 'entry_signal' in df_with_signals.columns, "Entry signals should be generated"
    assert 'exit_signal' in df_with_signals.columns, "Exit signals should be generated"
    assert 'hband_current' in df_with_signals.columns, "Current upper band should be calculated"
    assert 'filt_current' in df_with_signals.columns, "Current filter should be calculated"
    
    # Check that we have valid signals (after warm-up period)
    valid_data = df_with_signals.dropna()
    assert len(valid_data) > 0, "Should have valid signal data after warm-up"
    
    # Count signals
    entry_signals = valid_data['entry_signal'].sum()
    exit_signals = valid_data['exit_signal'].sum()
    
    print("✅ Signal generation test passed!")
    print(f"   - Generated {len(df)} bars of test data")
    print(f"   - Valid signal data: {len(valid_data)} bars")
    print(f"   - Entry signals: {entry_signals}")
    print(f"   - Exit signals: {exit_signals}")
    print(f"   - Signal rate: {entry_signals/len(valid_data)*100:.1f}%")
    
    # Test live signal generation
    test_live_signals(valid_data, signal_generator)
    
    return True


def test_live_signals(df_with_signals, signal_generator):
    """Test live signal generation functionality"""
    
    print("\n🔍 Testing live signal generation...")
    
    # Reset signal generator state
    signal_generator.reset_state()
    
    # Simulate live trading by processing data bar by bar
    signals_generated = []
    
    for i in range(len(df_with_signals)):
        # Get current bar data
        current_data = df_with_signals.iloc[:i+1]
        
        if len(current_data) < 2:
            continue
        
        # Generate live signals
        signals = signal_generator.generate_live_signals(current_data)
        
        if signals:
            for signal in signals:
                signals_generated.append({
                    'date': current_data.index[-1],
                    'action': signal['action'],
                    'price': signal['price'],
                    'reason': signal['reason']
                })
    
    print(f"   - Live signals generated: {len(signals_generated)}")
    
    if signals_generated:
        print("   - Sample signals:")
        for i, signal in enumerate(signals_generated[:5]):  # Show first 5 signals
            print(f"     {i+1}. {signal['date'].strftime('%Y-%m-%d')}: {signal['action']} @ ${signal['price']:.2f}")
    
    # Test position state tracking
    position_state = signal_generator.get_position_state()
    print(f"   - Final position state: {position_state}")
    
    # Verify signal logic
    test_signal_logic(df_with_signals)


def test_signal_logic(df_with_signals):
    """Test that signal logic is correct"""
    
    print("\n🔍 Testing signal logic...")
    
    # Get a sample of data with signals
    sample_data = df_with_signals.dropna().tail(50)
    
    if len(sample_data) == 0:
        print("   - No sample data available for logic testing")
        return
    
    # Test entry logic: price should be above upper band when entry signal is True
    entry_samples = sample_data[sample_data['entry_signal'] == True]
    if len(entry_samples) > 0:
        entry_logic_correct = all(
            entry_samples['Close'] > entry_samples['hband_current']
        )
        print(f"   - Entry logic correct: {entry_logic_correct}")
    
    # Test exit logic: price should be below upper band when exit signal is True
    exit_samples = sample_data[sample_data['exit_signal'] == True]
    if len(exit_samples) > 0:
        exit_logic_correct = all(
            exit_samples['Close'] < exit_samples['hband_current']
        )
        print(f"   - Exit logic correct: {exit_logic_correct}")
    
    # Test band relationships
    band_logic_correct = all(
        sample_data['hband_current'] > sample_data['lband_current']
    )
    print(f"   - Band relationships correct: {band_logic_correct}")


if __name__ == "__main__":
    test_signal_generation() 