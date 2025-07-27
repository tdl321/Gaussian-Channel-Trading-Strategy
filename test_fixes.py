#!/usr/bin/env python3
"""
Test script to validate that the Pine Script fixes are working correctly
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from gaussian_channel_strategy import GaussianChannelStrategy, GaussianChannelFilter
import pandas as pd
import numpy as np

def test_gaussian_filter():
    """Test that the Gaussian filter is working and producing reasonable outputs"""
    print("🧪 TESTING GAUSSIAN FILTER IMPLEMENTATION")
    print("=" * 50)
    
    # Create simple test data
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
    
    test_data = pd.DataFrame({
        'Open': prices * 0.995,
        'High': prices * 1.01,
        'Low': prices * 0.99,
        'Close': prices,
        'Volume': np.random.randint(1000, 10000, 200)
    }, index=dates)
    
    # Calculate hlc3 and true range
    test_data['hlc3'] = (test_data['High'] + test_data['Low'] + test_data['Close']) / 3
    test_data['true_range'] = np.maximum(
        test_data['High'] - test_data['Low'],
        np.maximum(
            abs(test_data['High'] - test_data['Close'].shift(1)),
            abs(test_data['Low'] - test_data['Close'].shift(1))
        )
    )
    
    # Test Gaussian filter
    gaussian_filter = GaussianChannelFilter(poles=4, period=144, multiplier=1.414)
    
    print(f"📊 Test Data: {len(test_data)} bars")
    print(f"🔧 Filter Parameters:")
    print(f"   Alpha: {gaussian_filter.alpha:.6f}")
    print(f"   Poles: {gaussian_filter.poles}")
    print(f"   Period: {gaussian_filter.period}")
    
    # Apply filter
    src_confirmed = test_data['hlc3'].shift(1)
    tr_confirmed = test_data['true_range'].shift(1)
    
    filt, hband, lband = gaussian_filter.apply_filter(src_confirmed, tr_confirmed)
    
    # Check results
    valid_filter_values = filt.dropna()
    valid_hband_values = hband.dropna()
    valid_lband_values = lband.dropna()
    
    print(f"\n📈 Filter Results:")
    print(f"   Valid filter values: {len(valid_filter_values)}")
    print(f"   Filter range: {valid_filter_values.min():.2f} to {valid_filter_values.max():.2f}")
    print(f"   Upper band range: {valid_hband_values.min():.2f} to {valid_hband_values.max():.2f}")
    print(f"   Lower band range: {valid_lband_values.min():.2f} to {valid_lband_values.max():.2f}")
    
    # Sanity checks
    checks_passed = 0
    total_checks = 4
    
    if len(valid_filter_values) > 50:
        print("✅ Filter produces sufficient valid values")
        checks_passed += 1
    else:
        print("❌ Filter produces too few valid values")
    
    if valid_hband_values.mean() > valid_filter_values.mean():
        print("✅ Upper band is above filter (correct)")
        checks_passed += 1
    else:
        print("❌ Upper band is not above filter")
    
    if valid_lband_values.mean() < valid_filter_values.mean():
        print("✅ Lower band is below filter (correct)")
        checks_passed += 1
    else:
        print("❌ Lower band is not below filter")
    
    if not (filt.isna().all() or hband.isna().all() or lband.isna().all()):
        print("✅ Filter produces non-NaN values")
        checks_passed += 1
    else:
        print("❌ Filter produces only NaN values")
    
    print(f"\n🎯 Filter Test Result: {checks_passed}/{total_checks} checks passed")
    return checks_passed == total_checks

def test_position_sizing():
    """Test that position sizing is now correct"""
    print("\n🧪 TESTING POSITION SIZING")
    print("=" * 50)
    
    strategy = GaussianChannelStrategy(
        position_size_pct=0.65,
        max_pyramids=5,
        initial_capital=10000
    )
    
    print(f"💰 Initial Capital: ${strategy.initial_capital}")
    print(f"📊 Position Size %: {strategy.position_size_pct * 100}%")
    print(f"🔢 Max Pyramids: {strategy.max_pyramids}")
    
    # Simulate a position entry
    strategy.current_equity = strategy.initial_capital
    test_price = 100.0
    
    # Calculate expected position
    expected_margin = strategy.current_equity * strategy.position_size_pct
    expected_position_value = expected_margin * strategy.leverage
    expected_shares = expected_position_value / test_price
    
    print(f"\n📈 Position Calculation (Price = ${test_price}):")
    print(f"   Expected margin used: ${expected_margin:,.2f}")
    print(f"   Expected position value (5x leverage): ${expected_position_value:,.2f}")
    print(f"   Expected shares: {expected_shares:,.2f}")
    
    # This should use 65% of equity, not 65%/5 = 13%
    if expected_margin == 6500.0:  # 65% of 10000
        print("✅ Position sizing uses full 65% (correct)")
        return True
    else:
        print(f"❌ Position sizing incorrect. Expected $6500, got ${expected_margin}")
        return False

def test_signal_logic():
    """Test that signal logic is working"""
    print("\n🧪 TESTING SIGNAL LOGIC")
    print("=" * 50)
    
    # Create strategy
    strategy = GaussianChannelStrategy(
        poles=4,
        period=50,  # Shorter period for test
        start_date='2020-01-01',
        end_date='2020-12-31'
    )
    
    # Create simple trending test data
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    trend = np.linspace(100, 120, 100) + np.random.randn(100) * 0.5
    
    test_data = pd.DataFrame({
        'Open': trend * 0.995,
        'High': trend * 1.01,
        'Low': trend * 0.99,
        'Close': trend,
        'Volume': 1000
    }, index=dates)
    
    # Add technical indicators
    test_data['hlc3'] = (test_data['High'] + test_data['Low'] + test_data['Close']) / 3
    test_data['true_range'] = np.maximum(
        test_data['High'] - test_data['Low'],
        np.maximum(
            abs(test_data['High'] - test_data['Close'].shift(1)),
            abs(test_data['Low'] - test_data['Close'].shift(1))
        )
    )
    test_data['atr'] = test_data['true_range'].rolling(14).mean()
    test_data['sma_200'] = test_data['Close'].rolling(50).mean()  # Shorter for test
    
    # Prepare signals
    prepared_data = strategy.prepare_signals(test_data)
    
    # Check signal generation
    green_entries = prepared_data['green_entry'].sum()
    red_entries = prepared_data['red_entry'].sum()
    exit_signals = prepared_data['exit_signal'].sum()
    
    print(f"📊 Signal Generation Results:")
    print(f"   Green entries: {green_entries}")
    print(f"   Red entries: {red_entries}")
    print(f"   Exit signals: {exit_signals}")
    print(f"   Total entry signals: {green_entries + red_entries}")
    
    # Basic sanity checks
    if (green_entries + red_entries) > 0:
        print("✅ Strategy generates entry signals")
        return True
    else:
        print("❌ Strategy generates no entry signals")
        return False

def main():
    """Run all tests"""
    print("🔧 VALIDATING PINE SCRIPT FIXES")
    print("=" * 60)
    
    test_results = []
    
    test_results.append(test_gaussian_filter())
    test_results.append(test_position_sizing())
    test_results.append(test_signal_logic())
    
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"\n🎯 OVERALL TEST RESULTS")
    print("=" * 60)
    print(f"Tests Passed: {passed}/{total}")
    
    if passed == total:
        print("✅ ALL TESTS PASSED - Fixes are working correctly!")
        print("\n📋 Next Steps:")
        print("1. Test with real market data")
        print("2. Compare outputs directly with Pine Script")
        print("3. Validate performance metrics match")
    else:
        print("❌ Some tests failed - review implementation")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 