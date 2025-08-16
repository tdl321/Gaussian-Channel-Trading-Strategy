#!/usr/bin/env python3
"""
Debug script to test backtest functionality and identify issues
"""

import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest import run_btc_backtest, load_asset_data, get_asset_strategy_params
from strategy.gaussian_filter import GaussianChannelFilter
from strategy.signals import SignalGenerator


def debug_signal_generation():
    """Debug signal generation to see if signals are being created"""
    
    print("🔍 Debugging Signal Generation")
    print("=" * 50)
    
    try:
        # Load BTC data
        data = load_asset_data("../data/btc_1d_data_2018_to_2025.csv", "BTC")
        print(f"✅ Loaded {len(data)} bars of BTC data")
        
        # Get strategy parameters
        params = get_asset_strategy_params("BTC")
        print(f"📊 Strategy Parameters: {params}")
        
        # Initialize Gaussian filter
        gaussian_filter = GaussianChannelFilter(
            poles=params['poles'],
            period=params['period'],
            multiplier=params['multiplier']
        )
        
        # Initialize signal generator
        config_params = {
            'POLES': params['poles'],
            'PERIOD': params['period'],
            'MULTIPLIER': params['multiplier']
        }
        signal_generator = SignalGenerator(gaussian_filter, config_params)
        
        # Prepare signals
        df_with_signals = signal_generator.prepare_signals(data)
        
        # Check for signals
        entry_signals = df_with_signals['entry_signal'].sum()
        exit_signals = df_with_signals['exit_signal'].sum()
        
        print(f"\n📈 Signal Analysis:")
        print(f"   Total bars: {len(df_with_signals)}")
        print(f"   Entry signals: {entry_signals}")
        print(f"   Exit signals: {exit_signals}")
        print(f"   Signal rate: {entry_signals/len(df_with_signals)*100:.2f}%")
        
        # Check if we have valid filter values
        valid_filt = df_with_signals['filt_current'].notna().sum()
        valid_hband = df_with_signals['hband_current'].notna().sum()
        
        print(f"\n🔧 Filter Analysis:")
        print(f"   Valid filter values: {valid_filt}")
        print(f"   Valid upper band values: {valid_hband}")
        print(f"   Filter completion rate: {valid_filt/len(df_with_signals)*100:.2f}%")
        
        # Show some sample data
        print(f"\n📊 Sample Data (last 5 bars):")
        sample_data = df_with_signals.tail(5)[['Close', 'filt_current', 'hband_current', 'entry_signal', 'exit_signal']]
        print(sample_data)
        
        return entry_signals > 0 and exit_signals > 0
        
    except Exception as e:
        print(f"❌ Error in signal generation: {e}")
        import traceback
        traceback.print_exc()
        return False


def debug_backtest_execution():
    """Debug backtest execution with minimal data"""
    
    print("\n🔍 Debugging Backtest Execution")
    print("=" * 50)
    
    try:
        # Run a minimal backtest with debug mode
        result = run_btc_backtest(
            initial_cash=10000,
            debug_mode=True,
            commission=0.001,
            slippage_perc=0.01
        )
        
        if result['success']:
            print("✅ Backtest completed successfully")
            
            # Check if trades were executed
            cerebro = result['cerebro']
            strategy = None
            
            # Try to get strategy instance
            try:
                if hasattr(cerebro, 'runstrats') and cerebro.runstrats:
                    strategy_list = cerebro.runstrats[0]
                    if isinstance(strategy_list, list) and len(strategy_list) > 0:
                        strategy = strategy_list[0]
            except:
                pass
            
            if strategy:
                print(f"\n📊 Strategy Statistics:")
                print(f"   Entry count: {getattr(strategy, 'entry_count', 'N/A')}")
                print(f"   Exit count: {getattr(strategy, 'exit_count', 'N/A')}")
                
                # Check position manager
                if hasattr(strategy, 'position_manager'):
                    position_stats = strategy.position_manager.get_trade_statistics()
                    print(f"   Total trades: {position_stats.get('total_trades', 0)}")
                    print(f"   Position manager stats: {position_stats}")
            
            return True
        else:
            print(f"❌ Backtest failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error in backtest execution: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run debug tests"""
    
    print("🚀 Starting Backtest Debug Session")
    print("=" * 60)
    
    # Test 1: Signal Generation
    signals_ok = debug_signal_generation()
    
    # Test 2: Backtest Execution
    backtest_ok = debug_backtest_execution()
    
    print("\n" + "=" * 60)
    print("📊 Debug Results Summary")
    print("=" * 60)
    print(f"Signal Generation: {'✅ PASSED' if signals_ok else '❌ FAILED'}")
    print(f"Backtest Execution: {'✅ PASSED' if backtest_ok else '❌ FAILED'}")
    
    if signals_ok and backtest_ok:
        print("\n🎉 All debug tests passed! The issue might be elsewhere.")
    else:
        print("\n⚠️  Some debug tests failed. Check the output above for issues.")


if __name__ == "__main__":
    main()
