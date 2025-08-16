#!/usr/bin/env python3
"""
Parameter Comparison: Optimized vs Original Gaussian Channel Parameters
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest import run_backtrader_backtest, load_asset_data
from strategy.gaussian_filter import GaussianChannelFilter
from strategy.signals import SignalGenerator


def run_parameter_backtest(data, params, param_name, initial_cash=10000):
    """
    Run backtest with specific parameters
    
    Args:
        data: DataFrame with OHLCV data
        params: Dictionary with strategy parameters
        param_name: Name for this parameter set
        initial_cash: Initial capital
        
    Returns:
        dict: Backtest results
    """
    print(f"\n🔄 Running {param_name} backtest...")
    print(f"   Parameters: {params}")
    
    try:
        # Import backtrader strategy class
        from backtest import GaussianChannelStrategy
        
        # Run backtest
        cerebro = run_backtrader_backtest(
            data=data,
            strategy_class=GaussianChannelStrategy,
            strategy_params=params,
            initial_cash=initial_cash,
            commission=0.001,
            slippage_perc=0.01,
            debug_mode=False,
            use_config=False
        )
        
        # Get results
        final_value = cerebro.broker.getvalue()
        total_return = (final_value - initial_cash) / initial_cash * 100
        
        # Get strategy instance for detailed stats (avoid the error)
        strategy = None
        entry_count = 0
        exit_count = 0
        position_stats = {}
        
        try:
            if hasattr(cerebro, 'runstrats') and cerebro.runstrats:
                strategy_list = cerebro.runstrats[0]
                if isinstance(strategy_list, list) and len(strategy_list) > 0:
                    strategy = strategy_list[0]
                    entry_count = getattr(strategy, 'entry_count', 0)
                    exit_count = getattr(strategy, 'exit_count', 0)
                    
                    # Get position manager statistics
                    if hasattr(strategy, 'position_manager'):
                        position_stats = strategy.position_manager.get_trade_statistics()
        except Exception as e:
            print(f"   Warning: Could not access strategy statistics: {e}")
        
        results = {
            'param_name': param_name,
            'parameters': params,
            'initial_cash': initial_cash,
            'final_value': final_value,
            'total_return_pct': total_return,
            'entry_count': entry_count,
            'exit_count': exit_count,
            'position_stats': position_stats
        }
        
        print(f"   ✅ {param_name} completed: {total_return:.2f}% return")
        return results
        
    except Exception as e:
        print(f"   ❌ {param_name} failed: {e}")
        return None


def calculate_optimization_score(results):
    """
    Calculate optimization score based on multiple metrics
    
    Args:
        results: Backtest results dictionary
        
    Returns:
        float: Optimization score (higher is better)
    """
    if not results or not results.get('position_stats'):
        return 0.0
    
    stats = results['position_stats']
    
    # Base metrics
    total_return = results['total_return_pct']
    total_trades = stats.get('total_trades', 0)
    win_rate = stats.get('win_rate', 0)
    avg_pnl = stats.get('avg_pnl', 0)
    
    # Risk metrics
    max_drawdown = 0  # Would need to calculate from trade history
    avg_duration = stats.get('avg_duration', 0)
    
    # Scoring weights
    return_weight = 0.4
    trade_count_weight = 0.2
    win_rate_weight = 0.2
    avg_pnl_weight = 0.1
    duration_weight = 0.1
    
    # Normalize metrics (simple normalization)
    normalized_return = min(total_return / 1000, 1.0)  # Cap at 1000%
    normalized_trades = min(total_trades / 50, 1.0)    # Cap at 50 trades
    normalized_win_rate = win_rate / 100               # 0-1 scale
    normalized_avg_pnl = min(avg_pnl / 100000, 1.0)   # Cap at $100k avg
    normalized_duration = max(0, 1 - (avg_duration / 100))  # Shorter is better
    
    # Calculate weighted score
    score = (
        normalized_return * return_weight +
        normalized_trades * trade_count_weight +
        normalized_win_rate * win_rate_weight +
        normalized_avg_pnl * avg_pnl_weight +
        normalized_duration * duration_weight
    )
    
    return score


def compare_parameters():
    """Compare optimized vs original parameters"""
    
    print("🔬 GAUSSIAN CHANNEL PARAMETER COMPARISON")
    print("=" * 60)
    
    # Load BTC data
    try:
        data = load_asset_data("../data/btc_1d_data_2018_to_2025.csv", "BTC")
        print(f"✅ Loaded {len(data)} bars of BTC data")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Define parameter sets
    optimized_params = {
        'poles': 5,
        'period': 135,
        'multiplier': 2.859,
        'leverage': 5,
        'position_size_pct': 1.0,
        'atr_period': 13,
        'stop_loss_pct': 0.05,
        'enable_stop_loss': False
    }
    
    original_params = {
        'poles': 6,
        'period': 144,
        'multiplier': 1.414,
        'leverage': 5,
        'position_size_pct': 1.0,
        'atr_period': 14,
        'stop_loss_pct': 0.05,
        'enable_stop_loss': False
    }
    
    # Run backtests
    results = {}
    
    # Test optimized parameters
    results['optimized'] = run_parameter_backtest(
        data, optimized_params, "Optimized Parameters (5/135/2.859)"
    )
    
    # Test original parameters
    results['original'] = run_parameter_backtest(
        data, original_params, "Original Parameters (6/144/1.414)"
    )
    
    # Calculate optimization scores
    for key, result in results.items():
        if result:
            result['optimization_score'] = calculate_optimization_score(result)
    
    # Print comparison
    print("\n" + "=" * 80)
    print("📊 PARAMETER COMPARISON RESULTS")
    print("=" * 80)
    
    for key, result in results.items():
        if result:
            print(f"\n🔹 {result['param_name'].upper()}:")
            print(f"   Initial Capital: ${result['initial_cash']:,.2f}")
            print(f"   Final Value: ${result['final_value']:,.2f}")
            print(f"   Total Return: {result['total_return_pct']:+.2f}%")
            print(f"   Entry Count: {result['entry_count']}")
            print(f"   Exit Count: {result['exit_count']}")
            
            if result['position_stats']:
                stats = result['position_stats']
                print(f"   Total Trades: {stats.get('total_trades', 'N/A')}")
                print(f"   Win Rate: {stats.get('win_rate', 'N/A'):.1f}%")
                print(f"   Total PnL: ${stats.get('total_pnl', 'N/A'):+,.2f}")
                print(f"   Average PnL: ${stats.get('avg_pnl', 'N/A'):+,.2f}")
                print(f"   Average Duration: {stats.get('avg_duration', 'N/A'):.1f} days")
            
            print(f"   Optimization Score: {result['optimization_score']:.4f}")
    
    # Determine winner
    if results['optimized'] and results['original']:
        opt_score = results['optimized']['optimization_score']
        orig_score = results['original']['optimization_score']
        
        print(f"\n🏆 WINNER ANALYSIS:")
        print(f"   Optimized Score: {opt_score:.4f}")
        print(f"   Original Score: {orig_score:.4f}")
        
        if opt_score > orig_score:
            winner = "OPTIMIZED"
            improvement = ((opt_score - orig_score) / orig_score) * 100
            print(f"   🎉 WINNER: {winner} parameters")
            print(f"   📈 Improvement: {improvement:+.1f}%")
        else:
            winner = "ORIGINAL"
            improvement = ((orig_score - opt_score) / opt_score) * 100
            print(f"   🎉 WINNER: {winner} parameters")
            print(f"   📈 Improvement: {improvement:+.1f}%")
    
    # Detailed analysis
    print(f"\n🔍 DETAILED ANALYSIS:")
    
    if results['optimized'] and results['original']:
        opt = results['optimized']
        orig = results['original']
        
        # Return comparison
        return_diff = opt['total_return_pct'] - orig['total_return_pct']
        print(f"   📊 Return Difference: {return_diff:+.2f}%")
        
        # Trade count comparison
        trade_diff = opt['entry_count'] - orig['entry_count']
        print(f"   📊 Trade Count Difference: {trade_diff:+d} trades")
        
        # Win rate comparison
        if opt['position_stats'] and orig['position_stats']:
            win_rate_diff = opt['position_stats']['win_rate'] - orig['position_stats']['win_rate']
            print(f"   📊 Win Rate Difference: {win_rate_diff:+.1f}%")
            
            # Average PnL comparison
            avg_pnl_diff = opt['position_stats']['avg_pnl'] - orig['position_stats']['avg_pnl']
            print(f"   📊 Average PnL Difference: ${avg_pnl_diff:+,.2f}")
    
    print("\n" + "=" * 80)


def analyze_parameter_characteristics():
    """Analyze the characteristics of each parameter set"""
    
    print("\n🔬 PARAMETER CHARACTERISTICS ANALYSIS")
    print("=" * 60)
    
    print("\n📊 OPTIMIZED PARAMETERS (5/135/2.859):")
    print("   • Poles: 5 (vs 6) - Less smoothing, faster response")
    print("   • Period: 135 (vs 144) - Shorter lookback, more responsive")
    print("   • Multiplier: 2.859 (vs 1.414) - Wider channels, fewer false signals")
    print("   • ATR Period: 13 (vs 14) - Faster volatility adaptation")
    
    print("\n📊 ORIGINAL PARAMETERS (6/144/1.414):")
    print("   • Poles: 6 - More smoothing, slower response")
    print("   • Period: 144 - Longer lookback, more stable")
    print("   • Multiplier: 1.414 - Tighter channels, more signals")
    print("   • ATR Period: 14 - Standard volatility calculation")
    
    print("\n🎯 EXPECTED DIFFERENCES:")
    print("   • Optimized: More aggressive, higher potential returns, more trades")
    print("   • Original: More conservative, lower drawdown, fewer trades")
    print("   • Optimized: Better for trending markets")
    print("   • Original: Better for choppy/sideways markets")


def main():
    """Run parameter comparison"""
    
    print("🚀 Starting Parameter Comparison Analysis")
    print("=" * 60)
    
    # Analyze parameter characteristics
    analyze_parameter_characteristics()
    
    # Run comparison
    compare_parameters()
    
    print("\n✅ Parameter comparison completed!")


if __name__ == "__main__":
    main()
