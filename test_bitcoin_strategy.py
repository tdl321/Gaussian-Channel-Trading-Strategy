#!/usr/bin/env python3
"""
Test the Gaussian Channel Strategy with Advanced Backtester on Bitcoin Data

This script demonstrates the full capabilities of the new architecture:
- Advanced slippage and margin management
- Realistic execution with next-bar timing
- Comprehensive risk management
- Professional performance analytics
"""

import pandas as pd
import numpy as np
from src.gaussian_channel_strategy import GaussianChannelStrategy, AdvancedBacktester
import matplotlib.pyplot as plt

def load_bitcoin_data(csv_path, start_date=None, end_date=None):
    """
    Custom loader for Bitcoin CSV data from CryptoDataDownload format
    
    Expected format:
    Line 1: Website header
    Line 2: Column headers (unix,date,symbol,open,high,low,close,Volume BTC,Volume USD)
    Line 3+: Data rows
    """
    try:
        # Read CSV skipping the first header line
        data = pd.read_csv(csv_path, skiprows=1)
        
        print(f"📊 Bitcoin CSV columns: {list(data.columns)}")
        
        # Convert date column to datetime and set as index
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)
        
        # Standardize column names to match strategy expectations
        column_mapping = {
            'open': 'Open',
            'high': 'High', 
            'low': 'Low',
            'close': 'Close',
            'Volume BTC': 'Volume'
        }
        
        data.rename(columns=column_mapping, inplace=True)
        
        # Convert to numeric (handle any formatting issues)
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Filter by date range
        if start_date:
            data = data[data.index >= pd.to_datetime(start_date)]
        if end_date:
            data = data[data.index <= pd.to_datetime(end_date)]
        
        # Sort by date (data might be in reverse chronological order)
        data.sort_index(inplace=True)
        
        if data.empty:
            raise ValueError(f"No data found in date range {start_date} to {end_date}")
        
        print(f"✅ Loaded {len(data)} rows of Bitcoin data")
        print(f"📅 Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
        print(f"💰 Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
        
        return data
        
    except Exception as e:
        print(f"❌ Error loading Bitcoin CSV data: {e}")
        return None

def test_bitcoin_strategy():
    """Test the strategy on Bitcoin data with various configurations"""
    print("=" * 70)
    print("BITCOIN STRATEGY TEST WITH ADVANCED BACKTESTER")
    print("=" * 70)
    print()
    
    # Initialize strategy with Bitcoin-optimized parameters
    strategy = GaussianChannelStrategy(
        poles=4,
        period=144,
        multiplier=1.414,
        mode_lag=False,
        mode_fast=False,
        atr_spacing=0.5,  # Slightly higher for Bitcoin volatility
        sma_length=200,
        enable_sma_filter=False,
        max_pyramids=3,  # Conservative pyramiding for crypto
        position_size_pct=0.5,  # 50% of equity per trade (conservative for crypto)
        start_date='2020-01-01',
        end_date='2024-01-01'
    )
    
    print("🔧 Strategy Configuration:")
    print(f"   Asset: Bitcoin (BTC/USD)")
    print(f"   Position Size: {strategy.position_size_pct*100}% of equity")
    print(f"   Max Pyramids: {strategy.max_pyramids}")
    print(f"   ATR Spacing: {strategy.atr_spacing}")
    print(f"   Period: {strategy.start_date} to {strategy.end_date}")
    print()
    
    # Load Bitcoin data with custom loader
    data = load_bitcoin_data('data/Gemini_BTCUSD_d.csv', strategy.start_date, strategy.end_date)
    
    if data is None:
        print("❌ Failed to load Bitcoin data")
        return []
    
    # Prepare signals using the strategy's method
    from src.gaussian_channel_strategy import calculate_rma
    
    # Calculate additional indicators needed for strategy
    data['hlc3'] = (data['High'] + data['Low'] + data['Close']) / 3
    data['true_range'] = np.maximum(
        data['High'] - data['Low'],
        np.maximum(
            abs(data['High'] - data['Close'].shift(1)),
            abs(data['Low'] - data['Close'].shift(1))
        )
    )
    
    # Use RMA (Recursive Moving Average) for ATR - EXACT PineScript match
    data['atr'] = calculate_rma(data['true_range'], 14)
    data[f'sma_{strategy.sma_length}'] = data['Close'].rolling(strategy.sma_length).mean()
    
    # Prepare signals
    data = strategy.prepare_signals(data)
    
    # Test with different backtester configurations
    test_configs = [
        {
            'name': 'Conservative Setup',
            'initial_capital': 50000,
            'commission_pct': 0.0025,  # 0.25% (crypto exchange fees)
            'slippage_ticks': 2,
            'margin_requirement': 0.5,  # 50% margin requirement
            'max_leverage': 2.0,  # Conservative leverage
            'verbose': True
        },
        {
            'name': 'Aggressive Setup',
            'initial_capital': 25000,
            'commission_pct': 0.001,   # 0.1% (lower fees)
            'slippage_ticks': 3,
            'margin_requirement': 0.25,  # 25% margin requirement
            'max_leverage': 4.0,  # Higher leverage
            'verbose': False
        }
    ]
    
    results_summary = []
    
    for config in test_configs:
        print(f"🧪 Testing: {config['name']}")
        print("-" * 50)
        
        try:
            # Run backtest with custom data
            backtester = strategy.run_backtest(
                data,
                **{k: v for k, v in config.items() if k not in ['name']}
            )
            
            # Calculate performance metrics
            metrics = strategy.calculate_performance_metrics(backtester)
            
            # Store results for comparison
            results_summary.append({
                'config': config['name'],
                'total_return': metrics['Total Return (%)'],
                'cagr': metrics['CAGR (%)'],
                'max_drawdown': metrics['Max Drawdown (%)'],
                'sharpe_ratio': metrics['Sharpe Ratio'],
                'num_trades': metrics['Number of Trades'],
                'margin_calls': metrics['Margin Calls Triggered'],
                'forced_liquidations': metrics['Forced Liquidations'],
                'slippage_cost': metrics['Total Slippage Cost'],
                'final_equity': metrics['Final Equity'],
                'backtester': backtester,
                'data': data  # Store data for plotting
            })
            
            print(f"✅ {config['name']} Results:")
            print(f"   Total Return: {metrics['Total Return (%)']}%")
            print(f"   CAGR: {metrics['CAGR (%)']}%")
            print(f"   Max Drawdown: {metrics['Max Drawdown (%)']}%")
            print(f"   Sharpe Ratio: {metrics['Sharpe Ratio']}")
            print(f"   Number of Trades: {metrics['Number of Trades']}")
            print(f"   Margin Calls: {metrics['Margin Calls Triggered']}")
            print(f"   Forced Liquidations: {metrics['Forced Liquidations']}")
            print(f"   Final Equity: ${metrics['Final Equity']:,.2f}")
            print()
                
        except Exception as e:
            print(f"❌ Error in {config['name']}: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    return results_summary

def compare_configurations(results_summary):
    """Compare different backtester configurations"""
    if len(results_summary) < 2:
        print("⚠️  Need at least 2 configurations to compare")
        return
    
    print("📊 CONFIGURATION COMPARISON")
    print("=" * 70)
    print()
    
    # Create comparison table
    print(f"{'Metric':<20} {'Conservative':<15} {'Aggressive':<15} {'Difference':<15}")
    print("-" * 65)
    
    conservative = results_summary[0]
    aggressive = results_summary[1]
    
    comparisons = [
        ('Total Return (%)', 'total_return'),
        ('CAGR (%)', 'cagr'),
        ('Max Drawdown (%)', 'max_drawdown'),
        ('Sharpe Ratio', 'sharpe_ratio'),
        ('Number of Trades', 'num_trades'),
        ('Margin Calls', 'margin_calls'),
        ('Final Equity ($)', 'final_equity')
    ]
    
    for metric_name, metric_key in comparisons:
        cons_val = conservative[metric_key]
        agg_val = aggressive[metric_key]
        
        if isinstance(cons_val, (int, float)) and isinstance(agg_val, (int, float)):
            diff = agg_val - cons_val
            diff_str = f"{diff:+.2f}" if metric_key != 'num_trades' else f"{diff:+.0f}"
        else:
            diff_str = "N/A"
        
        print(f"{metric_name:<20} {cons_val:<15.2f} {agg_val:<15.2f} {diff_str:<15}")
    
    print()

def analyze_risk_management(results_summary):
    """Analyze risk management effectiveness"""
    print("⚠️  RISK MANAGEMENT ANALYSIS")
    print("=" * 70)
    print()
    
    for result in results_summary:
        backtester = result['backtester']
        config = result['config']
        
        print(f"📋 {config} Risk Analysis:")
        print(f"   Initial Capital: ${backtester.initial_capital:,}")
        print(f"   Final Cash: ${backtester.cash:,.2f}")
        print(f"   Final Position: {backtester.position_size:.2f} shares")
        print(f"   Margin Used: ${backtester.margin_used:,.2f}")
        
        # Analyze margin call events
        if backtester.margin_call_log:
            margin_calls = [e for e in backtester.margin_call_log if e['event'] == 'MARGIN_CALL_TRIGGERED']
            liquidations = [e for e in backtester.margin_call_log if e['event'] == 'FORCED_LIQUIDATION']
            
            print(f"   Margin Events:")
            print(f"     Margin Calls: {len(margin_calls)}")
            print(f"     Forced Liquidations: {len(liquidations)}")
            
            if margin_calls:
                worst_margin = min(e.get('margin_level_pct', 100) for e in margin_calls)
                print(f"     Worst Margin Level: {worst_margin:.1f}%")
        else:
            print(f"   Margin Events: None (Good risk management)")
        
        # Analyze trade performance
        if backtester.trade_log:
            buy_trades = [t for t in backtester.trade_log if t['action'] == 'BUY']
            sell_trades = [t for t in backtester.trade_log if t['action'] == 'SELL']
            
            total_slippage = sum(abs(t.get('slippage', 0)) for t in backtester.trade_log)
            total_commissions = sum(t.get('commission', 0) for t in backtester.trade_log)
            
            print(f"   Trading Costs:")
            print(f"     Total Slippage: ${total_slippage:.2f}")
            print(f"     Total Commissions: ${total_commissions:.2f}")
            print(f"     Total Costs: ${total_slippage + total_commissions:.2f}")
        
        print()

def create_detailed_plot(results_summary):
    """Create detailed comparison plots"""
    if not results_summary:
        print("⚠️  No results to plot")
        return
    
    print("📈 Generating detailed comparison plots...")
    
    # Use the best performing configuration for detailed plotting
    best_result = max(results_summary, key=lambda x: x['total_return'])
    
    try:
        # Create strategy instance for plotting
        strategy = GaussianChannelStrategy(
            poles=4,
            period=144,
            multiplier=1.414,
            atr_spacing=0.5,
            max_pyramids=3,
            position_size_pct=0.5,
            start_date='2020-01-01',
            end_date='2024-01-01'
        )
        
        print(f"Creating detailed plot for: {best_result['config']}")
        
        # Plot results using the stored data and backtester
        strategy.plot_results(
            best_result['data'], 
            best_result['backtester'], 
            save_path='results/bitcoin_advanced_backtester_results.png'
        )
        
        print("✅ Detailed plot saved to: results/bitcoin_advanced_backtester_results.png")
        
    except Exception as e:
        print(f"❌ Error creating plot: {e}")

def main():
    """Run comprehensive Bitcoin strategy test"""
    print("🚀 Starting Bitcoin Strategy Test with Advanced Backtester...")
    print()
    
    # Run strategy tests
    results_summary = test_bitcoin_strategy()
    
    if results_summary:
        # Compare configurations
        compare_configurations(results_summary)
        
        # Analyze risk management
        analyze_risk_management(results_summary)
        
        # Create detailed plots
        create_detailed_plot(results_summary)
        
        print("=" * 70)
        print("🎯 BITCOIN STRATEGY TEST SUMMARY")
        print("=" * 70)
        print()
        print("✅ Advanced Backtester Features Demonstrated:")
        print("   📊 Realistic slippage modeling for crypto markets")
        print("   ⚠️  Comprehensive margin call monitoring")
        print("   🔄 Next-bar execution (no look-ahead bias)")
        print("   💰 Proper commission accounting for crypto exchanges")
        print("   🚨 Risk management with forced liquidations")
        print()
        print("✅ Strategy Performance Analysis:")
        print("   📈 Multiple configuration testing")
        print("   📊 Risk-adjusted return metrics")
        print("   ⚖️  Conservative vs Aggressive comparison")
        print("   🎯 Professional risk management validation")
        print()
        print("The advanced backtester provides institutional-quality")
        print("simulation for cryptocurrency trading strategies.")
        
    else:
        print("❌ No successful test results to analyze")

if __name__ == "__main__":
    main() 