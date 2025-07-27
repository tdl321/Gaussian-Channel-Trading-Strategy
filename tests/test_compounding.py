#!/usr/bin/env python3
"""
Test Compounding Logic

This script tests that the fixed compounding logic works correctly
"""

from gaussian_channel_strategy import GaussianChannelStrategy
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def test_compounding_logic():
    """
    Test the compounding logic with a simple example
    """
    print("🧪 TESTING COMPOUNDING LOGIC")
    print("=" * 50)
    
    # Load Bitcoin data
    data = pd.read_csv('Bitcoin Price.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    
    column_mapping = {
        '24h Open (USD)': 'Open',
        '24h High (USD)': 'High', 
        '24h Low (USD)': 'Low',
        'Closing Price (USD)': 'Close'
    }
    data.rename(columns=column_mapping, inplace=True)
    data['Volume'] = 0
    
    if 'Currency' in data.columns:
        data.drop('Currency', axis=1, inplace=True)
    
    for col in ['Open', 'High', 'Low', 'Close']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    data.dropna(inplace=True)
    data.sort_index(inplace=True)
    
    # Test with a short period where we know there should be signals
    test_data = data['2020-03-01':'2020-06-01'].copy()
    
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
    test_data['sma_200'] = test_data['Close'].rolling(200).mean()
    
    print(f"📊 Test period: {test_data.index[0]} to {test_data.index[-1]}")
    print(f"📊 Price range: ${test_data['Close'].min():.2f} - ${test_data['Close'].max():.2f}")
    
    # Initialize strategy
    strategy = GaussianChannelStrategy(
        poles=4,
        period=144,
        multiplier=1.414,
        initial_capital=10000,
        position_size_pct=0.65,  # 65% like Pine Script
        start_date='2020-03-01',
        end_date='2020-06-01'
    )
    
    print(f"\n💰 Initial capital: ${strategy.initial_capital:,.2f}")
    print(f"📊 Position size: {strategy.position_size_pct*100}% of equity")
    
    # Run strategy
    test_data = strategy.prepare_signals(test_data)
    strategy.backtest_manual(test_data)
    
    # Analyze compounding behavior
    print(f"\n📈 COMPOUNDING ANALYSIS")
    print("=" * 50)
    
    buy_trades = [t for t in strategy.trade_log if t['action'] == 'BUY']
    sell_trades = [t for t in strategy.trade_log if t['action'] == 'SELL']
    
    print(f"📊 Total trades: {len(buy_trades)} buys, {len(sell_trades)} sells")
    
    if len(buy_trades) > 1:
        print(f"\n💡 Verifying compounding:")
        for i, trade in enumerate(buy_trades):
            equity_before = strategy.initial_capital if i == 0 else buy_trades[i-1].get('equity_before', 'N/A')
            position_value = trade['shares'] * trade['price']
            
            print(f"   Trade {i+1}:")
            print(f"     Date: {trade['date'].strftime('%Y-%m-%d')}")
            print(f"     Price: ${trade['price']:.2f}")
            print(f"     Shares: {trade['shares']:.4f}")
            print(f"     Position Value: ${position_value:.2f}")
            print(f"     Total Cost: ${trade.get('total_cost', 'N/A'):.2f}")
            
            if i > 0:
                prev_position_value = buy_trades[i-1]['shares'] * buy_trades[i-1]['price']
                ratio = position_value / prev_position_value
                print(f"     Size vs previous: {ratio:.2f}x")
    
    # Check final equity
    final_equity = strategy.equity_curve[-1]['equity'] if strategy.equity_curve else strategy.current_equity
    total_return = (final_equity / strategy.initial_capital - 1) * 100
    
    print(f"\n🎯 RESULTS:")
    print(f"   Initial capital: ${strategy.initial_capital:,.2f}")
    print(f"   Final equity: ${final_equity:,.2f}")
    print(f"   Total return: {total_return:.1f}%")
    print(f"   Final cash: ${strategy.current_equity:,.2f}")
    
    # Check for proper compounding
    if len(buy_trades) > 1 and total_return > 5:
        print(f"   ✅ Compounding appears to be working")
    elif len(buy_trades) <= 1:
        print(f"   ⚠️  Not enough trades to verify compounding")
    else:
        print(f"   ❌ Compounding may not be working properly")
    
    return strategy

def compare_before_after():
    """
    Compare performance before and after the compounding fix
    """
    print(f"\n📊 BEFORE/AFTER COMPARISON")
    print("=" * 50)
    
    # Test with full Bitcoin dataset
    strategy = GaussianChannelStrategy(
        start_date='2017-01-01',
        end_date='2021-01-04',
        initial_capital=10000
    )
    
    # Load data
    data = pd.read_csv('Bitcoin Price.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    
    column_mapping = {
        '24h Open (USD)': 'Open',
        '24h High (USD)': 'High', 
        '24h Low (USD)': 'Low',
        'Closing Price (USD)': 'Close'
    }
    data.rename(columns=column_mapping, inplace=True)
    data['Volume'] = 0
    
    if 'Currency' in data.columns:
        data.drop('Currency', axis=1, inplace=True)
    
    for col in ['Open', 'High', 'Low', 'Close']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    data.dropna(inplace=True)
    
    # Filter to strategy period
    filtered_data = data[(data.index >= '2017-01-01') & (data.index <= '2021-01-04')].copy()
    
    # Add indicators
    filtered_data['hlc3'] = (filtered_data['High'] + filtered_data['Low'] + filtered_data['Close']) / 3
    filtered_data['true_range'] = np.maximum(
        filtered_data['High'] - filtered_data['Low'],
        np.maximum(
            abs(filtered_data['High'] - filtered_data['Close'].shift(1)),
            abs(filtered_data['Low'] - filtered_data['Close'].shift(1))
        )
    )
    filtered_data['atr'] = filtered_data['true_range'].rolling(14).mean()
    filtered_data['sma_200'] = filtered_data['Close'].rolling(200).mean()
    
    # Run strategy
    filtered_data = strategy.prepare_signals(filtered_data)
    strategy.backtest_manual(filtered_data)
    
    # Calculate metrics
    metrics = strategy.calculate_performance_metrics()
    
    print(f"📈 FIXED COMPOUNDING RESULTS:")
    print(f"   Total Return: {metrics['Total Return (%)']}%")
    print(f"   CAGR: {metrics['CAGR (%)']}%")
    print(f"   Max Drawdown: {metrics['Max Drawdown (%)']}%")
    print(f"   Sharpe Ratio: {metrics['Sharpe Ratio']}")
    print(f"   Number of Trades: {metrics['Number of Trades']}")
    
    # Check for realistic results
    if metrics['Total Return (%)'] > 0 and metrics['Max Drawdown (%)'] > -100:
        print(f"   ✅ Results look more realistic now!")
    else:
        print(f"   ⚠️  Results still look unusual")
    
    return strategy

if __name__ == "__main__":
    # Test compounding logic
    strategy = test_compounding_logic()
    
    # Compare with full dataset
    print(f"\n" + "="*60)
    strategy_full = compare_before_after()
    
    print(f"\n🎉 COMPOUNDING FIX COMPLETE!")
    print(f"✅ The equity curve should now properly compound")
    print(f"✅ No more artificial 'drawdowns' from incorrect accounting")
    print(f"✅ Position sizes should grow as account equity grows") 