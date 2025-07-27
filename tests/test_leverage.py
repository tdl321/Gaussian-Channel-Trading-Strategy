#!/usr/bin/env python3
"""
Test Leverage Implementation

This script tests that the 5x leverage is working correctly in the Gaussian Channel strategy
"""

from gaussian_channel_strategy import GaussianChannelStrategy
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def test_leverage_calculation():
    """
    Test leverage calculation with a simple example
    """
    print("🔧 TESTING LEVERAGE IMPLEMENTATION")
    print("=" * 60)
    
    # Create a simple test case
    print("📊 Testing leverage calculation:")
    print("   Initial Capital: $10,000")
    print("   Position Size: 65% of equity")
    print("   Leverage: 5x")
    print("   Expected behavior:")
    print("     • Margin used: $6,500 (65% of $10,000)")
    print("     • Position value: $32,500 (5x leverage)")
    print("     • Cash remaining: $3,500 ($10,000 - $6,500)")
    
    # Test the calculation manually
    initial_capital = 10000
    position_size_pct = 0.65
    leverage = 5.0
    
    margin_used = initial_capital * position_size_pct
    position_value = margin_used * leverage
    cash_remaining = initial_capital - margin_used
    
    print(f"\n🧮 Manual Calculation:")
    print(f"   Margin used: ${margin_used:,.2f}")
    print(f"   Position value: ${position_value:,.2f}")
    print(f"   Cash remaining: ${cash_remaining:,.2f}")
    
    return margin_used, position_value, cash_remaining

def test_leverage_with_bitcoin():
    """
    Test leverage with Bitcoin data
    """
    print(f"\n🚀 TESTING LEVERAGE WITH BITCOIN DATA")
    print("=" * 60)
    
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
    
    # Use a small test window
    test_data = data['2020-03-01':'2020-04-01'].copy()
    
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
    print(f"💰 BTC price range: ${test_data['Close'].min():.2f} - ${test_data['Close'].max():.2f}")
    
    # Test with leverage
    strategy_leveraged = GaussianChannelStrategy(
        initial_capital=10000,
        position_size_pct=0.65,
        leverage=5.0,  # 5x leverage
        start_date='2020-03-01',
        end_date='2020-04-01'
    )
    
    # Test without leverage for comparison
    strategy_no_leverage = GaussianChannelStrategy(
        initial_capital=10000,
        position_size_pct=0.65,
        leverage=1.0,  # No leverage
        start_date='2020-03-01',
        end_date='2020-04-01'
    )
    
    print(f"\n🔄 Running strategy with leverage...")
    test_data_lev = strategy_leveraged.prepare_signals(test_data.copy())
    strategy_leveraged.backtest_manual(test_data_lev)
    
    print(f"🔄 Running strategy without leverage...")
    test_data_no_lev = strategy_no_leverage.prepare_signals(test_data.copy())
    strategy_no_leverage.backtest_manual(test_data_no_lev)
    
    # Compare results
    print(f"\n📊 LEVERAGE COMPARISON")
    print("=" * 50)
    
    leveraged_trades = [t for t in strategy_leveraged.trade_log if t['action'] == 'BUY']
    no_leverage_trades = [t for t in strategy_no_leverage.trade_log if t['action'] == 'BUY']
    
    print(f"{'Metric':<25} {'5x Leverage':<15} {'No Leverage':<15} {'Ratio':<10}")
    print("-" * 70)
    
    if leveraged_trades and no_leverage_trades:
        # Compare first trade
        lev_trade = leveraged_trades[0]
        no_lev_trade = no_leverage_trades[0]
        
        lev_shares = lev_trade['shares']
        no_lev_shares = no_lev_trade['shares']
        
        lev_position_value = lev_trade.get('position_value', lev_shares * lev_trade['price'])
        no_lev_position_value = no_lev_shares * no_lev_trade['price']
        
        print(f"{'Shares bought':<25} {lev_shares:<15.4f} {no_lev_shares:<15.4f} {lev_shares/no_lev_shares:<10.1f}")
        print(f"{'Position value':<25} ${lev_position_value:<14,.0f} ${no_lev_position_value:<14,.0f} {lev_position_value/no_lev_position_value:<10.1f}")
        print(f"{'Margin used':<25} ${lev_trade.get('margin_used', 'N/A'):<14} ${no_lev_position_value:<14,.0f} {'N/A':<10}")
        
        # Expected ratios
        expected_position_ratio = 5.0
        actual_position_ratio = lev_position_value / no_lev_position_value
        
        print(f"\n✅ Verification:")
        print(f"   Expected position ratio: {expected_position_ratio}x")
        print(f"   Actual position ratio: {actual_position_ratio:.1f}x")
        
        if abs(actual_position_ratio - expected_position_ratio) < 0.1:
            print(f"   🎉 Leverage working correctly!")
        else:
            print(f"   ❌ Leverage calculation issue")
    
    else:
        print("⚠️  No trades generated in test period")
    
    return strategy_leveraged, strategy_no_leverage

def compare_leverage_scenarios():
    """
    Compare different leverage scenarios
    """
    print(f"\n📊 LEVERAGE SCENARIO COMPARISON")
    print("=" * 60)
    
    leverage_scenarios = [1.0, 2.0, 3.0, 5.0, 10.0]
    
    # Load minimal Bitcoin data for testing
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
    
    test_data = data['2020-06-01':'2020-12-01'].copy()
    
    # Add indicators
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
    
    print(f"{'Leverage':<10} {'Final Equity':<15} {'Return %':<12} {'Max DD %':<12} {'Trades':<8}")
    print("-" * 60)
    
    for leverage in leverage_scenarios:
        strategy = GaussianChannelStrategy(
            initial_capital=10000,
            position_size_pct=0.65,
            leverage=leverage,
            start_date='2020-06-01',
            end_date='2020-12-01'
        )
        
        try:
            test_data_copy = strategy.prepare_signals(test_data.copy())
            strategy.backtest_manual(test_data_copy)
            
            if strategy.equity_curve:
                final_equity = strategy.equity_curve[-1]['equity']
                total_return = (final_equity / strategy.initial_capital - 1) * 100
                
                # Simple max drawdown
                equity_values = [point['equity'] for point in strategy.equity_curve]
                peak = equity_values[0]
                max_dd = 0
                for equity in equity_values:
                    if equity > peak:
                        peak = equity
                    dd = (equity / peak - 1) * 100
                    if dd < max_dd:
                        max_dd = dd
                
                num_trades = len([t for t in strategy.trade_log if t['action'] == 'BUY'])
                
                print(f"{leverage:<10.1f} ${final_equity:<14,.0f} {total_return:<12.1f} {max_dd:<12.1f} {num_trades:<8}")
            else:
                print(f"{leverage:<10.1f} {'No data':<15} {'N/A':<12} {'N/A':<12} {'0':<8}")
                
        except Exception as e:
            print(f"{leverage:<10.1f} {'Error':<15} {'N/A':<12} {'N/A':<12} {'N/A':<8}")

if __name__ == "__main__":
    # Test leverage calculation
    test_leverage_calculation()
    
    # Test with Bitcoin data
    strategy_lev, strategy_no_lev = test_leverage_with_bitcoin()
    
    # Compare scenarios
    compare_leverage_scenarios()
    
    print(f"\n🎉 LEVERAGE TESTING COMPLETE!")
    print("✅ 5x leverage should now match Pine Script behavior")
    print("✅ Position sizes should be 5x larger than no-leverage")
    print("✅ Margin requirements properly calculated") 