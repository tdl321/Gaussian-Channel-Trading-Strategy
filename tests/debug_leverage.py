#!/usr/bin/env python3
"""
Debug Leverage Equity Calculation
"""

from gaussian_channel_strategy import GaussianChannelStrategy
import pandas as pd
import numpy as np

def debug_equity_calculation():
    # Load minimal Bitcoin data
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

    # Use a very short test period
    test_data = data['2020-03-01':'2020-04-15'].copy()
    
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

    print('🔍 DEBUGGING LEVERAGE EQUITY CALCULATION')
    print('=' * 60)
    print(f'Test period: {test_data.index[0]} to {test_data.index[-1]}')
    print(f'Price range: ${test_data["Close"].min():.2f} - ${test_data["Close"].max():.2f}')

    # Test with small leverage first
    strategy = GaussianChannelStrategy(
        initial_capital=10000,
        position_size_pct=0.65,
        leverage=2.0,  # Start with 2x to see the issue
        start_date='2020-03-01',
        end_date='2020-04-15'
    )

    print(f'\nInitial settings:')
    print(f'  Capital: ${strategy.initial_capital:,.2f}')
    print(f'  Position size: {strategy.position_size_pct*100}%')
    print(f'  Leverage: {strategy.leverage}x')

    # Manual equity tracking
    print(f'\n📊 STEP-BY-STEP EQUITY TRACKING:')
    
    test_data = strategy.prepare_signals(test_data)
    
    # Find first few signals manually
    entry_signals = test_data[test_data['green_entry'] | test_data['red_entry']]
    
    if len(entry_signals) > 0:
        print(f'\nFound {len(entry_signals)} entry signals')
        
        # Simulate first entry manually
        first_signal = entry_signals.iloc[0]
        price = first_signal['Close']
        
        print(f'\n1️⃣ FIRST ENTRY SIMULATION:')
        print(f'   Date: {entry_signals.index[0]}')
        print(f'   Price: ${price:.2f}')
        
        # Manual calculation
        margin_used = strategy.initial_capital * strategy.position_size_pct
        position_value = margin_used * strategy.leverage
        shares = position_value / price
        commission = position_value * strategy.commission
        total_cost = margin_used + commission
        
        print(f'   Margin used: ${margin_used:,.2f}')
        print(f'   Position value: ${position_value:,.2f}')
        print(f'   Shares: {shares:.4f}')
        print(f'   Commission: ${commission:.2f}')
        print(f'   Total cost: ${total_cost:.2f}')
        print(f'   Cash after entry: ${strategy.initial_capital - total_cost:,.2f}')
        
        # Test different price scenarios
        test_prices = [price * 0.95, price, price * 1.05, price * 1.10, price * 1.20]
        
        print(f'\n📈 EQUITY AT DIFFERENT PRICES:')
        print(f'{"Price":<10} {"Position Value":<15} {"Unrealized P&L":<15} {"Total Equity":<15}')
        print('-' * 60)
        
        for test_price in test_prices:
            current_position_value = shares * test_price
            unrealized_pnl = current_position_value - margin_used
            cash_remaining = strategy.initial_capital - total_cost
            total_equity = cash_remaining + unrealized_pnl
            
            print(f'${test_price:<9.2f} ${current_position_value:<14,.0f} ${unrealized_pnl:<14,.0f} ${total_equity:<14,.0f}')
        
        print(f'\n🧮 EQUITY CALCULATION BREAKDOWN:')
        print(f'   Formula: Total Equity = Cash + Unrealized P&L')
        print(f'   Where: Unrealized P&L = (Current Position Value) - (Original Margin Used)')
        print(f'   Cash = Initial Capital - Margin Used - Commission')
        
    else:
        print('❌ No entry signals found in test period')

    # Now run actual strategy for comparison
    print(f'\n🤖 ACTUAL STRATEGY RESULTS:')
    strategy.backtest_manual(test_data)
    
    if strategy.equity_curve:
        print(f'   Final equity: ${strategy.equity_curve[-1]["equity"]:,.2f}')
        print(f'   Number of trades: {len([t for t in strategy.trade_log if t["action"] == "BUY"])}')
        
        # Show first few equity points
        print(f'\n📊 First few equity points:')
        for i, point in enumerate(strategy.equity_curve[:5]):
            print(f'   {point["date"].strftime("%Y-%m-%d")}: ${point["equity"]:,.2f}')
    else:
        print('   No equity curve generated')

if __name__ == "__main__":
    debug_equity_calculation() 