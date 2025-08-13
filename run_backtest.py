#!/usr/bin/env python3
"""
Run backtest on BTC data with current strategy parameters
"""

import pandas as pd
import sys
import os

# Add strategy directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'gaussian_bot_hype'))

from strategy.backtest import (
    GaussianChannelStrategy, 
    create_backtrader_datafeed, 
    run_backtrader_backtest, 
    analyze_backtrader_results
)


def run_btc_backtest():
    """Run backtest on BTC data with current parameters"""
    
    print("🚀 Starting BTC Backtest with Current Parameters")
    print("=" * 60)
    
    # Load BTC data
    data_path = "gaussian_bot_hype/data/btc_1d_data_2018_to_2025.csv"
    print(f"Loading data from: {data_path}")
    
    try:
        # Load CSV data
        data = pd.read_csv(data_path)
        print(f"✅ Loaded {len(data)} rows of data")
        
        # Convert date column
        data['Date'] = pd.to_datetime(data['Open time'])
        data.set_index('Date', inplace=True)
        
        # Rename columns to match expected format
        data = data.rename(columns={
            'Open': 'Open',
            'High': 'High', 
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume'
        })
        
        # Select only required columns
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        print(f"✅ Data prepared: {len(data)} bars from {data.index[0]} to {data.index[-1]}")
        
        # Strategy parameters (matching live trading)
        strategy_params = {
            'poles': 6,           # Current live trading parameter
            'period': 144,        # Current live trading parameter  
            'multiplier': 1.414,  # Current live trading parameter
            'position_size_pct': 1.0,  # 100% position size
            'atr_period': 14      # ATR for channel calculation
        }
        
        print("\n📊 Strategy Parameters:")
        for key, value in strategy_params.items():
            print(f"   {key}: {value}")
        
        # Run backtest
        print("\n🔄 Running backtest...")
        cerebro = run_backtrader_backtest(
            data=data,
            strategy_class=GaussianChannelStrategy,
            strategy_params=strategy_params,
            initial_cash=10000,  # $10,000 starting capital
            commission=0.001,    # 0.1% commission
            slippage_perc=0.01   # 1% slippage
        )
        
        # Analyze results
        print("\n📈 Analyzing results...")
        results = analyze_backtrader_results(cerebro)
        
        # Print results
        print("\n" + "=" * 60)
        print("📊 BACKTEST RESULTS")
        print("=" * 60)
        print(f"Initial Capital: ${results['initial_cash']:,.2f}")
        print(f"Final Value: ${results['final_value']:,.2f}")
        print(f"Total Return: {results['total_return_pct']:.2f}%")
        print(f"Total Return: {results['total_return']:.4f}")
        
        # Get strategy instance for additional metrics
        strategy = results.get('strategy')
        if strategy is not None:
            try:
                print(f"\nTrading Statistics:")
                print(f"   Entry Count: {getattr(strategy, 'entry_count', 'N/A')}")
                print(f"   Last Entry Price: ${getattr(strategy, 'last_entry_price', 'N/A')}")
            except Exception as e:
                print(f"   Warning: Could not access strategy statistics: {e}")
        else:
            print(f"\nTrading Statistics: Strategy instance not available")
        
        print("\n✅ Backtest completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Backtest failed with error: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_btc_backtest()
    sys.exit(0 if success else 1) 