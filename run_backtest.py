#!/usr/bin/env python3
"""
Gaussian Channel Strategy - Backtest Runner
Run comprehensive backtests on different cryptocurrency data

Supports both BTC and ETH data files with different CSV formats
"""

import sys
import os
import pandas as pd
from datetime import datetime
import argparse

# Import backtesting components
from gaussian_bot_hype.strategy.backtest import (
    run_backtrader_backtest,
    GaussianChannelStrategy,
    analyze_backtrader_results
)


def load_data_file(data_path, symbol):
    """
    Load and prepare data file based on symbol type
    
    Args:
        data_path: Path to the CSV file
        symbol: Symbol name ('BTC' or 'ETH')
        
    Returns:
        pd.DataFrame: Prepared data with standard column names
    """
    print(f"📊 Loading {symbol} data from: {data_path}")
    
    try:
        # Load CSV data
        data = pd.read_csv(data_path)
        print(f"✅ Loaded {len(data)} rows of data")
        
        if symbol == 'BTC':
            # BTC data format: Open time,Open,High,Low,Close,Volume,Close time,...
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
            
        elif symbol == 'ETH':
            # ETH data format: Date,Open,High,Low,Close,Adj Close,Volume
            data['Date'] = pd.to_datetime(data['Date'])
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
        return data
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise


def get_strategy_params(symbol):
    """
    Get strategy parameters based on symbol
    
    Args:
        symbol: Symbol name ('BTC' or 'ETH')
        
    Returns:
        dict: Strategy parameters
    """
    if symbol == 'BTC':
        # BTC parameters (original, stable)
        return {
            'poles': 6,           # Original parameter
            'period': 144,        # Original parameter  
            'multiplier': 1.414,  # Original parameter
            'position_size_pct': 1.0,  # 100% position size
            'atr_period': 14      # Standard ATR
        }
    elif symbol == 'ETH':
        # ETH parameters (can be optimized for altcoin-like behavior)
        return {
            'poles': 6,           # Same as BTC for now
            'period': 144,        # Same as BTC for now
            'multiplier': 1.414,  # Same as BTC for now
            'position_size_pct': 1.0,  # 100% position size
            'atr_period': 14      # Standard ATR
        }
    else:
        raise ValueError(f"Unsupported symbol: {symbol}")


def run_backtest(symbol='BTC', data_path=None):
    """
    Run backtest on specified symbol data
    
    Args:
        symbol: Symbol to test ('BTC' or 'ETH')
        data_path: Optional custom data path
        
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"🚀 Starting {symbol} Backtest with Current Parameters")
    print("=" * 60)
    
    # Set default data path if not provided
    if data_path is None:
        if symbol == 'BTC':
            data_path = "gaussian_bot_hype/data/btc_1d_data_2018_to_2025.csv"
        elif symbol == 'ETH':
            data_path = "gaussian_bot_hype/data/ETH-USD (2017-2024).csv"
        else:
            raise ValueError(f"Unsupported symbol: {symbol}")
    
    try:
        # Load and prepare data
        data = load_data_file(data_path, symbol)
        
        # Get strategy parameters
        strategy_params = get_strategy_params(symbol)
        
        print(f"\n📊 Strategy Parameters for {symbol}:")
        for key, value in strategy_params.items():
            print(f"   {key}: {value}")
        
        # Run backtest
        print(f"\n🔄 Running {symbol} backtest...")
        cerebro = run_backtrader_backtest(
            data=data,
            strategy_class=GaussianChannelStrategy,
            strategy_params=strategy_params,
            initial_cash=10000,  # $10,000 starting capital
            commission=0.001,    # 0.1% commission
            slippage_perc=0.01   # 1% slippage
        )
        
        # Analyze results
        print(f"\n📈 Analyzing {symbol} results...")
        results = analyze_backtrader_results(cerebro)
        
        # Print results
        print(f"\n" + "=" * 60)
        print(f"📊 {symbol} BACKTEST RESULTS")
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
        
        print(f"\n✅ {symbol} backtest completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ {symbol} backtest failed with error: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False


def run_btc_backtest():
    """Run backtest on BTC data (legacy function for compatibility)"""
    return run_backtest('BTC')


def run_eth_backtest():
    """Run backtest on ETH data"""
    return run_backtest('ETH')


def main():
    """Main function with command line argument support"""
    parser = argparse.ArgumentParser(description='Run Gaussian Channel Strategy Backtest')
    parser.add_argument('--symbol', '-s', choices=['BTC', 'ETH'], default='BTC',
                       help='Symbol to backtest (default: BTC)')
    parser.add_argument('--data-path', '-d', 
                       help='Custom path to data file (optional)')
    
    args = parser.parse_args()
    
    print(f"🎯 Gaussian Channel Strategy Backtest")
    print(f"📈 Symbol: {args.symbol}")
    if args.data_path:
        print(f"📁 Custom data path: {args.data_path}")
    print("=" * 60)
    
    success = run_backtest(args.symbol, args.data_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 