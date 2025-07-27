#!/usr/bin/env python3
"""
Gaussian Channel Strategy Demo
Simple demonstration of the strategy with different configurations
"""

from gaussian_channel_strategy import GaussianChannelStrategy
import warnings
warnings.filterwarnings('ignore')

def run_basic_demo():
    """Run basic demo with default parameters"""
    print("=" * 60)
    print("GAUSSIAN CHANNEL STRATEGY - BASIC DEMO")
    print("=" * 60)
    
    # Default configuration
    strategy = GaussianChannelStrategy(
        start_date='2020-01-01',
        end_date='2024-01-01'
    )
    
    print("\n1. Running strategy on SPY with default parameters...")
    results = strategy.run_strategy('SPY', plot=False)
    
    if results:
        metrics = results['metrics']
        print(f"✅ Total Return: {metrics['Total Return (%)']}%")
        print(f"✅ CAGR: {metrics['CAGR (%)']}%")
        print(f"✅ Max Drawdown: {metrics['Max Drawdown (%)']}%")
        print(f"✅ Sharpe Ratio: {metrics['Sharpe Ratio']}")
        print(f"✅ Number of Trades: {metrics['Number of Trades']}")

def run_comparison_demo():
    """Compare different strategy configurations"""
    print("\n" + "=" * 60)
    print("PARAMETER COMPARISON DEMO")
    print("=" * 60)
    
    configs = [
        {
            'name': 'Default',
            'params': {
                'poles': 4,
                'period': 144,
                'multiplier': 1.414,
                'atr_spacing': 0.4
            }
        },
        {
            'name': 'Aggressive',
            'params': {
                'poles': 6,
                'period': 89,
                'multiplier': 1.0,
                'atr_spacing': 0.2,
                'mode_fast': True
            }
        },
        {
            'name': 'Conservative',
            'params': {
                'poles': 3,
                'period': 233,
                'multiplier': 2.0,
                'atr_spacing': 0.8,
                'enable_sma_filter': True,
                'max_pyramids': 3
            }
        }
    ]
    
    print(f"{'Config':<12} {'Return %':<10} {'CAGR %':<8} {'Drawdown %':<12} {'Sharpe':<8} {'Trades':<8}")
    print("-" * 65)
    
    for config in configs:
        strategy = GaussianChannelStrategy(
            start_date='2020-01-01',
            end_date='2024-01-01',
            **config['params']
        )
        
        results = strategy.run_strategy('SPY', plot=False)
        
        if results:
            m = results['metrics']
            print(f"{config['name']:<12} {m['Total Return (%)']:<10.1f} {m['CAGR (%)']:<8.1f} {m['Max Drawdown (%)']:<12.1f} {m['Sharpe Ratio']:<8.2f} {m['Number of Trades']:<8}")

def run_multi_symbol_demo():
    """Test strategy on multiple symbols"""
    print("\n" + "=" * 60)
    print("MULTI-SYMBOL DEMO")
    print("=" * 60)
    
    symbols = ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT']
    
    print(f"{'Symbol':<8} {'Return %':<10} {'CAGR %':<8} {'Drawdown %':<12} {'Sharpe':<8} {'Trades':<8}")
    print("-" * 60)
    
    for symbol in symbols:
        strategy = GaussianChannelStrategy(
            start_date='2020-01-01',
            end_date='2024-01-01'
        )
        
        try:
            results = strategy.run_strategy(symbol, plot=False)
            
            if results:
                m = results['metrics']
                print(f"{symbol:<8} {m['Total Return (%)']:<10.1f} {m['CAGR (%)']:<8.1f} {m['Max Drawdown (%)']:<12.1f} {m['Sharpe Ratio']:<8.2f} {m['Number of Trades']:<8}")
            else:
                print(f"{symbol:<8} {'Failed':<10}")
        except Exception as e:
            print(f"{symbol:<8} {'Error':<10} - {str(e)[:30]}")

def run_detailed_example():
    """Run detailed example with visualization"""
    print("\n" + "=" * 60)
    print("DETAILED EXAMPLE WITH VISUALIZATION")
    print("=" * 60)
    
    # Create strategy with specific parameters
    strategy = GaussianChannelStrategy(
        poles=4,
        period=144,
        multiplier=1.414,
        atr_spacing=0.4,
        enable_sma_filter=False,
        max_pyramids=5,
        initial_capital=10000,
        position_size_pct=0.65,
        start_date='2022-01-01',
        end_date='2024-01-01'
    )
    
    print("Running detailed backtest on AAPL...")
    print("This will display charts showing:")
    print("  1. Price action with Gaussian Channel")
    print("  2. Equity curve vs Buy & Hold")
    print("  3. Strategy drawdown")
    
    results = strategy.run_strategy('AAPL', plot=True, save_path='gaussian_strategy_results.png')
    
    if results:
        print("\n📊 Detailed Results:")
        for metric, value in results['metrics'].items():
            print(f"   {metric}: {value}")
        
        print(f"\n📈 Trade Summary:")
        buy_trades = [t for t in results['trade_log'] if t['action'] == 'BUY']
        sell_trades = [t for t in results['trade_log'] if t['action'] == 'SELL']
        
        print(f"   Total Buy Signals: {len(buy_trades)}")
        print(f"   Total Sell Signals: {len(sell_trades)}")
        
        if buy_trades:
            avg_entry = sum(t['price'] for t in buy_trades) / len(buy_trades)
            print(f"   Average Entry Price: ${avg_entry:.2f}")
        
        if sell_trades:
            avg_exit = sum(t['price'] for t in sell_trades) / len(sell_trades)
            print(f"   Average Exit Price: ${avg_exit:.2f}")

def main():
    """Run all demo functions"""
    print("🚀 GAUSSIAN CHANNEL STRATEGY - COMPREHENSIVE DEMO")
    print("This demo will showcase the strategy's capabilities")
    print("Please ensure you have internet connection for data download")
    
    try:
        # Run basic demo
        run_basic_demo()
        
        # Run comparison demo
        run_comparison_demo()
        
        # Run multi-symbol demo
        run_multi_symbol_demo()
        
        # Run detailed example with charts
        print("\n" + "=" * 60)
        print("Would you like to see detailed charts? (This will open matplotlib windows)")
        response = input("Enter 'y' for yes, any other key to skip: ").lower()
        
        if response == 'y':
            run_detailed_example()
        
        print("\n" + "=" * 60)
        print("✅ DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Next steps:")
        print("1. Review the code in gaussian_channel_strategy.py")
        print("2. Experiment with different parameters")
        print("3. Test on your own symbols and date ranges")
        print("4. Read the README.md for detailed documentation")
        print("\n⚠️  Remember: Past performance doesn't guarantee future results!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("Please check your internet connection and ensure all dependencies are installed:")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    main() 