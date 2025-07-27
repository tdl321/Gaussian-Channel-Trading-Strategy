#!/usr/bin/env python3
"""
Bitcoin Price Analysis using Gaussian Channel Strategy

This script loads the Gemini Bitcoin dataset and runs the Gaussian Channel strategy on it.
Updated to use the more comprehensive Gemini_BTCUSD_d.csv data file.
"""

from gaussian_channel_strategy import GaussianChannelStrategy, GaussianChannelBacktraderStrategy
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import backtrader for professional backtesting
import backtrader as bt

def load_gemini_csv(csv_path='../data/Gemini_BTCUSD_d.csv'):
    """
    Load the Gemini Bitcoin CSV format with comprehensive OHLC and volume data
    """
    try:
        # Load the CSV, skipping the first header line
        data = pd.read_csv(csv_path, skiprows=1)
        
        print(f"✅ Loaded Gemini Bitcoin CSV with {len(data)} rows")
        
        # Convert the date column and set as index
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)
        
        # Rename columns to standard OHLC format (capitalize first letter)
        column_mapping = {
            'open': 'Open',
            'high': 'High', 
            'low': 'Low',
            'close': 'Close',
            'Volume BTC': 'Volume'
        }
        
        data.rename(columns=column_mapping, inplace=True)
        
        # Keep only the OHLC and Volume columns we need
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Convert to numeric and handle any issues
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Remove any rows with NaN values
        data.dropna(inplace=True)
        
        # Calculate HLC3 (High + Low + Close) / 3 - needed for the strategy
        data['hlc3'] = (data['High'] + data['Low'] + data['Close']) / 3
        
        # Calculate True Range - needed for the strategy
        # True Range = max(High-Low, abs(High-PrevClose), abs(Low-PrevClose))
        prev_close = data['Close'].shift(1)
        tr1 = data['High'] - data['Low']
        tr2 = abs(data['High'] - prev_close)
        tr3 = abs(data['Low'] - prev_close)
        data['true_range'] = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Calculate ATR (Average True Range) - 14-period moving average of True Range
        data['atr'] = data['true_range'].rolling(window=14, min_periods=1).mean()
        
        # Sort by date to ensure proper chronological order (oldest first)
        data.sort_index(inplace=True)
        
        print(f"📅 Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
        print(f"💰 Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
        print(f"📊 Final dataset: {len(data)} clean rows")
        print(f"📈 Volume range: {data['Volume'].min():.2f} - {data['Volume'].max():.2f} BTC")
        
        return data
        
    except Exception as e:
        print(f"❌ Error loading Gemini CSV: {e}")
        return None

def load_bitcoin_csv(csv_path='../data/Bitcoin Price.csv'):
    """
    Load the original Bitcoin Price.csv format (kept for backward compatibility)
    """
    try:
        # Load the CSV
        data = pd.read_csv(csv_path)
        
        print(f"✅ Loaded Bitcoin CSV with {len(data)} rows")
        print(f"📅 Date range: {data['Date'].iloc[0]} to {data['Date'].iloc[-1]}")
        
        # Convert date column
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        
        # Rename columns to standard OHLC format
        column_mapping = {
            '24h Open (USD)': 'Open',
            '24h High (USD)': 'High', 
            '24h Low (USD)': 'Low',
            'Closing Price (USD)': 'Close'
        }
        
        data.rename(columns=column_mapping, inplace=True)
        
        # Add Volume column (set to 0 since not provided)
        data['Volume'] = 0
        
        # Remove Currency column as it's not needed
        if 'Currency' in data.columns:
            data.drop('Currency', axis=1, inplace=True)
        
        # Convert to numeric and handle any issues
        numeric_cols = ['Open', 'High', 'Low', 'Close']
        for col in numeric_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Remove any rows with NaN values
        data.dropna(inplace=True)
        
        # Calculate HLC3 (High + Low + Close) / 3 - needed for the strategy
        data['hlc3'] = (data['High'] + data['Low'] + data['Close']) / 3
        
        # Calculate True Range - needed for the strategy
        # True Range = max(High-Low, abs(High-PrevClose), abs(Low-PrevClose))
        prev_close = data['Close'].shift(1)
        tr1 = data['High'] - data['Low']
        tr2 = abs(data['High'] - prev_close)
        tr3 = abs(data['Low'] - prev_close)
        data['true_range'] = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Calculate ATR (Average True Range) - 14-period moving average of True Range
        data['atr'] = data['true_range'].rolling(window=14, min_periods=1).mean()
        
        # Sort by date to ensure proper order
        data.sort_index(inplace=True)
        
        print(f"💰 Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
        print(f"📊 Final dataset: {len(data)} clean rows")
        
        return data
        
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return None

def analyze_bitcoin_data(data, chart_title="Bitcoin Analysis"):
    """
    Run the full Gaussian Channel analysis on the provided data
    """
    if data is None or len(data) == 0:
        print("❌ No data to analyze")
        return None
    
    print("\n" + "="*60)
    print(f"🧮 RUNNING GAUSSIAN CHANNEL ANALYSIS - {chart_title}")
    print("="*60)
    
    try:
        # Initialize the strategy with parameters optimized for Bitcoin
        strategy = GaussianChannelStrategy(
            poles=4,           # Gaussian filter complexity
            period=144,        # Sampling period (about 5 months for daily data)
            multiplier=1.414,  # Channel width multiplier
            initial_capital=10000,
            commission=0.001   # 0.1% commission
        )
        
        # Prepare signals and run the backtest
        print("🔄 Preparing signals...")
        data_with_signals = strategy.prepare_signals(data)
        
        print("🔄 Running backtest...")
        strategy.backtest_manual(data_with_signals)
        
        # Calculate performance metrics
        print("🔄 Calculating performance metrics...")
        results = strategy.calculate_performance_metrics()
        
        if results is not None:
            print("\n✅ Analysis completed successfully!")
            
            # Print key statistics
            total_return = results['Total Return (%)']
            sharpe_ratio = results['Sharpe Ratio']
            max_drawdown = results['Max Drawdown (%)']
            win_rate = results['Win Rate (%)']
            num_trades = results['Number of Trades']
            
            print(f"\n📊 KEY PERFORMANCE METRICS:")
            print(f"💹 Total Return: {total_return:.2f}%")
            print(f"⚡ Sharpe Ratio: {sharpe_ratio:.2f}")
            print(f"📉 Max Drawdown: {max_drawdown:.2f}%")
            print(f"🎯 Win Rate: {win_rate:.1f}%")
            print(f"🎲 Total Trades: {num_trades}")
            
            # Generate the chart
            print(f"\n📈 Generating {chart_title} chart...")
            strategy.plot_results(data_with_signals, save_path='gaussian_strategy_results.png')
            
            return results
        else:
            print("❌ Strategy analysis failed")
            return None
            
    except Exception as e:
        print(f"❌ Error in analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_bitcoin_data_backtrader(data, chart_title="Bitcoin Analysis"):
    """
    Run the Gaussian Channel analysis using Backtrader for professional backtesting
    """
    if data is None or len(data) == 0:
        print("❌ No data to analyze")
        return None
    
    print("\n" + "="*60)
    print(f"🧮 RUNNING BACKTRADER GAUSSIAN CHANNEL ANALYSIS - {chart_title}")
    print("="*60)
    
    try:
        # Create a backtrader cerebro engine
        cerebro = bt.Cerebro()
        
        # Convert pandas DataFrame to backtrader data feed
        # Ensure we have the required columns in the right order
        data_bt = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        
        # Create backtrader data feed
        data_feed = bt.feeds.PandasData(
            dataname=data_bt,
            datetime=None,  # Use index as datetime
            open=0,
            high=1,
            low=2,
            close=3,
            volume=4,
            openinterest=-1
        )
        
        # Add data to cerebro
        cerebro.adddata(data_feed)
        
        # Add strategy with optimized parameters for Bitcoin
        cerebro.addstrategy(
            GaussianChannelBacktraderStrategy,
            poles=4,
            period=144,
            multiplier=1.414,
            atr_spacing=0.4,
            max_pyramids=5,
            position_size_pct=0.65,
            commission=0.001
        )
        
        # Set initial cash
        initial_cash = 10000
        cerebro.broker.setcash(initial_cash)
        
        # Set commission
        cerebro.broker.setcommission(commission=0.001)  # 0.1%
        
        # Add analyzers for performance metrics
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        
        print("🔄 Running Backtrader backtest...")
        
        # Run the backtest
        results = cerebro.run()
        strategy_result = results[0]
        
        # Get final portfolio value
        final_value = cerebro.broker.getvalue()
        
        print("🔄 Calculating performance metrics...")
        
        # Extract performance metrics
        sharpe_ratio = strategy_result.analyzers.sharpe.get_analysis().get('sharperatio', 0)
        if sharpe_ratio is None:
            sharpe_ratio = 0
            
        drawdown_info = strategy_result.analyzers.drawdown.get_analysis()
        max_drawdown = drawdown_info.get('max', {}).get('drawdown', 0)
        
        trade_info = strategy_result.analyzers.trades.get_analysis()
        total_trades = trade_info.get('total', {}).get('total', 0)
        won_trades = trade_info.get('won', {}).get('total', 0)
        win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
        
        returns_info = strategy_result.analyzers.returns.get_analysis()
        total_return = ((final_value - initial_cash) / initial_cash) * 100
        
        # Create results dictionary compatible with existing code
        results_dict = {
            'Total Return (%)': total_return,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown (%)': max_drawdown,
            'Win Rate (%)': win_rate,
            'Number of Trades': total_trades,
            'Initial Capital': initial_cash,
            'Final Value': final_value
        }
        
        print("\n✅ Backtrader analysis completed successfully!")
        
        # Print key statistics
        print(f"\n📊 KEY PERFORMANCE METRICS:")
        print(f"💹 Total Return: {total_return:.2f}%")
        print(f"⚡ Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"📉 Max Drawdown: {max_drawdown:.2f}%")
        print(f"🎯 Win Rate: {win_rate:.1f}%")
        print(f"🎲 Total Trades: {total_trades}")
        print(f"💰 Final Value: ${final_value:,.2f}")
        
        # Plot the results
        print(f"\n📈 Generating {chart_title} chart...")
        try:
            import matplotlib.pyplot as plt
            plt.style.use('default')
            cerebro.plot(style='candlestick', barup='green', bardown='red', figsize=(15, 10))
            plt.savefig('../results/gaussian_strategy_backtrader_results.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("📈 Chart saved as 'gaussian_strategy_backtrader_results.png'")
        except Exception as e:
            print(f"⚠️  Chart generation failed: {e}")
        
        return results_dict
        
    except Exception as e:
        print(f"❌ Error in Backtrader analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """
    Main analysis function - now uses the comprehensive Gemini dataset
    """
    print("🚀 Bitcoin Gaussian Channel Strategy Analysis")
    print("📊 Using comprehensive Gemini Exchange dataset")
    print("-" * 50)
    
    # Load the Gemini data (more comprehensive)
    print("\n💾 Loading Gemini Bitcoin data...")
    data = load_gemini_csv('../data/Gemini_BTCUSD_d.csv')
    
    if data is not None:
        # Run analysis on the full Gemini dataset using Backtrader
        results = analyze_bitcoin_data_backtrader(data, "Gemini BTCUSD Daily")
        
        if results:
            print(f"\n🎉 Analysis complete! Chart saved as 'gaussian_strategy_backtrader_results.png'")
            print(f"📈 Total return over {len(data)} days: {results['Total Return (%)']:.2f}%")
            
            # Compare data quality
            days_total = len(data)
            years_total = days_total / 365.25
            print(f"\n📊 Dataset Quality:")
            print(f"   • Total Days: {days_total}")
            print(f"   • Years of Data: {years_total:.1f}")
            print(f"   • Start Date: {data.index[0].strftime('%Y-%m-%d')}")
            print(f"   • End Date: {data.index[-1].strftime('%Y-%m-%d')}")
            print(f"   • Has Volume Data: Yes")
            print(f"   • Average Daily Volume: {data['Volume'].mean():.2f} BTC")
        
    else:
        print("⚠️  Gemini data not available, trying original dataset...")
        # Fallback to original data if Gemini data fails
        data = load_bitcoin_csv('../data/Bitcoin Price.csv')
        if data is not None:
            results = analyze_bitcoin_data_backtrader(data, "Bitcoin Price Daily")
        else:
            print("❌ No valid data source available")

if __name__ == "__main__":
    # Quick test to verify data loading
    print("🔍 Testing data loading...")
    test_data = load_gemini_csv('../data/Gemini_BTCUSD_d.csv')
    
    if test_data is not None and len(test_data) > 100:
        print("✅ Gemini data loaded successfully!")
        main()
    else:
        print("❌ Quick test failed - please check your Gemini_BTCUSD_d.csv file")
        print("   Ensure the file is in the current directory") 