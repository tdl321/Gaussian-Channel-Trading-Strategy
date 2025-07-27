#!/usr/bin/env python3
"""
Simple Bitcoin CSV Test

Tests the Gaussian Channel strategy on your Bitcoin Price.csv file
"""

from gaussian_channel_strategy import GaussianChannelStrategy
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def test_bitcoin_csv():
    """
    Simple test of Bitcoin CSV loading and strategy execution
    """
    print("🚀 TESTING BITCOIN CSV WITH GAUSSIAN CHANNEL STRATEGY")
    print("=" * 60)
    
    # Step 1: Load and prepare Bitcoin data manually
    print("📊 Loading Bitcoin data...")
    
    # Load the CSV
    data = pd.read_csv('Bitcoin Price.csv')
    
    # Convert date and set as index
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
    
    # Add Volume and remove Currency
    data['Volume'] = 0
    if 'Currency' in data.columns:
        data.drop('Currency', axis=1, inplace=True)
    
    # Convert to numeric
    for col in ['Open', 'High', 'Low', 'Close']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    data.dropna(inplace=True)
    data.sort_index(inplace=True)
    
    print(f"✅ Loaded {len(data)} rows of Bitcoin data")
    print(f"📅 Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
    print(f"💰 Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
    
    # Step 2: Initialize strategy  
    print(f"\n📈 Initializing Gaussian Channel Strategy...")
    strategy = GaussianChannelStrategy(
        poles=4,
        period=144,
        multiplier=1.414,
        atr_spacing=0.4,
        max_pyramids=5,
        initial_capital=10000,
        position_size_pct=0.65,
        leverage=5.0,  # Pine Script equivalent 5x leverage
        start_date='2017-01-01',
        end_date='2021-01-04'
    )
    
    # Step 3: Filter data to strategy date range
    start_date = pd.to_datetime('2017-01-01')
    end_date = pd.to_datetime('2021-01-04')
    
    filtered_data = data[(data.index >= start_date) & (data.index <= end_date)]
    
    print(f"📊 Filtered to {len(filtered_data)} rows for analysis period")
    print(f"📅 Analysis period: {filtered_data.index[0].strftime('%Y-%m-%d')} to {filtered_data.index[-1].strftime('%Y-%m-%d')}")
    
    if len(filtered_data) < 200:
        print("❌ Not enough data for analysis")
        return None
    
    # Step 4: Add required technical indicators manually
    print(f"\n🔧 Calculating technical indicators...")
    
    # Calculate hlc3
    filtered_data['hlc3'] = (filtered_data['High'] + filtered_data['Low'] + filtered_data['Close']) / 3
    
    # Calculate true range
    filtered_data['true_range'] = np.maximum(
        filtered_data['High'] - filtered_data['Low'],
        np.maximum(
            abs(filtered_data['High'] - filtered_data['Close'].shift(1)),
            abs(filtered_data['Low'] - filtered_data['Close'].shift(1))
        )
    )
    
    # Calculate ATR
    filtered_data['atr'] = filtered_data['true_range'].rolling(14).mean()
    
    # Calculate SMA
    filtered_data['sma_200'] = filtered_data['Close'].rolling(200).mean()
    
    print(f"✅ Technical indicators calculated")
    
    # Step 5: Run strategy
    print(f"\n🎯 Running Gaussian Channel Strategy...")
    
    try:
        # Prepare signals
        prepared_data = strategy.prepare_signals(filtered_data)
        
        # Run backtest
        strategy.backtest_manual(prepared_data)
        
        # Calculate performance
        metrics = strategy.calculate_performance_metrics()
        
        # Step 6: Display results
        print(f"\n" + "="*50)
        print("🎉 BITCOIN STRATEGY RESULTS")
        print("="*50)
        
        for metric, value in metrics.items():
            print(f"{metric:<25}: {value}")
        
        # Trade summary
        buy_trades = [t for t in strategy.trade_log if t['action'] == 'BUY']
        sell_trades = [t for t in strategy.trade_log if t['action'] == 'SELL']
        
        print(f"\n📊 Trading Summary:")
        print(f"   Buy Signals: {len(buy_trades)}")
        print(f"   Sell Signals: {len(sell_trades)}")
        
        if buy_trades:
            avg_entry = sum(t['price'] for t in buy_trades) / len(buy_trades)
            min_entry = min(t['price'] for t in buy_trades)
            max_entry = max(t['price'] for t in buy_trades)
            
            print(f"   Average Entry: ${avg_entry:,.2f}")
            print(f"   Entry Range: ${min_entry:,.2f} - ${max_entry:,.2f}")
        
        # Generate charts
        print(f"\n📊 Generating charts...")
        strategy.plot_results(prepared_data, save_path='bitcoin_strategy_chart.png')
        print(f"✅ Chart saved: bitcoin_strategy_chart.png")
        
        return {
            'data': prepared_data,
            'metrics': metrics,
            'strategy': strategy
        }
        
    except Exception as e:
        print(f"❌ Strategy execution failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def quick_bitcoin_performance():
    """
    Quick performance summary for different periods
    """
    print("\n" + "="*60)
    print("📊 BITCOIN PERFORMANCE ACROSS DIFFERENT PERIODS")
    print("="*60)
    
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
    
    # Test different periods
    periods = [
        ('2017-2019', '2017-01-01', '2019-12-31'),
        ('2018-2020', '2018-01-01', '2020-12-31'), 
        ('2019-2021', '2019-01-01', '2021-01-04')
    ]
    
    print(f"{'Period':<12} {'Days':<8} {'BTC Return %':<15} {'Strategy Return %':<18} {'Trades':<8}")
    print("-" * 70)
    
    for period_name, start_str, end_str in periods:
        start_date = pd.to_datetime(start_str)
        end_date = pd.to_datetime(end_str)
        
        period_data = data[(data.index >= start_date) & (data.index <= end_date)]
        
        if len(period_data) < 200:
            print(f"{period_name:<12} {'Too short':<50}")
            continue
        
        # Calculate Bitcoin buy-and-hold return
        btc_return = (period_data['Close'].iloc[-1] / period_data['Close'].iloc[0] - 1) * 100
        
        # Quick strategy test
        try:
            import numpy as np
            
            strategy = GaussianChannelStrategy(
                start_date=start_str,
                end_date=end_str,
                initial_capital=10000
            )
            
            # Add required indicators
            period_data['hlc3'] = (period_data['High'] + period_data['Low'] + period_data['Close']) / 3
            period_data['true_range'] = np.maximum(
                period_data['High'] - period_data['Low'],
                np.maximum(
                    abs(period_data['High'] - period_data['Close'].shift(1)),
                    abs(period_data['Low'] - period_data['Close'].shift(1))
                )
            )
            period_data['atr'] = period_data['true_range'].rolling(14).mean()
            period_data['sma_200'] = period_data['Close'].rolling(200).mean()
            
            # Run strategy
            period_data = strategy.prepare_signals(period_data)
            strategy.backtest_manual(period_data)
            metrics = strategy.calculate_performance_metrics()
            
            strategy_return = metrics['Total Return (%)']
            num_trades = metrics['Number of Trades']
            
            print(f"{period_name:<12} {len(period_data):<8} {btc_return:<15.1f} {strategy_return:<18.1f} {num_trades:<8}")
            
        except Exception as e:
            print(f"{period_name:<12} {len(period_data):<8} {btc_return:<15.1f} {'Error':<18} {'N/A':<8}")

if __name__ == "__main__":
    import numpy as np
    
    # Run main test
    result = test_bitcoin_csv()
    
    if result:
        print(f"\n✅ SUCCESS! Bitcoin analysis completed successfully!")
        
        # Show quick performance comparison
        quick_bitcoin_performance()
        
        print(f"\n🎉 ANALYSIS COMPLETE!")
        print(f"📁 Files generated:")
        print(f"   • bitcoin_strategy_chart.png")
        print(f"\n💡 Your Bitcoin data from 2013-2021 has been successfully analyzed!")
        
    else:
        print(f"\n❌ Analysis failed. Please check the error messages above.") 