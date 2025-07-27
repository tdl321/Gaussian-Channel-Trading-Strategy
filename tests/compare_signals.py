#!/usr/bin/env python3
"""
Signal Comparison Tool

This script helps identify exact differences between Pine Script and Python implementations
of the Gaussian Channel strategy by analyzing the signal generation in detail.
"""

from gaussian_channel_strategy import GaussianChannelStrategy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def load_bitcoin_data():
    """Load and prepare Bitcoin data"""
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
    
    return data

def debug_gaussian_filter(data, start_date='2017-01-01', end_date='2021-01-04'):
    """
    Debug the Gaussian filter calculation step by step
    """
    print("🔍 DEBUGGING GAUSSIAN FILTER IMPLEMENTATION")
    print("=" * 60)
    
    # Filter data to analysis period
    filtered_data = data[(data.index >= start_date) & (data.index <= end_date)].copy()
    
    # Add technical indicators
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
    
    print(f"📊 Data period: {filtered_data.index[0]} to {filtered_data.index[-1]}")
    print(f"📊 Total bars: {len(filtered_data)}")
    
    # Initialize strategy  
    strategy = GaussianChannelStrategy(
        poles=4,
        period=144,
        multiplier=1.414,
        atr_spacing=0.4,
        start_date=start_date,
        end_date=end_date
    )
    
    # Test Gaussian filter with Pine Script equivalent parameters
    print(f"\n🔧 Gaussian Filter Parameters:")
    print(f"   Poles (N): {strategy.poles}")
    print(f"   Period: {strategy.period}")  
    print(f"   Multiplier: {strategy.multiplier}")
    print(f"   Alpha: {strategy.gaussian_filter.alpha:.6f}")
    
    # Calculate Pine Script equivalent values
    beta = (1 - np.cos(4 * np.arcsin(1) / strategy.period)) / (np.power(1.414, 2/strategy.poles) - 1)
    alpha_pine = -beta + np.sqrt(np.power(beta, 2) + 2*beta)
    lag_pine = (strategy.period - 1) / (2 * strategy.poles)
    
    print(f"\n📐 Pine Script Calculations:")
    print(f"   Beta: {beta:.6f}")
    print(f"   Alpha (Pine): {alpha_pine:.6f}")
    print(f"   Lag: {lag_pine:.6f}")
    
    # Compare with our implementation
    print(f"\n⚖️  Comparison:")
    print(f"   Alpha difference: {abs(strategy.gaussian_filter.alpha - alpha_pine):.10f}")
    print(f"   {'✅ Match' if abs(strategy.gaussian_filter.alpha - alpha_pine) < 1e-10 else '❌ Mismatch'}")
    
    return filtered_data, strategy

def analyze_signal_differences(data, strategy):
    """
    Analyze where signals differ from expected Pine Script behavior
    """
    print(f"\n🎯 ANALYZING SIGNAL GENERATION")
    print("=" * 60)
    
    # Prepare signals
    prepared_data = strategy.prepare_signals(data)
    
    # Key signal columns
    signal_cols = ['filt', 'hband', 'lband', 'green_channel', 'green_entry', 'red_entry', 'exit_signal']
    
    print(f"📊 Signal Statistics:")
    for col in signal_cols:
        if col in prepared_data.columns:
            if prepared_data[col].dtype == bool:
                count = prepared_data[col].sum()
                print(f"   {col}: {count} signals")
            else:
                valid_count = prepared_data[col].notna().sum()
                print(f"   {col}: {valid_count} valid values")
    
    # Check for common issues
    print(f"\n🔍 Common Issues Check:")
    
    # 1. NaN values in filter
    nan_filt = prepared_data['filt'].isna().sum()
    print(f"   NaN values in filter: {nan_filt}")
    
    # 2. Green channel calculation
    green_true = prepared_data['green_channel'].sum()
    green_false = (~prepared_data['green_channel']).sum()
    print(f"   Green channel periods: {green_true}")
    print(f"   Red channel periods: {green_false}")
    
    # 3. Entry signal breakdown
    total_entries = prepared_data['green_entry'].sum() + prepared_data['red_entry'].sum()
    print(f"   Total entry signals: {total_entries}")
    print(f"   Green entries: {prepared_data['green_entry'].sum()}")
    print(f"   Red entries: {prepared_data['red_entry'].sum()}")
    
    # 4. Exit signals
    print(f"   Exit signals: {prepared_data['exit_signal'].sum()}")
    
    return prepared_data

def create_detailed_comparison_chart(data, window_start='2020-01-01', window_end='2020-06-01'):
    """
    Create a detailed chart of a specific time window to compare signals
    """
    print(f"\n📊 CREATING DETAILED COMPARISON CHART")
    print(f"Window: {window_start} to {window_end}")
    print("=" * 60)
    
    # Filter to window
    window_data = data[(data.index >= window_start) & (data.index <= window_end)].copy()
    
    if len(window_data) < 50:
        print("❌ Not enough data in window")
        return
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12))
    
    # Plot 1: Price with Gaussian Channel
    ax1.plot(window_data.index, window_data['Close'], 'k-', label='Close Price', linewidth=1)
    ax1.plot(window_data.index, window_data['filt'], 'b-', label='Gaussian Filter', linewidth=2)
    ax1.plot(window_data.index, window_data['hband'], 'r-', label='Upper Band', alpha=0.7)
    ax1.plot(window_data.index, window_data['lband'], 'r-', label='Lower Band', alpha=0.7)
    
    # Add entry/exit signals
    entry_dates = window_data[window_data['green_entry'] | window_data['red_entry']].index
    entry_prices = window_data.loc[entry_dates, 'Close']
    ax1.scatter(entry_dates, entry_prices, color='green', marker='^', s=100, zorder=5, label='Entries')
    
    exit_dates = window_data[window_data['exit_signal']].index
    exit_prices = window_data.loc[exit_dates, 'Close']
    ax1.scatter(exit_dates, exit_prices, color='red', marker='v', s=100, zorder=5, label='Exits')
    
    ax1.set_title(f'Gaussian Channel Signals Detail ({window_start} to {window_end})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Channel state and conditions
    ax2.fill_between(window_data.index, 0, 1, where=window_data['green_channel'], 
                     color='green', alpha=0.3, label='Green Channel')
    ax2.fill_between(window_data.index, 0, 1, where=~window_data['green_channel'], 
                     color='red', alpha=0.3, label='Red Channel')
    
    # Mark conditions
    ax2.scatter(window_data[window_data['price_above_band']].index, 
               [0.7] * window_data['price_above_band'].sum(), 
               color='blue', marker='|', s=50, label='Price > Upper Band')
    
    ax2.set_title('Channel State and Entry Conditions')
    ax2.set_ylabel('Channel State')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Filter slope and momentum
    window_data['filt_slope'] = window_data['filt'].diff()
    ax3.plot(window_data.index, window_data['filt_slope'], 'purple', linewidth=1, label='Filter Slope')
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_title('Filter Slope (Trend Direction)')
    ax3.set_ylabel('Slope')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('detailed_signal_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print signal details for this window
    print(f"\n📋 Window Signal Details:")
    print(f"   Entries: {(window_data['green_entry'] | window_data['red_entry']).sum()}")
    print(f"   Exits: {window_data['exit_signal'].sum()}")
    print(f"   Green channel %: {window_data['green_channel'].mean()*100:.1f}%")
    print(f"   Price above band events: {window_data['price_above_band'].sum()}")

def identify_pine_vs_python_differences():
    """
    Identify specific areas where Pine Script and Python might differ
    """
    print(f"\n🔍 POTENTIAL DIFFERENCES BETWEEN PINE SCRIPT & PYTHON")
    print("=" * 60)
    
    differences = [
        {
            'area': 'Data Indexing',
            'pine_script': 'Uses [1] for previous bar, [0] for current',
            'python': 'Uses .shift(1) for previous bar',
            'impact': 'Could cause off-by-one bar differences'
        },
        {
            'area': 'Confirmed Bar Logic',
            'pine_script': 'barstate.isconfirmed checks if bar is closed',
            'python': 'Uses .shift(1) to simulate confirmed bars',
            'impact': 'Timing of signals might differ by 1 bar'
        },
        {
            'area': 'Gaussian Filter',
            'pine_script': 'Complex recursive filter implementation',
            'python': 'Exponential smoothing approximation',
            'impact': 'Filter values might not match exactly'
        },
        {
            'area': 'True Range Calculation',
            'pine_script': 'ta.tr(true) function',
            'python': 'Manual calculation with np.maximum',
            'impact': 'Should be identical but worth checking'
        },
        {
            'area': 'Green Channel Detection',
            'pine_script': 'filt[1] > filt[2]',
            'python': 'filt.shift(1) > filt.shift(2)',
            'impact': 'Should be equivalent'
        }
    ]
    
    for diff in differences:
        print(f"\n📌 {diff['area']}:")
        print(f"   Pine Script: {diff['pine_script']}")
        print(f"   Python: {diff['python']}")
        print(f"   Impact: {diff['impact']}")

def recommend_fixes():
    """
    Recommend specific fixes to align the implementations
    """
    print(f"\n🔧 RECOMMENDED FIXES")
    print("=" * 60)
    
    fixes = [
        "1. Replace exponential smoothing with exact Pine Script Gaussian filter",
        "2. Verify ATR calculation matches Pine Script exactly",
        "3. Check confirmed bar logic - ensure signals use previous bar data",
        "4. Verify green channel calculation timing",
        "5. Test with smaller date ranges to isolate differences",
        "6. Add debug prints to compare intermediate values",
        "7. Check if pyramiding logic matches exactly"
    ]
    
    for fix in fixes:
        print(f"   {fix}")

def main():
    """
    Run complete signal comparison analysis
    """
    # Load data
    btc_data = load_bitcoin_data()
    
    # Debug filter implementation
    filtered_data, strategy = debug_gaussian_filter(btc_data)
    
    # Analyze signal generation
    prepared_data = analyze_signal_differences(filtered_data, strategy)
    
    # Create detailed comparison chart
    create_detailed_comparison_chart(prepared_data, '2020-01-01', '2020-06-01')
    
    # Identify potential differences
    identify_pine_vs_python_differences()
    
    # Recommend fixes
    recommend_fixes()
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"1. Review the detailed comparison chart: detailed_signal_comparison.png")
    print(f"2. Check if Gaussian filter values match Pine Script exactly") 
    print(f"3. Verify confirmed bar timing logic")
    print(f"4. Test with known Pine Script signal dates/prices")

if __name__ == "__main__":
    main() 