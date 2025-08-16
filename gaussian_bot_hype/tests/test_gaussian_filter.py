#!/usr/bin/env python3
"""
Comprehensive Test Script for Gaussian Channel Filter
Combines testing, visualization, and backtesting in one script
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import strategy modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy.gaussian_filter import GaussianChannelFilter
from strategy.signals import SignalGenerator
from strategy.backtest import run_backtrader_backtest, analyze_backtrader_results, plot_backtrader_results, run_comprehensive_backtest





def load_btc_data():
    """Load and prepare the full BTC dataset"""
    
    data_path = os.path.join('data', 'btc_1d_data_2018_to_2025.csv')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"BTC data not found at {data_path}")
    
    print("📊 Loading full BTC dataset...")
    
    # Load the CSV data
    df = pd.read_csv(data_path)
    
    # Convert timestamp to datetime
    df['Date'] = pd.to_datetime(df['Open time'])
    df.set_index('Date', inplace=True)
    
    # Rename columns to match our strategy format
    df = df.rename(columns={
        'Open': 'Open',
        'High': 'High', 
        'Low': 'Low',
        'Close': 'Close',
        'Volume': 'Volume'
    })
    
    # Calculate HLC3 and True Range
    df['hlc3'] = (df['High'] + df['Low'] + df['Close']) / 3
    
    df['tr1'] = df['High'] - df['Low']
    df['tr2'] = abs(df['High'] - df['Close'].shift(1))
    df['tr3'] = abs(df['Low'] - df['Close'].shift(1))
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    print(f"✅ Loaded {len(df)} days of BTC data")
    print(f"   Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"   Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
    
    return df


def test_gaussian_filter(df, poles=6, period=144, multiplier=1.414):
    """Test that Gaussian filter works with live trading parameters"""
    
    print("🧪 Testing Gaussian filter...")
    
    # Initialize Gaussian filter with live trading parameters
    gaussian_filter = GaussianChannelFilter(
        poles=poles,
        period=period,
        multiplier=multiplier
    )
    
    # Apply filter
    filt_result, hband_result, lband_result = gaussian_filter.apply_filter(
        df['hlc3'], 
        df['true_range']
    )
    
    # Check that we get results
    assert not filt_result.isna().all(), "Filter should produce non-NaN results"
    assert not hband_result.isna().all(), "Upper band should produce non-NaN results"
    assert not lband_result.isna().all(), "Lower band should produce non-NaN results"
    
    # Check that bands are reasonable
    valid_mask = ~filt_result.isna()
    if valid_mask.sum() > 0:
        assert (hband_result[valid_mask] > lband_result[valid_mask]).all(), "Upper band should be above lower band"
        assert (hband_result[valid_mask] > filt_result[valid_mask]).all(), "Upper band should be above filter"
        assert (lband_result[valid_mask] < filt_result[valid_mask]).all(), "Lower band should be below filter"
    
    print("✅ Gaussian filter test passed!")
    print(f"   - Generated {len(df)} bars of test data")
    print(f"   - Filter parameters: {gaussian_filter.poles} poles, {gaussian_filter.period} period, {gaussian_filter.multiplier} multiplier")
    print(f"   - Valid filter values: {valid_mask.sum()}/{len(df)} ({valid_mask.sum()/len(df)*100:.1f}%)")
    if valid_mask.any():
        print(f"   - First valid date: {df.index[valid_mask][0].strftime('%Y-%m-%d')}")
    
    return df, filt_result, hband_result, lband_result, gaussian_filter


def test_signal_generation(df, gaussian_filter):
    """Test signal generation with the Gaussian filter"""
    
    print("🎯 Testing signal generation...")
    
    # Create config parameters for signal generator
    config_params = {
        'POLES': gaussian_filter.poles,
        'PERIOD': gaussian_filter.period,
        'MULTIPLIER': gaussian_filter.multiplier
    }
    
    # Initialize signal generator
    signal_generator = SignalGenerator(gaussian_filter, config_params)
    
    # Prepare signals
    df_with_signals = signal_generator.prepare_signals(df.copy())
    
    # Get signal summary
    signal_summary = signal_generator.get_signal_summary(df_with_signals)
    
    print("✅ Signal generation test passed!")
    print(f"   - Total bars: {signal_summary.get('total_bars', 0)}")
    print(f"   - Entry signals: {signal_summary.get('entries', 0)}")
    print(f"   - Exit signals: {signal_summary.get('exits', 0)}")
    print(f"   - Entry rate: {signal_summary.get('entry_rate', 0)*100:.2f}%")
    
    return df_with_signals, signal_generator


def visualize_gaussian_filter(df, filt_result, hband_result, lband_result, gaussian_filter, 
                            save_plot=True, show_plot=True, title_suffix=""):
    """Create comprehensive visualization of Gaussian filter results"""
    
    # Set up the plot
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[3, 1])
    
    # Get valid data range
    valid_mask = ~filt_result.isna()
    
    # Plot 1: Price and Gaussian Filter
    ax1.plot(df.index, df['hlc3'], label='HLC3 Price', color='black', alpha=0.7, linewidth=1)
    ax1.plot(filt_result.index, filt_result, label='Gaussian Filter', color='blue', linewidth=2)
    ax1.plot(hband_result.index, hband_result, label='Upper Band', color='red', alpha=0.7, linewidth=1.5)
    ax1.plot(lband_result.index, lband_result, label='Lower Band', color='red', alpha=0.7, linewidth=1.5)
    
    # Fill the band area
    ax1.fill_between(df.index, hband_result, lband_result, alpha=0.1, color='red', label='Channel')
    
    # Formatting
    ax1.set_title(f'Gaussian Channel Filter Test{title_suffix}\n'
                  f'Parameters: {gaussian_filter.poles} poles, {gaussian_filter.period} period, '
                  f'{gaussian_filter.multiplier} multiplier', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 2: True Range
    ax2.plot(df.index, df['true_range'], label='True Range', color='green', alpha=0.7)
    ax2.set_ylabel('True Range', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis dates for second plot
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if requested
    if save_plot:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gaussian_filter_test_{timestamp}.png"
        save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Chart saved to: {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    
    plt.close()


def visualize_signals(df_with_signals, gaussian_filter, save_plot=True, show_plot=True, title_suffix=""):
    """Create comprehensive visualization of trading signals"""
    
    print("🎨 Creating signal visualization...")
    
    # Set up the plot with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Create grid layout
    gs = fig.add_gridspec(5, 1, height_ratios=[3, 1, 1, 1, 1], hspace=0.1)
    
    # Plot 1: Main price chart with signals
    ax1 = fig.add_subplot(gs[0])
    
    # Plot price data
    ax1.plot(df_with_signals.index, df_with_signals['Close'], label='BTC Close Price', color='black', alpha=0.8, linewidth=1.5)
    ax1.plot(df_with_signals.index, df_with_signals['hlc3'], label='HLC3', color='gray', alpha=0.6, linewidth=1)
    
    # Plot Gaussian filter and bands (confirmed data for non-repainting)
    ax1.plot(df_with_signals.index, df_with_signals['filt_confirmed'], label='Gaussian Filter (Confirmed)', color='blue', linewidth=2)
    ax1.plot(df_with_signals.index, df_with_signals['hband_confirmed'], label='Upper Band (Confirmed)', color='red', alpha=0.7, linewidth=1.5)
    ax1.plot(df_with_signals.index, df_with_signals['lband_confirmed'], label='Lower Band (Confirmed)', color='red', alpha=0.7, linewidth=1.5)
    
    # Fill the band area
    ax1.fill_between(df_with_signals.index, df_with_signals['hband_confirmed'], df_with_signals['lband_confirmed'], alpha=0.1, color='red', label='Channel')
    
    # Plot entry signals - small pinpoint markers
    entry_points = df_with_signals[df_with_signals['entry_signal']]
    if len(entry_points) > 0:
        ax1.scatter(entry_points.index, entry_points['Close'], color='blue', marker='o', s=8, label='Entry Signal', zorder=5, alpha=0.8)
    
    # Plot exit signals - small pinpoint markers
    exit_points = df_with_signals[df_with_signals['exit_signal']]
    if len(exit_points) > 0:
        ax1.scatter(exit_points.index, exit_points['Close'], color='magenta', marker='o', s=8, label='Exit Signal', zorder=5, alpha=0.8)
    
    # Formatting for main chart
    ax1.set_title(f'Gaussian Channel Trading Signals{title_suffix}\n'
                  f'Parameters: {gaussian_filter.poles} poles, {gaussian_filter.period} period, '
                  f'{gaussian_filter.multiplier} multiplier', 
                  fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('Price (USD)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Log scale for better visualization
    
    # Plot 2: Signal indicators
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    
    # Plot entry and exit signals as binary
    ax2.plot(df_with_signals.index, df_with_signals['entry_signal'].astype(int), label='Entry Signal', color='green', linewidth=2)
    ax2.plot(df_with_signals.index, df_with_signals['exit_signal'].astype(int), label='Exit Signal', color='red', linewidth=2)
    ax2.set_ylabel('Signals', fontsize=12)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.1, 1.1)
    
    # Plot 3: Channel conditions
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    
    # Plot channel conditions
    ax3.plot(df_with_signals.index, df_with_signals['green_channel'].astype(int), label='Green Channel (Confirmed)', color='green', alpha=0.7, linewidth=1)
    ax3.plot(df_with_signals.index, df_with_signals['channel_bullish'].astype(int), label='Channel Bullish (Current)', color='blue', alpha=0.7, linewidth=1)
    ax3.set_ylabel('Channel Conditions', fontsize=12)
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-0.1, 1.1)
    
    # Plot 4: True Range
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.plot(df_with_signals.index, df_with_signals['true_range'], label='True Range', color='green', alpha=0.7, linewidth=1)
    ax4.set_ylabel('True Range', fontsize=12)
    ax4.legend(loc='upper left', fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Volume
    ax5 = fig.add_subplot(gs[4], sharex=ax1)
    ax5.plot(df_with_signals.index, df_with_signals['Volume'], label='Volume', color='purple', alpha=0.7, linewidth=1)
    ax5.set_ylabel('Volume', fontsize=12)
    ax5.set_xlabel('Date', fontsize=12)
    ax5.legend(loc='upper left', fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    # Format x-axis dates for all subplots
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Add statistics box
    signal_summary = {
        'total_bars': len(df_with_signals),
        'entries': df_with_signals['entry_signal'].sum(),
        'exits': df_with_signals['exit_signal'].sum(),
        'entry_rate': df_with_signals['entry_signal'].sum() / len(df_with_signals) * 100
    }
    
    stats_text = f"""
Signal Statistics:
• Total Bars: {signal_summary['total_bars']:,}
• Entry Signals: {signal_summary['entries']:,}
• Exit Signals: {signal_summary['exits']:,}
• Entry Rate: {signal_summary['entry_rate']:.2f}%
• Green Channel: {df_with_signals['green_channel'].sum():,} bars
• Bullish Channel: {df_with_signals['channel_bullish'].sum():,} bars
    """
    
    # Add text box with statistics
    ax1.text(0.02, 0.98, stats_text.strip(), transform=ax1.transAxes, 
            verticalalignment='top', fontsize=9, fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_plot:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gaussian_signals_{timestamp}.png"
        save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Signal visualization saved to: {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    
    plt.close()
    
    return fig


def create_comprehensive_visualization(df, filt_result, hband_result, lband_result, gaussian_filter):
    """Create a comprehensive visualization of the entire dataset"""
    
    print("🎨 Creating comprehensive visualization...")
    
    # Set up the plot with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Create grid layout
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.1)
    
    # Plot 1: Main price chart with Gaussian filter
    ax1 = fig.add_subplot(gs[0])
    
    # Get valid data range
    valid_mask = ~filt_result.isna()
    
    # Plot price data
    ax1.plot(df.index, df['Close'], label='BTC Close Price', color='black', alpha=0.8, linewidth=1.5)
    ax1.plot(df.index, df['hlc3'], label='HLC3', color='gray', alpha=0.6, linewidth=1)
    
    # Plot Gaussian filter and bands
    ax1.plot(filt_result.index, filt_result, label='Gaussian Filter', color='blue', linewidth=2.5)
    ax1.plot(hband_result.index, hband_result, label='Upper Band', color='red', alpha=0.7, linewidth=1.5)
    ax1.plot(lband_result.index, lband_result, label='Lower Band', color='red', alpha=0.7, linewidth=1.5)
    
    # Fill the band area
    ax1.fill_between(df.index, hband_result, lband_result, alpha=0.1, color='red', label='Channel')
    
    # Add price annotations for key levels
    if valid_mask.any():
        current_price = df['Close'].iloc[-1]
        current_filter = filt_result.iloc[-1]
        current_upper = hband_result.iloc[-1]
        current_lower = lband_result.iloc[-1]
        
        # Add current values as text
        ax1.axhline(y=current_price, color='black', linestyle='--', alpha=0.5)
        ax1.axhline(y=current_filter, color='blue', linestyle='--', alpha=0.5)
        ax1.axhline(y=current_upper, color='red', linestyle='--', alpha=0.5)
        ax1.axhline(y=current_lower, color='red', linestyle='--', alpha=0.5)
        
        # Add text annotations
        ax1.text(df.index[-1], current_price, f' Price: ${current_price:,.0f}', 
                verticalalignment='bottom', fontsize=10, fontweight='bold')
        ax1.text(df.index[-1], current_filter, f' Filter: ${current_filter:,.0f}', 
                verticalalignment='bottom', fontsize=10, color='blue')
        ax1.text(df.index[-1], current_upper, f' Upper: ${current_upper:,.0f}', 
                verticalalignment='bottom', fontsize=10, color='red')
        ax1.text(df.index[-1], current_lower, f' Lower: ${current_lower:,.0f}', 
                verticalalignment='bottom', fontsize=10, color='red')
    
    # Formatting for main chart
    ax1.set_title(f'Bitcoin Gaussian Channel Filter Analysis (2018-2025)\n'
                  f'Parameters: {gaussian_filter.poles} poles, {gaussian_filter.period} period, '
                  f'{gaussian_filter.multiplier} multiplier', 
                  fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('Price (USD)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Log scale for better visualization
    
    # Plot 2: True Range
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(df.index, df['true_range'], label='True Range', color='green', alpha=0.7, linewidth=1)
    ax2.set_ylabel('True Range', fontsize=12)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Volume
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(df.index, df['Volume'], label='Volume', color='purple', alpha=0.7, linewidth=1)
    ax3.set_ylabel('Volume', fontsize=12)
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Price change percentage
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    price_change = df['Close'].pct_change() * 100
    ax4.plot(df.index, price_change, label='Daily % Change', color='orange', alpha=0.7, linewidth=1)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.set_ylabel('Daily % Change', fontsize=12)
    ax4.set_xlabel('Date', fontsize=12)
    ax4.legend(loc='upper left', fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # Format x-axis dates for all subplots
    for ax in [ax1, ax2, ax3, ax4]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Add statistics box
    if valid_mask.any():
        stats_text = f"""
Statistics (Valid Filter Period):
• Data Points: {valid_mask.sum():,} / {len(df):,}
• Price Range: ${df['Close'][valid_mask].min():,.0f} - ${df['Close'][valid_mask].max():,.0f}
• Current Price: ${df['Close'].iloc[-1]:,.0f}
• Filter Value: ${filt_result.iloc[-1]:,.0f}
• Channel Width: ${hband_result.iloc[-1] - lband_result.iloc[-1]:,.0f}
• Channel %: {((hband_result.iloc[-1] - lband_result.iloc[-1]) / filt_result.iloc[-1] * 100):.1f}%
        """
        
        # Add text box with statistics
        ax1.text(0.02, 0.98, stats_text.strip(), transform=ax1.transAxes, 
                verticalalignment='top', fontsize=9, fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    return fig


def create_zoom_visualizations(df, filt_result, hband_result, lband_result, gaussian_filter):
    """Create zoomed-in visualizations for different time periods"""
    
    print("🔍 Creating zoomed visualizations...")
    
    # Define time periods to zoom into
    zoom_periods = [
        ('2018-2019', '2018-01-01', '2019-12-31'),
        ('2020-2021', '2020-01-01', '2021-12-31'),
        ('2022-2023', '2022-01-01', '2023-12-31'),
        ('2024-2025', '2024-01-01', '2025-12-31'),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes.flatten()
    
    for i, (period_name, start_date, end_date) in enumerate(zoom_periods):
        ax = axes[i]
        
        # Filter data for the period
        mask = (df.index >= start_date) & (df.index <= end_date)
        period_df = df[mask]
        period_filt = filt_result[mask]
        period_hband = hband_result[mask]
        period_lband = lband_result[mask]
        
        if len(period_df) == 0:
            ax.text(0.5, 0.5, f'No data for {period_name}', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'{period_name} (No Data)', fontsize=12)
            continue
        
        # Plot the period
        ax.plot(period_df.index, period_df['Close'], label='BTC Price', color='black', alpha=0.8, linewidth=1.5)
        ax.plot(period_df.index, period_df['hlc3'], label='HLC3', color='gray', alpha=0.6, linewidth=1)
        
        # Plot Gaussian filter and bands
        ax.plot(period_filt.index, period_filt, label='Gaussian Filter', color='blue', linewidth=2)
        ax.plot(period_hband.index, period_hband, label='Upper Band', color='red', alpha=0.7, linewidth=1.5)
        ax.plot(period_lband.index, period_lband, label='Lower Band', color='red', alpha=0.7, linewidth=1.5)
        
        # Fill the band area
        ax.fill_between(period_df.index, period_hband, period_lband, alpha=0.1, color='red')
        
        # Formatting
        ax.set_title(f'{period_name}', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.suptitle('Gaussian Channel Filter - Zoomed Periods', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig


def compare_parameters():
    """Compare different Gaussian filter parameters"""
    
    print("📊 Loading BTC data for parameter comparison...")
    df = load_btc_data()
    # Use last 500 days for clearer comparison
    df = df.tail(500)
    
    # Test different parameter combinations
    param_sets = [
            (4, 144, 1.414, "4 poles, 144 period"),
    (6, 144, 1.414, "6 poles, 144 period (original)"),
    (8, 144, 1.414, "8 poles, 144 period"),
        (6, 72, 1.414, "6 poles, 72 period"),
        (6, 288, 1.414, "6 poles, 288 period"),
    ]
    
    fig, axes = plt.subplots(len(param_sets), 1, figsize=(15, 4*len(param_sets)))
    if len(param_sets) == 1:
        axes = [axes]
    
    for i, (poles, period, multiplier, title) in enumerate(param_sets):
        ax = axes[i]
        
        # Initialize filter
        gaussian_filter = GaussianChannelFilter(poles=poles, period=period, multiplier=multiplier)
        
        # Apply filter
        filt_result, hband_result, lband_result = gaussian_filter.apply_filter(
            df['hlc3'], 
            df['true_range']
        )
        
        # Plot
        ax.plot(df.index, df['hlc3'], label='HLC3 Price', color='black', alpha=0.7, linewidth=1)
        ax.plot(filt_result.index, filt_result, label='Gaussian Filter', color='blue', linewidth=2)
        ax.plot(hband_result.index, hband_result, label='Upper Band', color='red', alpha=0.7, linewidth=1.5)
        ax.plot(lband_result.index, lband_result, label='Lower Band', color='red', alpha=0.7, linewidth=1.5)
        ax.fill_between(df.index, hband_result, lband_result, alpha=0.1, color='red')
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    # Save comparison
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"gaussian_filter_comparison_{timestamp}.png"
    save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Parameter comparison saved to: {save_path}")
    
    plt.show()
    plt.close()


def run_backtest_with_debug():
    """Run comprehensive backtest with position management"""
    
    print("🔍 Starting Comprehensive Gaussian Channel Backtest")
    print("=" * 60)
    
    # Data path
    data_path = "data/btc_1d_data_2018_to_2025.csv"
    print(f"📊 Loading data from: {data_path}")
    
    # Run comprehensive backtest
    try:
        results = run_comprehensive_backtest(
            data_path=data_path,
            initial_cash=10000,
            commission=0.001,
            slippage_perc=0.01,
            debug_mode=True,  # Enable debug mode
            save_results=True
        )
        
        if results and results['success']:
            print("=" * 60)
            print("✅ Comprehensive backtest completed successfully!")
            
            # Ask if user wants to see additional analysis
            analysis_choice = input("📊 Would you like to see detailed trade analysis? (y/n): ").lower().strip()
            if analysis_choice in ['y', 'yes']:
                cerebro = results['cerebro']
                strategy = None
                try:
                    if hasattr(cerebro, 'runstrats') and cerebro.runstrats:
                        strategy_list = cerebro.runstrats[0]
                        if isinstance(strategy_list, list) and len(strategy_list) > 0:
                            strategy = strategy_list[0]
                except Exception as e:
                    print(f"Warning: Could not retrieve strategy instance: {e}")
                
                if strategy and hasattr(strategy, 'position_manager'):
                    print("\n📋 DETAILED TRADE ANALYSIS:")
                    print("=" * 60)
                    
                    # Show recent trades
                    recent_trades = strategy.position_manager.trades[-10:]  # Last 10 trades
                    if recent_trades:
                        print("Recent Trades:")
                        for i, trade in enumerate(recent_trades, 1):
                            status = trade.get('status', 'UNKNOWN')
                            if status == 'COMPLETED':
                                print(f"  {i}. {trade['entry_date']} → {trade['exit_date']} | "
                                      f"${trade['entry_price']:.2f} → ${trade['exit_price']:.2f} | "
                                      f"{trade['pnl_pct']:+.2f}% | {trade['duration_days']} days")
                            else:
                                print(f"  {i}. {trade['entry_date']} → OPEN | "
                                      f"${trade['entry_price']:.2f} → CURRENT | "
                                      f"OPEN | {trade.get('duration_days', 0)} days")
        else:
            print("❌ Backtest failed!")
            if results:
                print(f"Error: {results.get('error', 'Unknown error')}")
        
    except Exception as e:
        print(f"❌ Error during backtest: {e}")
        import traceback
        traceback.print_exc()


def simulate_trading_with_position_management(df_with_signals, gaussian_filter):
    """
    Simulate trading with position management logic
    Returns trade log and modified dataframe with actual trades
    """
    
    print("🎯 Simulating trading with position management...")
    
    # Initialize trading state
    in_position = False
    entry_price = None
    entry_date = None
    position_size = 0.0
    
    # Trade log
    trades = []
    
    # Copy dataframe to avoid modifying original
    df_trades = df_with_signals.copy()
    
    # Add trade columns
    df_trades['in_position'] = False
    df_trades['actual_entry'] = False
    df_trades['actual_exit'] = False
    df_trades['entry_price'] = np.nan
    df_trades['current_pnl'] = 0.0
    
    # Simulate trading bar by bar
    for i in range(len(df_trades)):
        current_date = df_trades.index[i]
        current_price = df_trades['Close'].iloc[i]
        entry_signal = df_trades['entry_signal'].iloc[i]
        exit_signal = df_trades['exit_signal'].iloc[i]
        
        # === ENTRY LOGIC ===
        if entry_signal and not in_position:
            # Execute entry
            in_position = True
            entry_price = current_price
            entry_date = current_date
            position_size = 1.0  # 100% position size
            
            # Record trade
            trade = {
                'entry_date': entry_date,
                'entry_price': entry_price,
                'entry_signal': True,
                'exit_date': None,
                'exit_price': None,
                'exit_signal': False,
                'pnl': 0.0,
                'pnl_pct': 0.0,
                'duration_days': 0
            }
            trades.append(trade)
            
            # Update dataframe
            df_trades.loc[current_date, 'in_position'] = True
            df_trades.loc[current_date, 'actual_entry'] = True
            df_trades.loc[current_date, 'entry_price'] = entry_price
            
            print(f"📈 ENTRY: {current_date.strftime('%Y-%m-%d')} at ${entry_price:,.2f}")
        
        # === EXIT LOGIC ===
        elif exit_signal and in_position:
            # Execute exit
            exit_price = current_price
            pnl = (exit_price - entry_price) / entry_price  # Simple PnL calculation
            duration_days = (current_date - entry_date).days
            
            # Update last trade
            if trades:
                trades[-1].update({
                    'exit_date': current_date,
                    'exit_price': exit_price,
                    'exit_signal': True,
                    'pnl': exit_price - entry_price,
                    'pnl_pct': pnl * 100,
                    'duration_days': duration_days
                })
            
            # Reset position
            in_position = False
            entry_price = None
            entry_date = None
            position_size = 0.0
            
            # Update dataframe
            df_trades.loc[current_date, 'in_position'] = False
            df_trades.loc[current_date, 'actual_exit'] = True
            
            print(f"📉 EXIT: {current_date.strftime('%Y-%m-%d')} at ${exit_price:,.2f} | PnL: {pnl*100:+.2f}% | Duration: {duration_days} days")
        
        # === UPDATE CURRENT PnL ===
        if in_position and entry_price:
            current_pnl = (current_price - entry_price) / entry_price
            df_trades.loc[current_date, 'current_pnl'] = current_pnl
    
    # Handle open position at end
    if in_position:
        print(f"⚠️  OPEN POSITION: Still in position at end of data")
        if trades:
            trades[-1]['status'] = 'OPEN'
    
    # Calculate trade statistics
    completed_trades = [t for t in trades if t.get('exit_date') is not None]
    open_trades = [t for t in trades if t.get('status') == 'OPEN']
    
    if completed_trades:
        total_trades = len(completed_trades)
        winning_trades = len([t for t in completed_trades if t['pnl'] > 0])
        losing_trades = len([t for t in completed_trades if t['pnl'] < 0])
        win_rate = winning_trades / total_trades * 100
        
        total_pnl = sum(t['pnl'] for t in completed_trades)
        avg_pnl = total_pnl / total_trades
        avg_duration = sum(t['duration_days'] for t in completed_trades) / total_trades
        
        print(f"\n📊 TRADE STATISTICS:")
        print(f"   Total Trades: {total_trades}")
        print(f"   Winning Trades: {winning_trades}")
        print(f"   Losing Trades: {losing_trades}")
        print(f"   Win Rate: {win_rate:.1f}%")
        print(f"   Total PnL: {total_pnl*100:+.2f}%")
        print(f"   Average PnL per Trade: {avg_pnl*100:+.2f}%")
        print(f"   Average Duration: {avg_duration:.1f} days")
        print(f"   Open Positions: {len(open_trades)}")
    
    return df_trades, trades


def visualize_actual_trades(df_trades, trades, gaussian_filter, save_plot=True, show_plot=True, title_suffix=""):
    """Create comprehensive visualization of actual trades with position management"""
    
    print("🎨 Creating actual trades visualization...")
    
    # Set up the plot with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Create grid layout
    gs = fig.add_gridspec(5, 1, height_ratios=[3, 1, 1, 1, 1], hspace=0.1)
    
    # Plot 1: Main price chart with actual trades
    ax1 = fig.add_subplot(gs[0])
    
    # Plot price data
    ax1.plot(df_trades.index, df_trades['Close'], label='BTC Close Price', color='black', alpha=0.8, linewidth=1.5)
    ax1.plot(df_trades.index, df_trades['hlc3'], label='HLC3', color='gray', alpha=0.6, linewidth=1)
    
    # Plot Gaussian filter and bands (confirmed data for non-repainting)
    ax1.plot(df_trades.index, df_trades['filt_confirmed'], label='Gaussian Filter (Confirmed)', color='blue', linewidth=2)
    ax1.plot(df_trades.index, df_trades['hband_confirmed'], label='Upper Band (Confirmed)', color='red', alpha=0.7, linewidth=1.5)
    ax1.plot(df_trades.index, df_trades['lband_confirmed'], label='Lower Band (Confirmed)', color='red', alpha=0.7, linewidth=1.5)
    
    # Fill the band area
    ax1.fill_between(df_trades.index, df_trades['hband_confirmed'], df_trades['lband_confirmed'], alpha=0.1, color='red', label='Channel')
    
    # Plot ACTUAL entry points (not raw signals)
    actual_entries = df_trades[df_trades['actual_entry']]
    if len(actual_entries) > 0:
        ax1.scatter(actual_entries.index, actual_entries['Close'], color='green', marker='^', s=100, label='Actual Entry', zorder=5, alpha=0.8)
    
    # Plot ACTUAL exit points (not raw signals)
    actual_exits = df_trades[df_trades['actual_exit']]
    if len(actual_exits) > 0:
        ax1.scatter(actual_exits.index, actual_exits['Close'], color='red', marker='v', s=100, label='Actual Exit', zorder=5, alpha=0.8)
    
    # Connect entry to exit points for each trade
    completed_trades = [t for t in trades if t.get('exit_date') is not None]
    for trade in completed_trades:
        entry_date = trade['entry_date']
        exit_date = trade['exit_date']
        entry_price = trade['entry_price']
        exit_price = trade['exit_price']
        
        # Plot trade line
        color = 'green' if trade['pnl'] > 0 else 'red'
        alpha = 0.6 if trade['pnl'] > 0 else 0.4
        ax1.plot([entry_date, exit_date], [entry_price, exit_price], color=color, alpha=alpha, linewidth=2)
        
        # Add PnL annotation
        mid_date = entry_date + (exit_date - entry_date) / 2
        mid_price = (entry_price + exit_price) / 2
        pnl_text = f"{trade['pnl_pct']:+.1f}%"
        ax1.annotate(pnl_text, (mid_date, mid_price), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, color=color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Formatting for main chart
    ax1.set_title(f'Gaussian Channel Actual Trades{title_suffix}\n'
                  f'Parameters: {gaussian_filter.poles} poles, {gaussian_filter.period} period, '
                  f'{gaussian_filter.multiplier} multiplier', 
                  fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('Price (USD)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Log scale for better visualization
    
    # Plot 2: Position status
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    
    # Plot position status
    ax2.plot(df_trades.index, df_trades['in_position'].astype(int), label='In Position', color='blue', linewidth=2)
    ax2.set_ylabel('Position Status', fontsize=12)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.1, 1.1)
    
    # Plot 3: Current PnL
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    
    # Plot current PnL
    ax3.plot(df_trades.index, df_trades['current_pnl'] * 100, label='Current PnL %', color='purple', linewidth=1.5)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.set_ylabel('Current PnL (%)', fontsize=12)
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Channel conditions
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    
    # Plot channel conditions
    ax4.plot(df_trades.index, df_trades['green_channel'].astype(int), label='Green Channel (Confirmed)', color='green', alpha=0.7, linewidth=1)
    ax4.plot(df_trades.index, df_trades['channel_bullish'].astype(int), label='Channel Bullish (Current)', color='blue', alpha=0.7, linewidth=1)
    ax4.set_ylabel('Channel Conditions', fontsize=12)
    ax4.legend(loc='upper left', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-0.1, 1.1)
    
    # Plot 5: Volume
    ax5 = fig.add_subplot(gs[4], sharex=ax1)
    ax5.plot(df_trades.index, df_trades['Volume'], label='Volume', color='purple', alpha=0.7, linewidth=1)
    ax5.set_ylabel('Volume', fontsize=12)
    ax5.set_xlabel('Date', fontsize=12)
    ax5.legend(loc='upper left', fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    # Format x-axis dates for all subplots
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Calculate and display trade statistics
    completed_trades = [t for t in trades if t.get('exit_date') is not None]
    open_trades = [t for t in trades if t.get('status') == 'OPEN']
    
    if completed_trades:
        total_trades = len(completed_trades)
        winning_trades = len([t for t in completed_trades if t['pnl'] > 0])
        losing_trades = len([t for t in completed_trades if t['pnl'] < 0])
        win_rate = winning_trades / total_trades * 100
        
        total_pnl = sum(t['pnl'] for t in completed_trades)
        avg_pnl = total_pnl / total_trades
        avg_duration = sum(t['duration_days'] for t in completed_trades) / total_trades
        
        stats_text = f"""
Trade Statistics:
• Total Trades: {total_trades}
• Winning Trades: {winning_trades}
• Losing Trades: {losing_trades}
• Win Rate: {win_rate:.1f}%
• Total PnL: {total_pnl*100:+.2f}%
• Avg PnL per Trade: {avg_pnl*100:+.2f}%
• Avg Duration: {avg_duration:.1f} days
• Open Positions: {len(open_trades)}
        """
        
        # Add text box with statistics
        ax1.text(0.02, 0.98, stats_text.strip(), transform=ax1.transAxes, 
                verticalalignment='top', fontsize=9, fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_plot:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gaussian_actual_trades_{timestamp}.png"
        save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Actual trades visualization saved to: {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    
    plt.close()
    
    return fig


def create_trade_log_report(trades, save_report=True):
    """Create a detailed trade log report"""
    
    print("📋 Creating trade log report...")
    
    if not trades:
        print("No trades to report")
        return
    
    # Create trade log DataFrame
    trade_data = []
    for i, trade in enumerate(trades):
        trade_data.append({
            'Trade #': i + 1,
            'Entry Date': trade['entry_date'].strftime('%Y-%m-%d'),
            'Entry Price': f"${trade['entry_price']:,.2f}",
            'Exit Date': trade['exit_date'].strftime('%Y-%m-%d') if trade.get('exit_date') else 'OPEN',
            'Exit Price': f"${trade['exit_price']:,.2f}" if trade.get('exit_price') else 'N/A',
            'Duration (Days)': trade.get('duration_days', 0),
            'PnL ($)': f"${trade.get('pnl', 0):,.2f}",
            'PnL (%)': f"{trade.get('pnl_pct', 0):+.2f}%",
            'Status': 'COMPLETED' if trade.get('exit_date') else 'OPEN'
        })
    
    trade_df = pd.DataFrame(trade_data)
    
    # Print trade log
    print("\n" + "="*100)
    print("📊 DETAILED TRADE LOG")
    print("="*100)
    print(trade_df.to_string(index=False))
    print("="*100)
    
    # Save trade log if requested
    if save_report:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trade_log_{timestamp}.csv"
        save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save as CSV
        trade_df.to_csv(save_path, index=False)
        print(f"📄 Trade log saved to: {save_path}")
    
    return trade_df


def save_visualizations(fig_main, fig_zoom, gaussian_filter):
    """Save the visualizations to files"""
    
    # Create results directory
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save main visualization
    main_filename = f"btc_gaussian_full_dataset_{timestamp}.png"
    main_path = os.path.join(results_dir, main_filename)
    fig_main.savefig(main_path, dpi=300, bbox_inches='tight')
    print(f"📊 Main visualization saved to: {main_path}")
    
    # Save zoom visualization
    zoom_filename = f"btc_gaussian_zoom_periods_{timestamp}.png"
    zoom_path = os.path.join(results_dir, zoom_filename)
    fig_zoom.savefig(zoom_path, dpi=300, bbox_inches='tight')
    print(f"📊 Zoom visualization saved to: {zoom_path}")
    
    return main_path, zoom_path


def main():
    """Main function to run comprehensive testing"""
    
    print("🚀 Comprehensive Gaussian Channel Filter Testing")
    print("=" * 70)
    
    # Test options
    print("\n📋 Available Tests:")
    print("1. Gaussian Filter Test with BTC Data (Channel Only)")
    print("2. Parameter Comparison (Different filter settings)")
    print("3. Signal Generation & Visualization (Channel + Signals)")
    print("4. Position Management & Actual Trades Analysis")
    print("5. Backtest with Debug Mode")
    print("6. Run All Tests (Comprehensive Analysis)")
    
    choice = input("\n🎯 Select test option (1-6): ").strip()
    
    try:
        if choice == "1":
            # Test 1: Gaussian Filter Test with BTC Data (Channel Only)
            print("\n1️⃣ Loading BTC data...")
            df = load_btc_data()
            
            print("\n2️⃣ Testing Gaussian filter...")
            df, filt_result, hband_result, lband_result, gaussian_filter = test_gaussian_filter(df)
            
            print("\n3️⃣ Creating channel visualization...")
            visualize_gaussian_filter(df, filt_result, hband_result, lband_result, gaussian_filter, 
                                    save_plot=True, show_plot=True, title_suffix=" (BTC Data)")
            
            print("\n✅ Gaussian filter test completed!")
            
        elif choice == "2":
            # Test 2: Parameter Comparison
            print("\n📊 Parameter Comparison Test")
            print("This test compares different Gaussian filter settings:")
            print("• 4 poles vs 6 poles vs 8 poles (smoothing sensitivity)")
            print("• 72 period vs 144 period vs 288 period (sampling period)")
            print("• Shows how different parameters affect the filter behavior")
            print("\n1️⃣ Creating parameter comparison chart...")
            compare_parameters()
            
        elif choice == "3":
            # Test 3: Signal Generation & Visualization (Channel + Signals)
            print("\n1️⃣ Loading BTC data...")
            df = load_btc_data()
            
            print("\n2️⃣ Testing Gaussian filter...")
            df, filt_result, hband_result, lband_result, gaussian_filter = test_gaussian_filter(df)
            
            print("\n3️⃣ Testing signal generation...")
            df_with_signals, signal_generator = test_signal_generation(df, gaussian_filter)
            
            print("\n4️⃣ Creating channel + signal visualization...")
            visualize_signals(df_with_signals, gaussian_filter, save_plot=True, show_plot=True, title_suffix=" (Channel + Signals)")
            
            print("\n✅ Signal generation & visualization completed!")
            
        elif choice == "4":
            # Test 4: Position Management & Actual Trades Analysis
            print("\n1️⃣ Loading BTC data...")
            df = load_btc_data()
            
            print("\n2️⃣ Testing Gaussian filter...")
            df, filt_result, hband_result, lband_result, gaussian_filter = test_gaussian_filter(df)
            
            print("\n3️⃣ Testing signal generation...")
            df_with_signals, signal_generator = test_signal_generation(df, gaussian_filter)
            
            print("\n4️⃣ Simulating trading with position management...")
            df_trades, trades = simulate_trading_with_position_management(df_with_signals, gaussian_filter)
            
            print("\n5️⃣ Creating actual trades visualization...")
            visualize_actual_trades(df_trades, trades, gaussian_filter, save_plot=True, show_plot=True, title_suffix=" (Position Management)")
            
            print("\n6️⃣ Creating trade log report...")
            create_trade_log_report(trades, save_report=True)
            
            print("\n✅ Position management analysis completed!")
            
        elif choice == "5":
            # Test 5: Backtest with debug mode
            run_backtest_with_debug()
            
        elif choice == "6":
            # Test 6: Run All Tests (Comprehensive Analysis)
            print("\n🔄 Running comprehensive analysis...")
            
            # Load BTC data
            print("\n1️⃣ Loading BTC data...")
            df = load_btc_data()
            
            # Test Gaussian filter
            print("\n2️⃣ Testing Gaussian filter...")
            df, filt_result, hband_result, lband_result, gaussian_filter = test_gaussian_filter(df)
            
            # Test signal generation
            print("\n3️⃣ Testing signal generation...")
            df_with_signals, signal_generator = test_signal_generation(df, gaussian_filter)
            
            # Create comprehensive visualization with signals
            print("\n4️⃣ Creating comprehensive visualization with signals...")
            visualize_signals(df_with_signals, gaussian_filter, save_plot=True, show_plot=True, title_suffix=" (Comprehensive Analysis)")
            
            # Position management analysis
            print("\n5️⃣ Simulating trading with position management...")
            df_trades, trades = simulate_trading_with_position_management(df_with_signals, gaussian_filter)
            
            print("\n6️⃣ Creating actual trades visualization...")
            visualize_actual_trades(df_trades, trades, gaussian_filter, save_plot=True, show_plot=True, title_suffix=" (Comprehensive Analysis)")
            
            print("\n7️⃣ Creating trade log report...")
            create_trade_log_report(trades, save_report=True)
            
            # Run backtest
            print("\n8️⃣ Running backtest...")
            run_backtest_with_debug()
            
            print("\n✅ Comprehensive analysis completed!")
            
        else:
            print("❌ Invalid choice. Please select 1-6.")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 