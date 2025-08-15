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
from strategy.backtest import run_backtrader_backtest, analyze_backtrader_results, plot_backtrader_results


def create_test_data(days=200, base_price=45000, volatility=0.02):
    """Create realistic test data for Gaussian filter testing"""
    
    dates = pd.date_range('2024-01-01', periods=days, freq='D')
    np.random.seed(42)  # For reproducible results
    
    # Generate realistic price data
    prices = []
    for i in range(days):
        # Random walk with some trend
        if i == 0:
            price = base_price
        else:
            change = np.random.normal(0, volatility)  # Daily volatility
            price = prices[-1] * (1 + change)
        prices.append(price)
    
    # Create OHLCV data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Simple OHLC from close price
        high = close * (1 + abs(np.random.normal(0, 0.01)))
        low = close * (1 - abs(np.random.normal(0, 0.01)))
        open_price = close * (1 + np.random.normal(0, 0.005))
        volume = np.random.uniform(1000, 10000)
        
        data.append({
            'Date': date,
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    
    # Calculate HLC3 and True Range (same as in strategy)
    df['hlc3'] = (df['High'] + df['Low'] + df['Close']) / 3
    
    df['tr1'] = df['High'] - df['Low']
    df['tr2'] = abs(df['High'] - df['Close'].shift(1))
    df['tr3'] = abs(df['Low'] - df['Close'].shift(1))
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    return df


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


def test_gaussian_filter(df=None, poles=6, period=144, multiplier=1.414):
    """Test that Gaussian filter works with live trading parameters"""
    
    if df is None:
        df = create_test_data()
        print("🧪 Testing with synthetic data...")
    else:
        print("🧪 Testing with real BTC data...")
    
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
    
    df = create_test_data(days=300)
    
    # Test different parameter combinations
    param_sets = [
        (4, 144, 1.414, "4 poles, 144 period"),
        (6, 144, 1.414, "6 poles, 144 period (current)"),
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
    """Run backtest with debug mode enabled"""
    
    print("🔍 Starting Gaussian Channel Backtest with Debug Mode")
    print("=" * 60)
    
    # Load data
    data_path = "data/btc_1d_data_2018_to_2025.csv"
    print(f"📊 Loading data from: {data_path}")
    
    try:
        data = pd.read_csv(data_path)
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        print(f"✅ Loaded {len(data)} bars from {data.index[0].date()} to {data.index[-1].date()}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Strategy parameters (matching live trading)
    strategy_params = {
        'poles': 6,
        'period': 144,
        'multiplier': 1.414,
        'position_size_pct': 1.0,
        'atr_period': 14
    }
    
    print(f"⚙️ Strategy parameters: {strategy_params}")
    print("=" * 60)
    
    # Run backtest with debug mode
    try:
        cerebro = run_backtrader_backtest(
            data=data,
            strategy_class=None,  # Will use default GaussianChannelStrategy
            strategy_params=strategy_params,
            initial_cash=10000,
            commission=0.001,
            slippage_perc=0.01,
            debug_mode=True  # Enable debug mode
        )
        
        print("=" * 60)
        print("📈 Analyzing results...")
        
        # Analyze results
        results = analyze_backtrader_results(cerebro)
        
        print(f"💰 Initial Cash: ${results['initial_cash']:,.2f}")
        print(f"💰 Final Value: ${results['final_value']:,.2f}")
        print(f"📊 Total Return: {results['total_return_pct']:.2f}%")
        
        # Get strategy instance for additional info
        strategy = results.get('strategy')
        if strategy:
            print(f"🔄 Total Entries: {strategy.entry_count}")
            print(f"📊 Total Bars Processed: {len(strategy.hlc3_history) if hasattr(strategy, 'hlc3_history') else 'Unknown'}")
        
        print("=" * 60)
        print("✅ Backtest completed successfully!")
        
        # Ask if user wants to plot
        plot_choice = input("📊 Would you like to see the plot? (y/n): ").lower().strip()
        if plot_choice in ['y', 'yes']:
            plot_backtrader_results(cerebro)
        
    except Exception as e:
        print(f"❌ Error during backtest: {e}")
        import traceback
        traceback.print_exc()


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
    print("1. Basic Gaussian Filter Test (Synthetic Data)")
    print("2. Gaussian Filter Test with Real BTC Data")
    print("3. Full BTC Dataset Visualization")
    print("4. Parameter Comparison")
    print("5. Backtest with Debug Mode")
    print("6. Run All Tests")
    
    choice = input("\n🎯 Select test option (1-6): ").strip()
    
    try:
        if choice == "1":
            # Test 1: Basic functionality with synthetic data
            print("\n1️⃣ Testing with synthetic data...")
            df, filt_result, hband_result, lband_result, gaussian_filter = test_gaussian_filter()
            
            # Visualize synthetic data results
            print("\n2️⃣ Creating visualization for synthetic data...")
            visualize_gaussian_filter(df, filt_result, hband_result, lband_result, gaussian_filter, 
                                    save_plot=True, show_plot=True, title_suffix=" (Synthetic Data)")
            
        elif choice == "2":
            # Test 2: Real BTC data
            print("\n1️⃣ Testing with real BTC data...")
            df = load_btc_data()
            df, filt_result, hband_result, lband_result, gaussian_filter = test_gaussian_filter(df)
            
            # Visualize real data results
            print("\n2️⃣ Creating visualization for real BTC data...")
            visualize_gaussian_filter(df, filt_result, hband_result, lband_result, gaussian_filter, 
                                    save_plot=True, show_plot=True, title_suffix=" (Real BTC Data)")
            
        elif choice == "3":
            # Test 3: Full BTC dataset visualization
            print("\n1️⃣ Loading full BTC dataset...")
            df = load_btc_data()
            
            print("\n2️⃣ Applying Gaussian filter...")
            df, filt_result, hband_result, lband_result, gaussian_filter = test_gaussian_filter(df)
            
            # Create visualizations
            print("\n3️⃣ Creating visualizations...")
            
            # Main comprehensive visualization
            fig_main = create_comprehensive_visualization(df, filt_result, hband_result, lband_result, gaussian_filter)
            
            # Zoomed period visualizations
            fig_zoom = create_zoom_visualizations(df, filt_result, hband_result, lband_result, gaussian_filter)
            
            # Save visualizations
            print("\n4️⃣ Saving visualizations...")
            main_path, zoom_path = save_visualizations(fig_main, fig_zoom, gaussian_filter)
            
            # Show plots
            print("\n5️⃣ Displaying visualizations...")
            plt.show()
            
            print("\n✅ Full dataset visualization complete!")
            print(f"📁 Check the 'results' folder for saved images:")
            print(f"   • {os.path.basename(main_path)}")
            print(f"   • {os.path.basename(zoom_path)}")
            
        elif choice == "4":
            # Test 4: Parameter comparison
            print("\n1️⃣ Creating parameter comparison chart...")
            compare_parameters()
            
        elif choice == "5":
            # Test 5: Backtest with debug mode
            run_backtest_with_debug()
            
        elif choice == "6":
            # Test 6: Run all tests
            print("\n🔄 Running all tests...")
            
            # Test 1: Synthetic data
            print("\n1️⃣ Testing with synthetic data...")
            df_synth, filt_synth, hband_synth, lband_synth, gaussian_filter_synth = test_gaussian_filter()
            visualize_gaussian_filter(df_synth, filt_synth, hband_synth, lband_synth, gaussian_filter_synth, 
                                    save_plot=True, show_plot=False, title_suffix=" (Synthetic Data)")
            
            # Test 2: Real BTC data
            print("\n2️⃣ Testing with real BTC data...")
            df_real = load_btc_data()
            df_real, filt_real, hband_real, lband_real, gaussian_filter_real = test_gaussian_filter(df_real)
            visualize_gaussian_filter(df_real, filt_real, hband_real, lband_real, gaussian_filter_real, 
                                    save_plot=True, show_plot=False, title_suffix=" (Real BTC Data)")
            
            # Test 3: Full dataset visualization
            print("\n3️⃣ Creating full dataset visualization...")
            fig_main = create_comprehensive_visualization(df_real, filt_real, hband_real, lband_real, gaussian_filter_real)
            fig_zoom = create_zoom_visualizations(df_real, filt_real, hband_real, lband_real, gaussian_filter_real)
            main_path, zoom_path = save_visualizations(fig_main, fig_zoom, gaussian_filter_real)
            
            # Test 4: Parameter comparison
            print("\n4️⃣ Creating parameter comparison...")
            compare_parameters()
            
            # Test 5: Backtest
            print("\n5️⃣ Running backtest...")
            run_backtest_with_debug()
            
            print("\n✅ All tests completed!")
            
        else:
            print("❌ Invalid choice. Please select 1-6.")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 