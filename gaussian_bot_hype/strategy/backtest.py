#!/usr/bin/env python3
"""
Gaussian Channel Strategy - Backtrader Implementation
Clean module for backtrader-based backtesting

This module will contain the backtrader strategy implementation
and related backtesting utilities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Backtrader for professional backtesting
import backtrader as bt

# Import our Gaussian filter from the separate module
from .gaussian_filter import GaussianChannelFilter, calculate_rma


class GaussianChannelStrategy(bt.Strategy):
    """
    Gaussian Channel Strategy implemented as a backtrader strategy
    
    This strategy implements the Gaussian Channel filter with:
    - Long-only trend following (matches live trading exactly)
    - Dynamic exit based on price closing below upper band
    - 6 poles, 144 period, 1.414 multiplier (matches live trading)
    - No pyramiding or ATR spacing (simplified)
    - 100% position size per trade
    """
    
    params = (
        ('poles', 6),                    # Number of poles for Gaussian filter (matches live trading)
        ('period', 144),                 # Sampling period for Gaussian filter
        ('multiplier', 1.414),           # Band width multiplier
        ('position_size_pct', 1.0),      # Position size as percentage of portfolio (100%)
        ('atr_period', 14),              # ATR calculation period (needed for channel calculation)
    )
    
    def __init__(self):
        """Initialize the strategy with indicators"""
        # Initialize Gaussian filter
        self.gaussian_filter = GaussianChannelFilter(
            poles=self.params.poles,
            period=self.params.period,
            multiplier=self.params.multiplier
        )
        
        # Calculate HLC3 (High, Low, Close average)
        self.hlc3 = (self.data.high + self.data.low + self.data.close) / 3
        
        # Calculate True Range
        self.tr1 = self.data.high - self.data.low
        # Use simple arithmetic for absolute values since bt.indicators.Abs doesn't exist
        self.tr2 = self.data.high - self.data.close(-1)  # Previous close
        self.tr3 = self.data.low - self.data.close(-1)   # Previous close
        # We'll calculate the max in the next() method since we need to handle abs() manually
        
        # Strategy state variables (simplified to match live trading)
        self.entry_count = 0
        self.last_entry_price = None
        
        # Store all historical data for Gaussian filter calculation
        self.hlc3_history = []
        self.tr_history = []
        self.close_history = []
        
        # Store calculated values for signal generation
        self.filt_values = []
        self.hband_values = []
        self.lband_values = []
        self.green_channel_values = []
        
        # Store confirmed values for non-repainting
        self.filt_confirmed_values = []
        self.hband_confirmed_values = []
        self.lband_confirmed_values = []
        
    def next(self):
        """
        Main strategy logic - called for each bar
        
        This method implements the EXACT same logic as live trading:
        1. Apply Gaussian filter to current data
        2. Entry: Close > current upper band
        3. Exit: Close < current upper band
        4. No green channel requirement (simplified like live trading)
        """
        # Get current bar data
        current_hlc3 = self.hlc3[0]
        current_close = self.data.close[0]
        
        # Calculate True Range for current bar
        tr1 = self.tr1[0]  # High - Low
        tr2 = abs(self.tr2[0])  # abs(High - Previous Close)
        tr3 = abs(self.tr3[0])  # abs(Low - Previous Close)
        current_tr = max(tr1, tr2, tr3)
        
        # Accumulate historical data
        self.hlc3_history.append(current_hlc3)
        self.tr_history.append(current_tr)
        self.close_history.append(current_close)
        
        # Skip if not enough data for Gaussian filter (need period + buffer)
        min_required = self.params.period + 25
        if len(self.hlc3_history) < min_required:
            return
        
        # Apply Gaussian filter to accumulated data (EXACT same as live trading)
        hlc3_series = pd.Series(self.hlc3_history)
        tr_series = pd.Series(self.tr_history)
        
        # Apply the filter
        filt_result, hband_result, lband_result = self.gaussian_filter.apply_filter(
            hlc3_series, tr_series
        )
        
        # Get current values (EXACT same as live trading)
        current_filt = filt_result.iloc[-1]
        current_hband = hband_result.iloc[-1]
        current_lband = lband_result.iloc[-1]
        
        # Skip if we don't have valid filter values
        if np.isnan(current_filt) or np.isnan(current_hband):
            return
        
        # Store values for analysis
        self.filt_values.append(current_filt)
        self.hband_values.append(current_hband)
        self.lband_values.append(current_lband)
        
        # === ENTRY/EXIT CONDITIONS (EXACT same as live trading) ===
        # Entry: Current bar close above current band (regardless of channel color)
        can_enter = current_close > current_hband
        
        # Exit: Current bar close below current band
        exit_signal = current_close < current_hband
        
        # === TRADING LOGIC (EXACT same as live trading) ===
        if not self.position:  # No position
            if can_enter:
                # Calculate position size (100% of available cash)
                size = self.broker.getcash() * self.params.position_size_pct / current_close
                self.buy(size=size)
                self.entry_count += 1
                self.last_entry_price = current_close
                self.log(f'BUY EXECUTED: {current_close:.2f}, Size: {size:.2f}')
        
        else:  # Have position
            if exit_signal:
                # Close entire position
                self.close()
                self.last_entry_price = None
                self.log(f'SELL EXECUTED: {current_close:.2f}')
    
    def log(self, txt, dt=None):
        """Logging function for strategy events"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}: {txt}')


def create_backtrader_datafeed(data, datetime_col='Date'):
    """
    Convert pandas DataFrame to backtrader data feed
    
    Args:
        data: DataFrame with OHLCV data
        datetime_col: Name of datetime column
        
    Returns:
        backtrader.feeds.PandasData object
    """
    # Ensure data has proper datetime index
    if not isinstance(data.index, pd.DatetimeIndex):
        data = data.copy()
        if datetime_col in data.columns:
            data[datetime_col] = pd.to_datetime(data[datetime_col])
            data.set_index(datetime_col, inplace=True)
        else:
            # Assume index should be datetime
            data.index = pd.to_datetime(data.index)
    
    # Verify data has required columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Create backtrader data feed
    datafeed = bt.feeds.PandasData(
        dataname=data,
        datetime=None,  # Use index as datetime
        open='Open',
        high='High', 
        low='Low',
        close='Close',
        volume='Volume',
        openinterest=None
    )
    
    return datafeed


def run_backtrader_backtest(data, strategy_class, strategy_params=None, 
                          initial_cash=10000, commission=0.001, 
                          slippage_perc=0.01, margin=0.2):
    """
    Run backtest using backtrader
    
    Args:
        data: DataFrame with OHLCV data
        strategy_class: Backtrader strategy class
        strategy_params: Strategy parameters dict
        initial_cash: Initial capital
        commission: Commission percentage
        slippage_perc: Slippage percentage
        margin: Margin requirement
        
    Returns:
        backtrader.Cerebro instance with results
    """
    # Create Cerebro engine
    cerebro = bt.Cerebro()
    
    # Add data feed with error checking
    try:
        datafeed = create_backtrader_datafeed(data)
        cerebro.adddata(datafeed)
        print(f"✅ Data feed added successfully")
    except Exception as e:
        print(f"❌ Error creating data feed: {e}")
        raise
    
    # Configure broker
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)
    cerebro.broker.set_slippage_perc(slippage_perc)
    # Note: set_margin is not available in standard backtrader
    # Margin handling is done at the strategy level
    
    # Add strategy with error checking
    try:
        if strategy_params:
            cerebro.addstrategy(strategy_class, **strategy_params)
        else:
            cerebro.addstrategy(strategy_class)
        print(f"✅ Strategy added successfully")
    except Exception as e:
        print(f"❌ Error adding strategy: {e}")
        raise
    
    # Run backtest
    try:
        cerebro.run()
        print(f"✅ Backtest completed successfully")
    except Exception as e:
        print(f"❌ Error running backtest: {e}")
        raise
    
    return cerebro


def analyze_backtrader_results(cerebro):
    """
    Analyze backtrader results and return performance metrics
    
    Args:
        cerebro: Backtrader Cerebro instance with results
        
    Returns:
        dict: Performance metrics
    """
    # Get final portfolio value
    final_value = cerebro.broker.getvalue()
    initial_value = cerebro.broker.startingcash
    total_return = (final_value - initial_value) / initial_value
    
    # More robust strategy instance retrieval
    strategy = None
    try:
        if hasattr(cerebro, 'runstrats') and cerebro.runstrats:
            strategy_list = cerebro.runstrats[0]
            if isinstance(strategy_list, list) and len(strategy_list) > 0:
                strategy = strategy_list[0]  # First strategy instance
        elif hasattr(cerebro, 'strategy') and cerebro.strategy:
            strategy = cerebro.strategy
        elif hasattr(cerebro, '_strategy') and cerebro._strategy:
            strategy = cerebro._strategy
    except Exception as e:
        print(f"Warning: Could not retrieve strategy instance: {e}")
    
    # Basic metrics
    metrics = {
        'initial_cash': initial_value,
        'final_value': final_value,
        'total_return': total_return,
        'total_return_pct': total_return * 100,
        'strategy': strategy
    }
    
    return metrics


def plot_backtrader_results(cerebro, save_path=None):
    """
    Plot backtrader results
    
    Args:
        cerebro: Backtrader Cerebro instance with results
        save_path: Optional path to save plot
    """
    # Plot results
    cerebro.plot(style='candlestick', barup='green', bardown='red')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show() 