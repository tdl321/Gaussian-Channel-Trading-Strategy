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
    - Long-only trend following
    - Dynamic exit based on price closing below upper band
    - Configurable parameters for poles, period, and multiplier
    """
    
    params = (
        ('poles', 4),                    # Number of poles for Gaussian filter
        ('period', 144),                 # Sampling period for Gaussian filter
        ('multiplier', 1.414),           # Band width multiplier
        ('atr_spacing', 2.0),            # ATR spacing for pyramiding (currently disabled)
        ('max_pyramids', 0),             # Maximum pyramiding entries (0 = disabled)
        ('position_size_pct', 1.0),      # Position size as percentage of portfolio
        ('atr_period', 14),              # ATR calculation period
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
        
        # ATR calculation removed since we're not using it in the current strategy
        # self.atr = bt.indicators.EMA(self.true_range, period=self.params.atr_period)
        
        # Strategy state variables
        self.entry_count = 0
        self.last_entry_price = None
        self.sufficient_data = False
        
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
        
        This method implements the Gaussian Channel strategy with non-repainting logic:
        1. Use confirmed (previous bar) data for trend determination
        2. Use current bar data for entry/exit decisions
        3. Execute trades based on signals
        """
        # Skip if not enough data for Gaussian filter
        if len(self.data) < self.params.period + 25:
            return
        
        # === CONFIRMED DATA (non-repainting) ===
        # Use previous bar data for trend determination to avoid repainting
        if len(self.data) < 2:
            return  # Need at least 2 bars for confirmed data
        
        # Previous bar data (confirmed)
        prev_hlc3 = self.hlc3[-1]  # Previous bar's hlc3
        
        # Calculate True Range for previous bar
        prev_tr1 = self.tr1[-1]  # High - Low
        prev_tr2 = abs(self.tr2[-1])  # abs(High - Previous Close)
        prev_tr3 = abs(self.tr3[-1])  # abs(Low - Previous Close)
        prev_tr = max(prev_tr1, prev_tr2, prev_tr3)
        
        # Apply Gaussian filter to confirmed data
        filt_confirmed, hband_confirmed, lband_confirmed = self.gaussian_filter.apply_filter(
            pd.Series([prev_hlc3]), 
            pd.Series([prev_tr])
        )
        
        # Store confirmed values
        current_filt_confirmed = filt_confirmed.iloc[0] if not filt_confirmed.empty else np.nan
        current_hband_confirmed = hband_confirmed.iloc[0] if not hband_confirmed.empty else np.nan
        current_lband_confirmed = lband_confirmed.iloc[0] if not lband_confirmed.empty else np.nan
        
        self.filt_confirmed_values.append(current_filt_confirmed)
        self.hband_confirmed_values.append(current_hband_confirmed)
        self.lband_confirmed_values.append(current_lband_confirmed)
        
        # Determine green channel (non-repainting) - Based on confirmed data
        if len(self.filt_confirmed_values) >= 2:
            green_channel = self.filt_confirmed_values[-1] > self.filt_confirmed_values[-2]
        else:
            green_channel = False
        
        self.green_channel_values.append(green_channel)
        
        # === CURRENT BAR DATA (for entries only) ===
        # Use current bar data for entry/exit decisions (0-bar delay)
        current_hlc3 = self.hlc3[0]  # Current bar's hlc3
        
        # Calculate True Range for current bar
        tr1 = self.tr1[0]  # High - Low
        tr2 = abs(self.tr2[0])  # abs(High - Previous Close)
        tr3 = abs(self.tr3[0])  # abs(Low - Previous Close)
        current_tr = max(tr1, tr2, tr3)
        
        # Apply Gaussian filter to current bar
        filt_current, hband_current, lband_current = self.gaussian_filter.apply_filter(
            pd.Series([current_hlc3]), 
            pd.Series([current_tr])
        )
        
        # Store current values
        current_filt = filt_current.iloc[0] if not filt_current.empty else np.nan
        current_hband = hband_current.iloc[0] if not hband_current.empty else np.nan
        current_lband = lband_current.iloc[0] if not lband_current.empty else np.nan
        
        self.filt_values.append(current_filt)
        self.hband_values.append(current_hband)
        self.lband_values.append(current_lband)
        
        # Skip if we don't have valid filter values
        if np.isnan(current_filt_confirmed) or np.isnan(current_hband) or np.isnan(current_filt):
            return
        
        # === GREEN CHANNEL CONDITION (Current Bar) ===
        # Define "green channel" when current filter is rising (vs previous current bar)
        if len(self.filt_values) >= 2:
            green_channel_realtime = current_filt > self.filt_values[-2]
        else:
            green_channel_realtime = False
        
        # === ENTRY CONDITIONS (Simplified) ===
        current_close = self.data.close[0]
        
        # Entry: Current bar close above current band (regardless of channel color)
        can_enter = current_close > current_hband
        
        # === EXIT CONDITION (using current bar data) ===
        exit_signal = current_close < current_hband
        
        # === TRADING LOGIC ===
        if not self.position:  # No position
            if can_enter:
                # Calculate position size
                size = self.broker.getcash() * self.params.position_size_pct / current_close
                self.buy(size=size)
                self.entry_count = 1
                self.last_entry_price = current_close
                self.log(f'BUY EXECUTED: {current_close:.2f}, Size: {size:.2f}')
        
        else:  # Have position
            if exit_signal:
                # Close entire position
                self.close()
                self.entry_count = 0
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
    # Ensure datetime index
    if datetime_col in data.columns:
        data = data.copy()
        data[datetime_col] = pd.to_datetime(data[datetime_col])
        data.set_index(datetime_col, inplace=True)
    
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
    
    # Add data feed
    datafeed = create_backtrader_datafeed(data)
    cerebro.adddata(datafeed)
    
    # Configure broker
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)
    cerebro.broker.set_slippage_perc(slippage_perc)
    # Note: set_margin is not available in standard backtrader
    # Margin handling is done at the strategy level
    
    # Add strategy
    if strategy_params:
        cerebro.addstrategy(strategy_class, **strategy_params)
    else:
        cerebro.addstrategy(strategy_class)
    
    # Run backtest
    cerebro.run()
    
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
    
    # Get strategy instance (access from cerebro after running)
    strategy = None
    if hasattr(cerebro, 'runstrats') and cerebro.runstrats:
        # The strategy instance is stored in runstrats
        strategy_list = cerebro.runstrats[0]
        if isinstance(strategy_list, list) and len(strategy_list) > 0:
            strategy = strategy_list[0]  # First strategy instance
    elif hasattr(cerebro, 'strategy') and cerebro.strategy:
        strategy = cerebro.strategy
    
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