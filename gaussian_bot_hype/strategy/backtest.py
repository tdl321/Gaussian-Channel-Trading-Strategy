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

# Import our Gaussian filter and signal generator from the separate modules
from .gaussian_filter import GaussianChannelFilter, calculate_rma
from .signals import SignalGenerator


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
        # Initialize Gaussian filter (EXACT same as live trading)
        self.gaussian_filter = GaussianChannelFilter(
            poles=self.params.poles,
            period=self.params.period,
            multiplier=self.params.multiplier
        )
        
        # Initialize signal generator (EXACT same as live trading)
        config_params = {
            'POLES': self.params.poles,
            'PERIOD': self.params.period,
            'MULTIPLIER': self.params.multiplier
        }
        self.signal_generator = SignalGenerator(self.gaussian_filter, config_params)
        
        # Strategy state variables (EXACT same as live trading)
        self.entry_count = 0
        self.last_entry_price = None
        
        # Store all historical data for signal generation (EXACT same as live trading)
        self.data_history = []
        self.current_bar_data = {}
        
    def next(self):
        """
        Main strategy logic - called for each bar
        
        This method implements the EXACT same logic as live trading using SignalGenerator:
        1. Prepare current bar data
        2. Use SignalGenerator to generate signals (EXACT same as live trading)
        3. Execute trades based on signals
        """
        # Get current bar data (EXACT same format as live trading)
        current_date = self.datas[0].datetime.date(0)
        current_bar = {
            'Date': current_date,
            'Open': self.data.open[0],
            'High': self.data.high[0],
            'Low': self.data.low[0],
            'Close': self.data.close[0],
            'Volume': self.data.volume[0]
        }
        
        # Add current bar to history (EXACT same as live trading)
        self.data_history.append(current_bar)
        
        # Skip if not enough data for Gaussian filter (need period + buffer)
        min_required = self.params.period + 25
        if len(self.data_history) < min_required:
            self.log(f'Waiting for enough data: {len(self.data_history)}/{min_required}')
            return
        
        # Convert history to DataFrame (EXACT same as live trading)
        df = pd.DataFrame(self.data_history)
        df.set_index('Date', inplace=True)
        
        # Use SignalGenerator to prepare signals (EXACT same as live trading)
        df_with_signals = self.signal_generator.prepare_signals(df)
        
        # Get current bar signals (EXACT same as live trading)
        current_index = len(df_with_signals) - 1
        entry_signal = df_with_signals['entry_signal'].iloc[current_index]
        exit_signal = df_with_signals['exit_signal'].iloc[current_index]
        
        # Get current values for logging
        current_close = self.data.close[0]
        current_hband = df_with_signals['hband_current'].iloc[current_index]
        current_filt = df_with_signals['filt_current'].iloc[current_index]
        
        # Skip if we don't have valid filter values
        if np.isnan(current_filt) or np.isnan(current_hband):
            return
        
        # === DIAGNOSTIC LOGGING (Every bar for complete visibility) ===
        self.log(
            f'Close={current_close:.2f}, Filt={current_filt:.2f}, '
            f'HBand={current_hband:.2f}, Entry={"YES" if entry_signal else "NO"}, Exit={"YES" if exit_signal else "NO"}'
        )
        
        # === TRADING LOGIC (EXACT same as live trading) ===
        if not self.position:
            if entry_signal:
                size = self.broker.getcash() * self.params.position_size_pct / current_close
                self.buy(size=size)
                self.entry_count += 1
                self.last_entry_price = current_close
        else:
            if exit_signal:
                self.close()
                self.last_entry_price = None
    
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}: {txt}')

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED: {order.executed.price:.2f}, Size: {order.executed.size:.4f}')
            elif order.issell():
                self.log(f'SELL EXECUTED: {order.executed.price:.2f}, Size: {order.executed.size:.4f}')

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log(f'TRADE CLOSED: Gross PnL: {trade.pnl:.2f}, Net PnL: {trade.pnlcomm:.2f}')


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
                          slippage_perc=0.01, margin=0.2, debug_mode=False):
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
    
    # Handle slippage (set to 0 for debugging if requested)
    if debug_mode:
        print(f"🔧 DEBUG MODE: Setting slippage to 0 for testing")
        slippage_perc = 0
    
    cerebro.broker.set_slippage_perc(slippage_perc)
    print(f"💰 Broker configured: Cash=${initial_cash}, Commission={commission*100}%, Slippage={slippage_perc*100}%")
    
    # Note: set_margin is not available in standard backtrader
    # Margin handling is done at the strategy level
    
    # Add strategy with error checking
    try:
        if strategy_class is None:
            # Use default GaussianChannelStrategy
            cerebro.addstrategy(GaussianChannelStrategy, **strategy_params) if strategy_params else cerebro.addstrategy(GaussianChannelStrategy)
        else:
            cerebro.addstrategy(strategy_class, **strategy_params) if strategy_params else cerebro.addstrategy(strategy_class)
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