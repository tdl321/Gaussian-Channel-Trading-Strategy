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
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy.gaussian_filter import GaussianChannelFilter, calculate_rma
from strategy.signals import SignalGenerator


class PositionManager:
    """
    Position management system for backtesting
    Handles position state, leverage, risk management, and trade tracking
    """
    
    def __init__(self, leverage=5, max_position_size=1.0, stop_loss_pct=0.05):
        """
        Initialize position manager
        
        Args:
            leverage: Leverage multiplier (from config)
            max_position_size: Maximum position size as fraction of portfolio
            stop_loss_pct: Stop loss percentage (optional)
        """
        self.leverage = leverage
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        
        # Position state
        self.in_position = False
        self.entry_price = None
        self.entry_date = None
        self.position_size = 0.0
        self.entry_value = 0.0
        
        # Trade tracking
        self.trades = []
        self.current_trade = None
        
        # Risk management
        self.max_drawdown = 0.0
        self.peak_value = 0.0
        
    def can_enter(self, current_price, portfolio_value):
        """
        Check if we can enter a new position
        
        Args:
            current_price: Current asset price
            portfolio_value: Current portfolio value
            
        Returns:
            bool: True if can enter, False otherwise
        """
        if self.in_position:
            return False
        
        # Check if we have enough capital
        required_margin = (current_price * self.max_position_size) / self.leverage
        return portfolio_value >= required_margin
    
    def can_exit(self):
        """Check if we can exit current position"""
        return self.in_position
    
    def enter_position(self, price, date, portfolio_value):
        """
        Enter a new position
        
        Args:
            price: Entry price
            date: Entry date
            portfolio_value: Portfolio value at entry
        """
        if self.in_position:
            return False
        
        # Calculate position size with leverage
        position_value = portfolio_value * self.max_position_size
        self.position_size = (position_value * self.leverage) / price
        self.entry_price = price
        self.entry_date = date
        self.entry_value = position_value
        self.in_position = True
        
        # Start tracking current trade
        self.current_trade = {
            'entry_date': date,
            'entry_price': price,
            'entry_value': position_value,
            'position_size': self.position_size,
            'leverage': self.leverage,
            'exit_date': None,
            'exit_price': None,
            'exit_value': None,
            'pnl': 0.0,
            'pnl_pct': 0.0,
            'duration_days': 0,
            'status': 'OPEN'
        }
        
        return True
    
    def exit_position(self, price, date):
        """
        Exit current position
        
        Args:
            price: Exit price
            date: Exit date
            
        Returns:
            dict: Trade result or None if no position
        """
        if not self.in_position:
            return None
        
        # Calculate exit values with proper leverage PnL calculation
        price_change_pct = (price - self.entry_price) / self.entry_price
        pnl_pct = price_change_pct * self.leverage * 100  # Leveraged PnL percentage
        pnl = self.entry_value * (pnl_pct / 100)  # Actual dollar PnL
        exit_value = self.entry_value + pnl
        duration_days = (date - self.entry_date).days
        
        # Complete current trade
        if self.current_trade:
            self.current_trade.update({
                'exit_date': date,
                'exit_price': price,
                'exit_value': exit_value,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'duration_days': duration_days,
                'status': 'COMPLETED'
            })
            self.trades.append(self.current_trade)
        
        # Reset position state
        self.in_position = False
        self.entry_price = None
        self.entry_date = None
        self.position_size = 0.0
        self.entry_value = 0.0
        
        trade_result = self.current_trade.copy()
        self.current_trade = None
        
        return trade_result
    
    def check_stop_loss(self, current_price):
        """
        Check if stop loss has been hit
        
        Args:
            current_price: Current asset price
            
        Returns:
            bool: True if stop loss hit, False otherwise
        """
        if not self.in_position or not self.stop_loss_pct:
            return False
        
        loss_pct = (self.entry_price - current_price) / self.entry_price
        return loss_pct >= self.stop_loss_pct
    
    def get_position_info(self):
        """Get current position information"""
        return {
            'in_position': self.in_position,
            'entry_price': self.entry_price,
            'entry_date': self.entry_date,
            'position_size': self.position_size,
            'entry_value': self.entry_value,
            'leverage': self.leverage
        }
    
    def get_trade_statistics(self):
        """Get trade statistics"""
        if not self.trades:
            return {}
        
        completed_trades = [t for t in self.trades if t['status'] == 'COMPLETED']
        if not completed_trades:
            return {}
        
        total_trades = len(completed_trades)
        winning_trades = len([t for t in completed_trades if t['pnl_pct'] > 0])
        losing_trades = len([t for t in completed_trades if t['pnl_pct'] < 0])
        win_rate = winning_trades / total_trades * 100
        
        total_pnl = sum(t['pnl'] for t in completed_trades)
        avg_pnl = total_pnl / total_trades
        avg_duration = sum(t['duration_days'] for t in completed_trades) / total_trades
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'avg_duration': avg_duration,
            'open_trades': 1 if self.in_position else 0
        }


class GaussianChannelStrategy(bt.Strategy):
    """
    Gaussian Channel Strategy implemented as a backtrader strategy
    
    This strategy implements the Gaussian Channel filter with:
    - Long-only trend following (matches live trading exactly)
    - Dynamic exit based on price closing below upper band
    - 6 poles, 144 period, 1.414 multiplier (matches live trading)
    - Proper position management with leverage
    - Risk management and trade tracking
    - 100% position size per trade
    """
    
    params = (
        ('poles', 6),                    # Number of poles for Gaussian filter (matches live trading)
        ('period', 144),                 # Sampling period for Gaussian filter
        ('multiplier', 1.414),           # Band width multiplier
        ('leverage', 5),                 # Leverage multiplier (from config)
        ('position_size_pct', 1.0),      # Position size as percentage of portfolio (100%)
        ('atr_period', 14),              # ATR calculation period (needed for channel calculation)
        ('stop_loss_pct', 0.05),         # Stop loss percentage (5%)
        ('enable_stop_loss', False),     # Disable stop loss by default
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
        
        # Initialize position manager
        self.position_manager = PositionManager(
            leverage=self.params.leverage,
            max_position_size=self.params.position_size_pct,
            stop_loss_pct=self.params.stop_loss_pct if self.params.enable_stop_loss else None
        )
        
        # Strategy state variables
        self.entry_count = 0
        self.exit_count = 0
        
        # Store all historical data for signal generation (EXACT same as live trading)
        self.data_history = []
        self.current_bar_data = {}
        
    def next(self):
        """
        Main strategy logic - called for each bar
        
        This method implements the EXACT same logic as live trading using SignalGenerator:
        1. Prepare current bar data
        2. Use SignalGenerator to generate signals (EXACT same as live trading)
        3. Execute trades based on signals with position management
        """
        # Get current bar data (EXACT same format as live trading)
        current_date = self.datas[0].datetime.date(0)
        current_price = self.data.close[0]
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
        current_hband = df_with_signals['hband_current'].iloc[current_index]
        current_filt = df_with_signals['filt_current'].iloc[current_index]
        
        # Skip if we don't have valid filter values
        if np.isnan(current_filt) or np.isnan(current_hband):
            return
        
        # Get current portfolio value
        portfolio_value = self.broker.getvalue()
        
        # === POSITION MANAGEMENT LOGIC ===
        
        # Check stop loss first (if enabled)
        if self.params.enable_stop_loss and self.position_manager.check_stop_loss(current_price):
            if self.position_manager.can_exit():
                trade_result = self.position_manager.exit_position(current_price, current_date)
                if trade_result:
                    self.close()  # Close all positions
                    self.exit_count += 1
                    self.log(f'STOP LOSS EXIT: {current_date} at ${current_price:.2f} | PnL: {trade_result["pnl_pct"]:+.2f}%')
        
        # Check exit signal
        elif exit_signal and self.position_manager.can_exit():
            trade_result = self.position_manager.exit_position(current_price, current_date)
            if trade_result:
                self.close()  # Close all positions
                self.exit_count += 1
                
                # Update broker cash with actual PnL from the trade
                actual_pnl = trade_result['pnl']
                current_cash = self.broker.getcash()
                new_cash = current_cash + actual_pnl
                self.broker.setcash(new_cash)
                
                self.log(f'EXIT SIGNAL: {current_date} at ${current_price:.2f} | PnL: {trade_result["pnl_pct"]:+.2f}% | Duration: {trade_result["duration_days"]} days | Portfolio: ${self.broker.getvalue():.2f}')
        
        # Check entry signal
        elif entry_signal and self.position_manager.can_enter(current_price, portfolio_value):
            if self.position_manager.enter_position(current_price, current_date, portfolio_value):
                # Calculate position size for backtrader
                position_size = self.position_manager.position_size
                self.buy(size=position_size)
                self.entry_count += 1
                self.log(f'ENTRY SIGNAL: {current_date} at ${current_price:.2f} | Size: {position_size:.4f} | Leverage: {self.params.leverage}x | Portfolio: ${portfolio_value:.2f}')
        
        # === DIAGNOSTIC LOGGING (Every 100 bars for visibility) ===
        if len(self.data_history) % 100 == 0:
            position_info = self.position_manager.get_position_info()
            self.log(
                f'Portfolio=${portfolio_value:.2f}, Close=${current_price:.2f}, '
                f'InPosition={position_info["in_position"]}, Entry={entry_signal}, Exit={exit_signal}'
            )
    
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
    
    def stop(self):
        """Called when strategy stops - print final statistics"""
        stats = self.position_manager.get_trade_statistics()
        if stats:
            final_portfolio = self.broker.getvalue()
            initial_portfolio = self.broker.startingcash
            total_return = (final_portfolio - initial_portfolio) / initial_portfolio * 100
            
            print("\n" + "="*60)
            print("📊 FINAL TRADE STATISTICS")
            print("="*60)
            print(f"Initial Portfolio: ${initial_portfolio:,.2f}")
            print(f"Final Portfolio: ${final_portfolio:,.2f}")
            print(f"Total Return: {total_return:+.2f}%")
            print(f"Total Trades: {stats['total_trades']}")
            print(f"Winning Trades: {stats['winning_trades']}")
            print(f"Losing Trades: {stats['losing_trades']}")
            print(f"Win Rate: {stats['win_rate']:.1f}%")
            print(f"Total PnL: ${stats['total_pnl']:,.2f}")
            print(f"Average PnL per Trade: ${stats['avg_pnl']:,.2f}")
            print(f"Average Duration: {stats['avg_duration']:.1f} days")
            print(f"Open Positions: {stats['open_trades']}")
            print("="*60)


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


def run_backtrader_backtest(data, strategy_class=None, strategy_params=None, 
                          initial_cash=10000, commission=0.001, 
                          slippage_perc=0.01, debug_mode=False, use_config=True):
    """
    Run backtest using backtrader with position management
    
    Args:
        data: DataFrame with OHLCV data
        strategy_class: Backtrader strategy class (default: GaussianChannelStrategy)
        strategy_params: Strategy parameters dict
        initial_cash: Initial capital
        commission: Commission percentage
        slippage_perc: Slippage percentage
        debug_mode: Enable debug mode (reduces logging)
        use_config: Use config.py parameters for strategy settings
        
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
    
    # Load config parameters if requested
    if use_config:
        try:
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from config import POLES, PERIOD, MULTIPLIER, LEVERAGE
            
            # Override strategy params with config values
            if strategy_params is None:
                strategy_params = {}
            
            strategy_params.update({
                'poles': POLES,
                'period': PERIOD,
                'multiplier': MULTIPLIER,
                'leverage': LEVERAGE
            })
            
            print(f"✅ Loaded config parameters: Poles={POLES}, Period={PERIOD}, Multiplier={MULTIPLIER}, Leverage={LEVERAGE}x")
        except Exception as e:
            print(f"⚠️  Could not load config parameters: {e}")
            print("   Using default strategy parameters")
    
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


def create_backtest_report(cerebro, save_report=True, report_title="Gaussian Channel Backtest Report"):
    """
    Create comprehensive backtest report with trade analysis
    
    Args:
        cerebro: Backtrader Cerebro instance with results
        save_report: Save report to file
        report_title: Title for the report
        
    Returns:
        dict: Report data
    """
    print(f"\n📊 Creating {report_title}...")
    
    # Get basic metrics
    final_value = cerebro.broker.getvalue()
    initial_value = cerebro.broker.startingcash
    total_return = (final_value - initial_value) / initial_value
    
    # Get strategy instance for detailed analysis
    strategy = None
    try:
        if hasattr(cerebro, 'runstrats') and cerebro.runstrats:
            strategy_list = cerebro.runstrats[0]
            if isinstance(strategy_list, list) and len(strategy_list) > 0:
                strategy = strategy_list[0]
        elif hasattr(cerebro, 'strategy') and cerebro.strategy:
            strategy = cerebro.strategy
        elif hasattr(cerebro, '_strategy') and cerebro._strategy:
            strategy = cerebro._strategy
    except Exception as e:
        print(f"Warning: Could not retrieve strategy instance: {e}")
        strategy = None
    
    # Get position manager statistics if available
    position_stats = {}
    if strategy and hasattr(strategy, 'position_manager'):
        position_stats = strategy.position_manager.get_trade_statistics()
    
    # Create report data
    report_data = {
        'title': report_title,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'initial_cash': initial_value,
        'final_value': final_value,
        'total_return': total_return,
        'total_return_pct': total_return * 100,
        'position_stats': position_stats,
        'strategy_params': {}
    }
    
    # Add strategy parameters if available
    if strategy:
        report_data['strategy_params'] = {
            'poles': getattr(strategy.params, 'poles', 'N/A'),
            'period': getattr(strategy.params, 'period', 'N/A'),
            'multiplier': getattr(strategy.params, 'multiplier', 'N/A'),
            'leverage': getattr(strategy.params, 'leverage', 'N/A'),
            'stop_loss_pct': getattr(strategy.params, 'stop_loss_pct', 'N/A'),
            'enable_stop_loss': getattr(strategy.params, 'enable_stop_loss', 'N/A')
        }
    
    # Print report
    print("\n" + "="*80)
    print(f"📊 {report_title}")
    print("="*80)
    print(f"Report Generated: {report_data['timestamp']}")
    print()
    
    print("💰 PORTFOLIO PERFORMANCE:")
    print(f"   Initial Capital: ${initial_value:,.2f}")
    print(f"   Final Value: ${final_value:,.2f}")
    print(f"   Total Return: {total_return*100:+.2f}%")
    print(f"   Absolute PnL: ${final_value - initial_value:+,.2f}")
    print()
    
    if strategy_params := report_data['strategy_params']:
        print("⚙️ STRATEGY PARAMETERS:")
        print(f"   Poles: {strategy_params['poles']}")
        print(f"   Period: {strategy_params['period']}")
        print(f"   Multiplier: {strategy_params['multiplier']}")
        print(f"   Leverage: {strategy_params['leverage']}x")
        print(f"   Stop Loss: {strategy_params['stop_loss_pct']*100}%" if strategy_params['stop_loss_pct'] != 'N/A' else "   Stop Loss: Disabled")
        print()
    
    if position_stats:
        print("📈 TRADE STATISTICS:")
        print(f"   Total Trades: {position_stats['total_trades']}")
        print(f"   Winning Trades: {position_stats['winning_trades']}")
        print(f"   Losing Trades: {position_stats['losing_trades']}")
        print(f"   Win Rate: {position_stats['win_rate']:.1f}%")
        print(f"   Total PnL: ${position_stats['total_pnl']:+,.2f}")
        print(f"   Average PnL per Trade: ${position_stats['avg_pnl']:+,.2f}")
        print(f"   Average Duration: {position_stats['avg_duration']:.1f} days")
        print(f"   Open Positions: {position_stats['open_trades']}")
        print()
        
        # Calculate additional metrics
        if position_stats['total_trades'] > 0:
            profit_factor = abs(sum(t['pnl'] for t in strategy.position_manager.trades if t['pnl'] > 0) / 
                              sum(t['pnl'] for t in strategy.position_manager.trades if t['pnl'] < 0)) if sum(t['pnl'] for t in strategy.position_manager.trades if t['pnl'] < 0) != 0 else float('inf')
            print(f"   Profit Factor: {profit_factor:.2f}")
            
            # Calculate max drawdown from trades
            cumulative_pnl = 0
            peak_pnl = 0
            max_drawdown = 0
            for trade in strategy.position_manager.trades:
                cumulative_pnl += trade['pnl']
                if cumulative_pnl > peak_pnl:
                    peak_pnl = cumulative_pnl
                drawdown = peak_pnl - cumulative_pnl
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            
            print(f"   Max Drawdown: ${max_drawdown:,.2f}")
    
    print("="*80)
    
    # Save report if requested
    if save_report:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"backtest_report_{timestamp}.txt"
        save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            f.write(f"{report_title}\n")
            f.write("="*80 + "\n")
            f.write(f"Report Generated: {report_data['timestamp']}\n\n")
            
            f.write("PORTFOLIO PERFORMANCE:\n")
            f.write(f"Initial Capital: ${initial_value:,.2f}\n")
            f.write(f"Final Value: ${final_value:,.2f}\n")
            f.write(f"Total Return: {total_return*100:+.2f}%\n")
            f.write(f"Absolute PnL: ${final_value - initial_value:+,.2f}\n\n")
            
            if position_stats:
                f.write("TRADE STATISTICS:\n")
                f.write(f"Total Trades: {position_stats['total_trades']}\n")
                f.write(f"Winning Trades: {position_stats['winning_trades']}\n")
                f.write(f"Losing Trades: {position_stats['losing_trades']}\n")
                f.write(f"Win Rate: {position_stats['win_rate']:.1f}%\n")
                f.write(f"Total PnL: ${position_stats['total_pnl']:+,.2f}\n")
                f.write(f"Average PnL per Trade: ${position_stats['avg_pnl']:+,.2f}\n")
                f.write(f"Average Duration: {position_stats['avg_duration']:.1f} days\n")
                f.write(f"Open Positions: {position_stats['open_trades']}\n\n")
        
        print(f"📄 Report saved to: {save_path}")
    
    return report_data


def run_comprehensive_backtest(data_path, initial_cash=10000, commission=0.001, 
                             slippage_perc=0.01, debug_mode=False, save_results=True):
    """
    Run comprehensive backtest with full analysis and reporting
    
    Args:
        data_path: Path to CSV data file
        initial_cash: Initial capital
        commission: Commission percentage
        slippage_perc: Slippage percentage
        debug_mode: Enable debug mode
        save_results: Save results and reports
        
    Returns:
        dict: Complete backtest results
    """
    print("🚀 Starting Comprehensive Gaussian Channel Backtest")
    print("="*60)
    
    # Load data
    try:
        data = pd.read_csv(data_path)
        
        # Handle different date column names
        if 'Date' in data.columns:
            date_col = 'Date'
        elif 'Open time' in data.columns:
            date_col = 'Open time'
        else:
            # Try to find any datetime-like column
            datetime_cols = [col for col in data.columns if 'time' in col.lower() or 'date' in col.lower()]
            if datetime_cols:
                date_col = datetime_cols[0]
            else:
                raise ValueError("No date/time column found in data")
        
        data[date_col] = pd.to_datetime(data[date_col])
        data.set_index(date_col, inplace=True)
        
        # Ensure required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        print(f"✅ Loaded {len(data)} bars from {data.index[0].date()} to {data.index[-1].date()}")
        print(f"   Date column: {date_col}")
        print(f"   Available columns: {list(data.columns)}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None
    
    # Run backtest
    try:
        cerebro = run_backtrader_backtest(
            data=data,
            strategy_class=None,  # Use default GaussianChannelStrategy
            strategy_params=None,  # Use config parameters
            initial_cash=initial_cash,
            commission=commission,
            slippage_perc=slippage_perc,
            debug_mode=debug_mode,
            use_config=True  # Use config.py parameters
        )
        
        print("✅ Backtest completed successfully!")
        
        # Create comprehensive report
        report_data = create_backtest_report(
            cerebro, 
            save_report=save_results,
            report_title="Gaussian Channel Comprehensive Backtest Report"
        )
        
        # Plot results if matplotlib is available
        try:
            if save_results:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plot_filename = f"backtest_plot_{timestamp}.png"
                plot_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', plot_filename)
                os.makedirs(os.path.dirname(plot_path), exist_ok=True)
                
                print(f"📊 Generating plot...")
                plot_backtrader_results(cerebro, save_path=plot_path)
                print(f"📊 Plot saved to: {plot_path}")
            else:
                plot_backtrader_results(cerebro)
        except Exception as e:
            print(f"⚠️  Could not generate plot: {e}")
        
        return {
            'cerebro': cerebro,
            'report_data': report_data,
            'success': True
        }
        
    except Exception as e:
        print(f"❌ Error during backtest: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)} 


def load_asset_data(data_path, symbol):
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


def get_asset_strategy_params(symbol):
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
        # ETH parameters (optimized for altcoin-like behavior)
        return {
            'poles': 3,           # Faster response (optimized)
            'period': 72,         # Faster adaptation (optimized)
            'multiplier': 1.8,    # Wider channel (optimized)
            'position_size_pct': 1.0,  # 100% position size
            'atr_period': 7       # Faster volatility (optimized)
        }
    else:
        raise ValueError(f"Unsupported symbol: {symbol}")


def run_asset_backtest(symbol='BTC', data_path=None, initial_cash=10000, 
                      commission=0.001, slippage_perc=0.01, debug_mode=False):
    """
    Run backtest on specified symbol data with asset-specific handling
    
    Args:
        symbol: Symbol to test ('BTC' or 'ETH')
        data_path: Optional custom data path
        initial_cash: Initial capital
        commission: Commission percentage
        slippage_perc: Slippage percentage
        debug_mode: Enable debug mode
        
    Returns:
        dict: Backtest results
    """
    print(f"🚀 Starting {symbol} Backtest with Asset-Specific Parameters")
    print("=" * 60)
    
    # Set default data path if not provided
    if data_path is None:
        if symbol == 'BTC':
            data_path = "../data/btc_1d_data_2018_to_2025.csv"
        elif symbol == 'ETH':
            data_path = "../data/ETH-USD (2017-2024).csv"
        else:
            raise ValueError(f"Unsupported symbol: {symbol}")
    
    try:
        # Load and prepare data with asset-specific handling
        data = load_asset_data(data_path, symbol)
        
        # Get asset-specific strategy parameters
        strategy_params = get_asset_strategy_params(symbol)
        
        print(f"\n📊 Strategy Parameters for {symbol}:")
        for key, value in strategy_params.items():
            print(f"   {key}: {value}")
        
        # Run backtest with asset-specific parameters
        print(f"\n🔄 Running {symbol} backtest...")
        cerebro = run_backtrader_backtest(
            data=data,
            strategy_class=GaussianChannelStrategy,
            strategy_params=strategy_params,
            initial_cash=initial_cash,
            commission=commission,
            slippage_perc=slippage_perc,
            debug_mode=debug_mode,
            use_config=False  # Use asset-specific parameters instead of config
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
        return {
            'cerebro': cerebro,
            'results': results,
            'success': True
        }
        
    except Exception as e:
        print(f"\n❌ {symbol} backtest failed with error: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def run_btc_backtest(data_path=None, **kwargs):
    """Run backtest on BTC data (convenience function)"""
    return run_asset_backtest('BTC', data_path, **kwargs)


def run_eth_backtest(data_path=None, **kwargs):
    """Run backtest on ETH data (convenience function)"""
    return run_asset_backtest('ETH', data_path, **kwargs) 