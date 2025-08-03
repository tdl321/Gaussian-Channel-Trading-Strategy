#!/usr/bin/env python3
"""
Gaussian Channel Strategy - Python Implementation
Converted from Pine Script v3.1 - EXACT 1:1 MATCH

A comprehensive trading strategy implementation featuring:
- Gaussian Channel filter with configurable poles
- ATR-based pyramiding (up to 5 entries)
- Bull market SMA filter
- Non-repainting confirmed-bar logic
- EXACT match to PineScript execution model
- Proper separation of strategy logic and backtester execution
- Advanced margin call and slippage system at backtester level
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

# Optional: Install with pip install backtesting
try:
    from backtesting import Backtest, Strategy
    from backtesting.lib import crossover
    BACKTESTING_AVAILABLE = True
except ImportError:
    print("Warning: backtesting.py not available. Install with: pip install backtesting")
    BACKTESTING_AVAILABLE = False

from scipy import stats


def calculate_rma(series, period):
    """
    Calculate Recursive Moving Average (RMA) / Wilder's smoothing
    This matches PineScript's ta.atr() calculation exactly
    
    RMA formula: RMA = (previous_RMA * (period-1) + current_value) / period
    This is equivalent to EMA with alpha = 1/period
    """
    alpha = 1.0 / period
    rma = series.ewm(alpha=alpha, adjust=False).mean()
    return rma


class GaussianChannelFilter:
    """
    Gaussian Channel Filter implementation converted from Pine Script
    Reproduces the f_filt9x and f_pole functions EXACTLY
    """
    
    def __init__(self, poles=4, period=144, multiplier=1.414, mode_lag=False, mode_fast=False):
        self.poles = poles
        self.period = period
        self.multiplier = multiplier
        self.mode_lag = mode_lag
        self.mode_fast = mode_fast
        
        # Pre-calculate filter coefficients
        self.lag = (period - 1) / (2 * poles)
        beta = (1 - np.cos(4 * np.arcsin(1) / period)) / (np.power(1.414, 2/poles) - 1)
        self.alpha = -beta + np.sqrt(np.power(beta, 2) + 2*beta)
        
        # Initialize filter state arrays for exact Pine Script implementation
        self.filter_states_src = {}
        self.filter_states_tr = {}
        for i in range(1, 10):  # Support up to 9 poles
            self.filter_states_src[i] = np.zeros(10)
            self.filter_states_tr[i] = np.zeros(10)
    
    def _get_binomial_weights(self, pole_num):
        """Get exact binomial coefficient weights as used in Pine Script"""
        # These are the exact weights from the Pine Script implementation
        weights = {
            1: [0, 0, 0, 0, 0, 0, 0, 0],
            2: [1, 0, 0, 0, 0, 0, 0, 0],
            3: [3, 1, 0, 0, 0, 0, 0, 0],
            4: [6, 4, 1, 0, 0, 0, 0, 0],
            5: [10, 10, 5, 1, 0, 0, 0, 0],
            6: [15, 20, 15, 6, 1, 0, 0, 0],
            7: [21, 35, 35, 21, 7, 1, 0, 0],
            8: [28, 56, 70, 56, 28, 8, 1, 0],
            9: [36, 84, 126, 126, 84, 36, 9, 1]
        }
        return weights.get(pole_num, [0] * 8)
    
    def _f_filt9x(self, alpha, source_val, pole_num, filter_state):
        """
        Exact implementation of Pine Script f_filt9x function
        """
        x = 1 - alpha
        weights = self._get_binomial_weights(pole_num)
        
        # Shift previous filter values
        filter_state[1:] = filter_state[:-1]
        
        # Calculate new filter value using exact Pine Script formula
        filt_val = np.power(alpha, pole_num) * source_val
        filt_val += pole_num * x * filter_state[1]
        
        # Apply binomial weights with alternating signs
        m2, m3, m4, m5, m6, m7, m8, m9 = weights
        
        if pole_num >= 2:
            filt_val -= m2 * np.power(x, 2) * filter_state[2]
        if pole_num >= 3:
            filt_val += m3 * np.power(x, 3) * filter_state[3]
        if pole_num >= 4:
            filt_val -= m4 * np.power(x, 4) * filter_state[4]
        if pole_num >= 5:
            filt_val += m5 * np.power(x, 5) * filter_state[5]
        if pole_num >= 6:
            filt_val -= m6 * np.power(x, 6) * filter_state[6]
        if pole_num >= 7:
            filt_val += m7 * np.power(x, 7) * filter_state[7]
        if pole_num >= 8:
            filt_val -= m8 * np.power(x, 8) * filter_state[8]
        if pole_num == 9:
            filt_val += m9 * np.power(x, 9) * filter_state[9]
        
        filter_state[0] = filt_val
        return filt_val
    
    def _f_pole(self, alpha, source_series, tr_series, pole_num):
        """
        Exact implementation of Pine Script f_pole function
        """
        n = len(source_series)
        filt_src = np.full(n, np.nan)
        filt_tr = np.full(n, np.nan)
        filt1_src = np.full(n, np.nan)
        filt1_tr = np.full(n, np.nan)
        
        # Process each data point
        for i in range(n):
            if not (np.isnan(source_series.iloc[i]) or np.isnan(tr_series.iloc[i])):
                # Calculate filter for the specified number of poles
                filt_src[i] = self._f_filt9x(alpha, source_series.iloc[i], pole_num, self.filter_states_src[pole_num])
                filt_tr[i] = self._f_filt9x(alpha, tr_series.iloc[i], pole_num, self.filter_states_tr[pole_num])
                
                # Also calculate single pole for fast mode
                filt1_src[i] = self._f_filt9x(alpha, source_series.iloc[i], 1, self.filter_states_src[1])
                filt1_tr[i] = self._f_filt9x(alpha, tr_series.iloc[i], 1, self.filter_states_tr[1])
        
        return filt_src, filt_tr, filt1_src, filt1_tr
    
    def apply_filter(self, source_data, true_range_data):
        """Apply exact Pine Script Gaussian filter to source and true range data"""
        # Drop NaN values but preserve index for later reindexing
        valid_mask = source_data.notna() & true_range_data.notna()
        
        if valid_mask.sum() < self.period:
            # Not enough data, return NaN series
            return (pd.Series(np.nan, index=source_data.index),
                    pd.Series(np.nan, index=source_data.index),
                    pd.Series(np.nan, index=source_data.index))
        
        # Get valid data
        valid_src = source_data[valid_mask]
        valid_tr = true_range_data[valid_mask]
        
        # Apply lag mode adjustment if enabled (exactly as in Pine Script)
        srcdata = valid_src.copy()
        trdata = valid_tr.copy()
        
        if self.mode_lag:
            lag_shift = max(1, int(self.lag))
            if len(srcdata) > lag_shift:
                for i in range(lag_shift, len(srcdata)):
                    srcdata.iloc[i] += (srcdata.iloc[i] - srcdata.iloc[i - lag_shift])
                    trdata.iloc[i] += (trdata.iloc[i] - trdata.iloc[i - lag_shift])
        
        # Apply the exact Pine Script filter
        filtn_src, filtn_tr, filt1_src, filt1_tr = self._f_pole(self.alpha, srcdata, trdata, self.poles)
        
        # Apply fast mode if enabled (exactly as in Pine Script)
        if self.mode_fast:
            filt_src = (filtn_src + filt1_src) / 2
            filt_tr = (filtn_tr + filt1_tr) / 2
        else:
            filt_src = filtn_src
            filt_tr = filtn_tr
        
        # Calculate bands
        hband = filt_src + filt_tr * self.multiplier
        lband = filt_src - filt_tr * self.multiplier
        
        # Create result series with original index
        filt_result = pd.Series(np.nan, index=source_data.index)
        hband_result = pd.Series(np.nan, index=source_data.index)
        lband_result = pd.Series(np.nan, index=source_data.index)
        
        # Fill in the calculated values
        filt_result.loc[valid_src.index] = filt_src
        hband_result.loc[valid_src.index] = hband
        lband_result.loc[valid_src.index] = lband
        
        return filt_result, hband_result, lband_result


class AdvancedBacktester:
    """
    Advanced backtester with proper slippage, margin management, and execution logic
    Handles all execution mechanics separately from strategy logic
    """
    
    def __init__(self, 
                 initial_capital=10000,
                 commission_pct=0.001,
                 slippage_ticks=1,
                 slippage_per_tick=0.0001,  # 0.01% per tick
                 margin_requirement=0.2,    # 20% margin requirement
                 maintenance_margin_pct=0.75,  # 75% of initial margin triggers margin call
                 forced_liquidation_buffer=0.05,  # 5% buffer before forced liquidation
                 max_leverage=5.0,
                 verbose=True):
        
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_ticks = slippage_ticks
        self.slippage_per_tick = slippage_per_tick
        self.margin_requirement = margin_requirement
        self.maintenance_margin_pct = maintenance_margin_pct
        self.forced_liquidation_buffer = forced_liquidation_buffer
        self.max_leverage = max_leverage
        self.verbose = verbose
        
        # Account state
        self.cash = initial_capital
        self.position_size = 0
        self.position_cost_basis = 0
        self.margin_used = 0
        
        # Order management
        self.pending_orders = []
        
        # Tracking and logging
        self.trade_log = []
        self.equity_curve = []
        self.margin_call_log = []
        self.margin_call_active = False
        
    def reset(self):
        """Reset backtester state"""
        self.cash = self.initial_capital
        self.position_size = 0
        self.position_cost_basis = 0
        self.margin_used = 0
        self.pending_orders = []
        self.trade_log = []
        self.equity_curve = []
        self.margin_call_log = []
        self.margin_call_active = False
    
    def get_total_equity(self, current_price):
        """Calculate total account equity including unrealized P&L"""
        if self.position_size == 0:
            return self.cash
        
        position_market_value = self.position_size * current_price
        unrealized_pnl = position_market_value - self.position_cost_basis
        return self.cash + unrealized_pnl
    
    def apply_slippage(self, price, is_buy=True, volume_factor=1.0):
        """
        Apply realistic slippage based on order direction and size
        
        Args:
            price: Base execution price
            is_buy: True for buy orders, False for sell orders
            volume_factor: Multiplier for slippage based on order size (1.0 = normal)
        """
        base_slippage = price * (self.slippage_ticks * self.slippage_per_tick)
        adjusted_slippage = base_slippage * volume_factor
        
        if is_buy:
            return price + adjusted_slippage  # Pay more when buying
        else:
            return price - adjusted_slippage  # Receive less when selling
    
    def check_margin_call(self, current_price):
        """
        Check if account is in margin call and calculate margin metrics
        
        Returns:
            dict: Margin analysis including is_margin_call, margin_level, etc.
        """
        if self.position_size == 0 or self.margin_used == 0:
            return {
                'is_margin_call': False,
                'margin_level_pct': 100.0,
                'required_maintenance': 0,
                'current_equity': self.cash,
                'is_forced_liquidation': False
            }
        
        current_equity = self.get_total_equity(current_price)
        required_maintenance = self.margin_used * self.maintenance_margin_pct
        margin_level_pct = (current_equity / required_maintenance * 100) if required_maintenance > 0 else 100.0
        
        is_margin_call = current_equity < required_maintenance
        forced_liquidation_threshold = required_maintenance * (1 - self.forced_liquidation_buffer)
        is_forced_liquidation = current_equity < forced_liquidation_threshold
        
        return {
            'is_margin_call': is_margin_call,
            'margin_level_pct': margin_level_pct,
            'required_maintenance': required_maintenance,
            'current_equity': current_equity,
            'is_forced_liquidation': is_forced_liquidation,
            'margin_used': self.margin_used,
            'position_value': self.position_size * current_price
        }
    
    def place_order(self, order_type, size, reason, atr_distance=None):
        """
        Place an order to be executed on next bar
        
        Args:
            order_type: 'BUY' or 'SELL'
            size: Position size in shares or percentage
            reason: Reason for the order (for logging)
            atr_distance: ATR distance for pyramiding orders
        """
        order = {
            'type': order_type,
            'size': size,
            'reason': reason,
            'atr_distance': atr_distance,
            'timestamp': None  # Will be filled when executed
        }
        self.pending_orders.append(order)
        
        if self.verbose:
            print(f"📋 ORDER PLACED: {order_type} {size} shares - {reason}")
    
    def execute_pending_orders(self, date, open_price, high_price, low_price, close_price):
        """
        Execute all pending orders at market open with slippage
        
        Args:
            date: Current bar date
            open_price: Opening price for execution
            high_price: High price of the bar
            low_price: Low price of the bar
            close_price: Closing price for equity calculation
        """
        executed_orders = []
        
        for order in self.pending_orders:
            if order['type'] == 'BUY':
                success = self._execute_buy_order(order, date, open_price)
            elif order['type'] == 'SELL':
                success = self._execute_sell_order(order, date, open_price)
            else:
                success = False
                
            if success:
                executed_orders.append(order)
        
        # Remove executed orders
        self.pending_orders = [o for o in self.pending_orders if o not in executed_orders]
        
        # Check for margin calls after execution
        margin_analysis = self.check_margin_call(close_price)
        self._handle_margin_call(margin_analysis, date, close_price)
        
        # Record equity curve
        self.equity_curve.append({
            'date': date,
            'equity': margin_analysis['current_equity'],
            'cash': self.cash,
            'position_value': margin_analysis.get('position_value', 0),
            'margin_used': self.margin_used,
            'margin_level_pct': margin_analysis['margin_level_pct']
        })
    
    def _execute_buy_order(self, order, date, execution_price):
        """Execute a buy order with proper margin and slippage handling"""
        try:
            # Calculate order size
            if isinstance(order['size'], float) and order['size'] <= 1.0:
                # Percentage of equity
                total_equity = self.get_total_equity(execution_price)
                position_value = total_equity * order['size']
            else:
                # Absolute dollar amount
                position_value = order['size']
            
            # Apply slippage
            volume_factor = min(2.0, position_value / 100000)  # Larger orders = more slippage
            slipped_price = self.apply_slippage(execution_price, is_buy=True, volume_factor=volume_factor)
            
            # Calculate shares and costs
            shares = position_value / slipped_price
            total_cost = shares * slipped_price
            commission = total_cost * self.commission_pct
            
            # Check if using margin
            cash_needed = total_cost + commission
            if cash_needed > self.cash:
                # Use margin
                margin_needed = cash_needed - self.cash
                if margin_needed > self.cash * (self.max_leverage - 1):
                    if self.verbose:
                        print(f"❌ INSUFFICIENT MARGIN: Need ${margin_needed:,.2f}, Max available: ${self.cash * (self.max_leverage - 1):,.2f}")
                    return False
                
                self.margin_used += margin_needed
                actual_cash_used = self.cash
                self.cash = 0
            else:
                # Cash purchase
                actual_cash_used = cash_needed
                self.cash -= cash_needed
            
            # Update position
            if self.position_size == 0:
                self.position_cost_basis = total_cost
            else:
                # Average cost basis for pyramiding
                total_cost_basis = self.position_cost_basis + total_cost
                self.position_cost_basis = total_cost_basis
            
            self.position_size += shares
            
            # Log the trade
            self.trade_log.append({
                'date': date,
                'action': 'BUY',
                'price': slipped_price,
                'shares': shares,
                'total_cost': total_cost,
                'commission': commission,
                'reason': order['reason'],
                'margin_used': margin_needed if cash_needed > actual_cash_used else 0,
                'slippage': slipped_price - execution_price,
                'cash_after': self.cash,
                'atr_distance': order.get('atr_distance')
            })
            
            if self.verbose:
                print(f"📈 BUY EXECUTED: {date.strftime('%Y-%m-%d')} | "
                      f"Price: ${slipped_price:.2f} | Shares: {shares:.2f} | "
                      f"Slippage: ${slipped_price - execution_price:.4f} | {order['reason']}")
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"❌ BUY ORDER FAILED: {e}")
            return False
    
    def _execute_sell_order(self, order, date, execution_price):
        """Execute a sell order (close all positions)"""
        if self.position_size == 0:
            return False
        
        try:
            # Apply slippage
            volume_factor = min(2.0, (self.position_size * execution_price) / 100000)
            slipped_price = self.apply_slippage(execution_price, is_buy=False, volume_factor=volume_factor)
            
            # Calculate proceeds
            gross_proceeds = self.position_size * slipped_price
            commission = gross_proceeds * self.commission_pct
            net_proceeds = gross_proceeds - commission
            
            # Calculate P&L
            total_pnl = gross_proceeds - self.position_cost_basis
            
            # Return cash and clear margin
            self.cash += net_proceeds
            if self.margin_used > 0:
                # Pay back margin
                margin_to_repay = min(self.margin_used, self.cash)
                self.cash -= margin_to_repay
                self.margin_used -= margin_to_repay
            
            # Log the trade
            self.trade_log.append({
                'date': date,
                'action': 'SELL',
                'price': slipped_price,
                'shares': self.position_size,
                'gross_proceeds': gross_proceeds,
                'commission': commission,
                'net_proceeds': net_proceeds,
                'total_pnl': total_pnl,
                'reason': order['reason'],
                'slippage': execution_price - slipped_price,  # Negative slippage for sells
                'cash_after': self.cash,
                'margin_repaid': margin_to_repay if self.margin_used > 0 else 0
            })
            
            if self.verbose:
                print(f"📉 SELL EXECUTED: {date.strftime('%Y-%m-%d')} | "
                      f"Price: ${slipped_price:.2f} | Shares: {self.position_size:.2f} | "
                      f"P&L: ${total_pnl:.2f} | {order['reason']}")
            
            # Reset position
            self.position_size = 0
            self.position_cost_basis = 0
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"❌ SELL ORDER FAILED: {e}")
            return False
    
    def _handle_margin_call(self, margin_analysis, date, price):
        """Handle margin call logic and forced liquidations"""
        is_margin_call = margin_analysis['is_margin_call']
        is_forced_liquidation = margin_analysis['is_forced_liquidation']
        
        # Margin call triggered
        if is_margin_call and not self.margin_call_active:
            self.margin_call_active = True
            margin_event = {
                'date': date,
                'price': price,
                'event': 'MARGIN_CALL_TRIGGERED',
                'margin_level_pct': margin_analysis['margin_level_pct'],
                'required_maintenance': margin_analysis['required_maintenance'],
                'current_equity': margin_analysis['current_equity']
            }
            self.margin_call_log.append(margin_event)
            
            if self.verbose:
                print(f"⚠️  MARGIN CALL TRIGGERED: {date.strftime('%Y-%m-%d')} | "
                      f"Margin Level: {margin_analysis['margin_level_pct']:.1f}%")
        
        # Forced liquidation
        if is_forced_liquidation and self.position_size > 0:
            # Force immediate liquidation
            liquidation_order = {
                'type': 'SELL',
                'size': self.position_size,
                'reason': 'FORCED_LIQUIDATION',
                'atr_distance': None
            }
            self._execute_sell_order(liquidation_order, date, price)
            
            self.margin_call_log.append({
                'date': date,
                'price': price,
                'event': 'FORCED_LIQUIDATION',
                'margin_level_pct': margin_analysis['margin_level_pct']
            })
            
            if self.verbose:
                print(f"🚨 FORCED LIQUIDATION: {date.strftime('%Y-%m-%d')} | "
                      f"Margin Level: {margin_analysis['margin_level_pct']:.1f}%")
        
        # Margin call resolved
        elif not is_margin_call and self.margin_call_active:
            self.margin_call_active = False
            self.margin_call_log.append({
                'date': date,
                'price': price,
                'event': 'MARGIN_CALL_RESOLVED',
                'margin_level_pct': margin_analysis['margin_level_pct']
            })
            
            if self.verbose:
                print(f"✅ MARGIN CALL RESOLVED: {date.strftime('%Y-%m-%d')} | "
                      f"Margin Level: {margin_analysis['margin_level_pct']:.1f}%")


class GaussianChannelStrategy:
    """
    Main strategy class focused purely on signal generation
    Execution mechanics are handled by the AdvancedBacktester
    """
    
    def __init__(self, 
                 poles=4,
                 period=144, 
                 multiplier=1.414,
                 mode_lag=False,
                 mode_fast=False,
                 atr_spacing=0.4,
                 sma_length=200,
                 enable_sma_filter=False,
                 max_pyramids=5,
                 position_size_pct=0.65,  # % of total equity per trade
                 start_date='2018-01-01',
                 end_date='2069-12-31'):
        
        self.poles = poles
        self.period = period
        self.multiplier = multiplier
        self.mode_lag = mode_lag
        self.mode_fast = mode_fast
        self.atr_spacing = atr_spacing
        self.sma_length = sma_length
        self.enable_sma_filter = enable_sma_filter
        self.max_pyramids = max_pyramids
        self.position_size_pct = position_size_pct
        self.start_date = start_date
        self.end_date = end_date
        
        self.gaussian_filter = GaussianChannelFilter(poles, period, multiplier, mode_lag, mode_fast)
        
        # Strategy state
        self.last_entry_price = None
        self.entry_count = 0
    
    def reset_state(self):
        """Reset strategy state"""
        self.last_entry_price = None
        self.entry_count = 0
    

    
    def load_csv_data(self, csv_path, date_column='Date', start_date=None, end_date=None):
        """Load price data from CSV file"""
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date
            
        try:
            # Load CSV
            data = pd.read_csv(csv_path)
            
            # Handle date column
            if date_column in data.columns:
                data[date_column] = pd.to_datetime(data[date_column])
                data.set_index(date_column, inplace=True)
            else:
                # Try to find date column with common names
                date_cols = ['date', 'Date', 'timestamp', 'Timestamp', 'time', 'Time']
                found_date_col = None
                for col in date_cols:
                    if col in data.columns:
                        found_date_col = col
                        break
                
                if found_date_col:
                    data[found_date_col] = pd.to_datetime(data[found_date_col])
                    data.set_index(found_date_col, inplace=True)
                else:
                    # Assume first column is date if no date column found
                    data.iloc[:, 0] = pd.to_datetime(data.iloc[:, 0])
                    data.set_index(data.columns[0], inplace=True)
            
            # Standardize column names (handle both upper and lowercase)
            column_mapping = {
                'open': 'Open', 'Open': 'Open',
                'high': 'High', 'High': 'High', 
                'low': 'Low', 'Low': 'Low',
                'close': 'Close', 'Close': 'Close',
                'volume': 'Volume', 'Volume': 'Volume',
                # Bitcoin-specific common column names
                'price_open': 'Open', 'price_high': 'High', 
                'price_low': 'Low', 'price_close': 'Close',
                'volume_traded': 'Volume'
            }
            
            # Rename columns to standard format
            for old_name, new_name in column_mapping.items():
                if old_name in data.columns:
                    data.rename(columns={old_name: new_name}, inplace=True)
            
            # Check if we have required columns
            required_cols = ['Open', 'High', 'Low', 'Close']
            missing_cols = [col for col in required_cols if col not in data.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}. Available columns: {list(data.columns)}")
            
            # Add Volume column if missing (set to 0)
            if 'Volume' not in data.columns:
                data['Volume'] = 0
                print("Warning: Volume column not found, setting to 0")
            
            # Convert to numeric (handle any string formatting issues)
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_cols:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Filter by date range
            if start_date:
                data = data[data.index >= pd.to_datetime(start_date)]
            if end_date:
                data = data[data.index <= pd.to_datetime(end_date)]
            
            if data.empty:
                raise ValueError(f"No data found in date range {start_date} to {end_date}")
            
            # Sort by date to ensure proper order
            data.sort_index(inplace=True)
            
            # Calculate additional indicators
            data['hlc3'] = (data['High'] + data['Low'] + data['Close']) / 3
            data['true_range'] = np.maximum(
                data['High'] - data['Low'],
                np.maximum(
                    abs(data['High'] - data['Close'].shift(1)),
                    abs(data['Low'] - data['Close'].shift(1))
                )
            )
            
            # Use RMA (Recursive Moving Average) for ATR - EXACT PineScript match
            data['atr'] = calculate_rma(data['true_range'], 14)
            data[f'sma_{self.sma_length}'] = data['Close'].rolling(self.sma_length).mean()
            
            print(f"✅ Loaded {len(data)} rows from CSV")
            print(f"📅 Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
            print(f"💰 Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
            
            return data
            
        except Exception as e:
            print(f"❌ Error loading CSV data: {e}")
            return None
    
    def prepare_signals(self, data):
        """Prepare trading signals based on Gaussian Channel"""
        # Use confirmed bars (shift by 1) - EXACT PineScript match
        src_confirmed = data['hlc3'].shift(1)
        tr_confirmed = data['true_range'].shift(1)
        
        # Apply Gaussian filter
        filt, hband, lband = self.gaussian_filter.apply_filter(src_confirmed, tr_confirmed)
        
        # Add to dataframe
        data['filt'] = filt
        data['hband'] = hband
        data['lband'] = lband
        
        # Green channel condition (filter rising) - EXACT Pine Script: filt[1] > filt[2]
        data['green_channel'] = (data['filt'] > data['filt'].shift(1)).fillna(False)
        
        # Bull market filter - EXACT Pine Script: close[1] > sma200[1]
        if self.enable_sma_filter:
            data['bull_market'] = data['Close'].shift(1) > data[f'sma_{self.sma_length}'].shift(1)
        else:
            data['bull_market'] = True
        
        # Entry/Exit conditions - EXACT Pine Script logic
        close_confirmed = data['Close'].shift(1)  # This is close[1] in Pine Script
        
        data['price_above_band'] = (close_confirmed > data['hband']).fillna(False)
        data['price_below_band'] = (close_confirmed < data['hband']).fillna(False)
        
        data['green_entry'] = (
            data['green_channel'] & 
            data['price_above_band'] &
            data['bull_market']
        )
        
        data['red_entry'] = (
            ~data['green_channel'] & 
            data['price_above_band'] &
            data['bull_market']
        )
        
        # Exit condition
        data['exit_signal'] = data['price_below_band']
        
        return data
    
    def generate_signals(self, data, i, backtester):
        """
        Generate trading signals for a specific bar
        This is called by the backtester for each bar
        
        Args:
            data: DataFrame with OHLC and indicators
            i: Current bar index
            backtester: AdvancedBacktester instance
        """
        current_date = data.index[i]
        current_price = data['Close'].iloc[i]
        atr_val = data['atr'].iloc[i]
        
        # Check date range
        if current_date < pd.to_datetime(self.start_date) or current_date > pd.to_datetime(self.end_date):
            return
        
        # Entry logic based on current bar's signal
        can_enter = (data['green_entry'].iloc[i] or data['red_entry'].iloc[i])
        
        # Base entry (first position)
        if can_enter and backtester.position_size == 0:
            backtester.place_order('BUY', self.position_size_pct, "Base Entry")
            self.entry_count = 1
            self.last_entry_price = current_price
        
        # Pyramiding entry
        elif can_enter and backtester.position_size > 0 and self.entry_count < self.max_pyramids:
            if self.last_entry_price is not None and atr_val > 0:
                # Use previous bar's close for ATR distance calculation (Pine Script uses close[1])
                prev_close = data['Close'].iloc[i-1] if i > 0 else current_price
                atr_distance = (prev_close - self.last_entry_price) / atr_val
                if atr_distance >= self.atr_spacing:
                    backtester.place_order('BUY', self.position_size_pct, 
                                         f"Pyramid {self.entry_count}", atr_distance)
                    self.entry_count += 1
                    self.last_entry_price = current_price
        
        # Exit logic
        if data['exit_signal'].iloc[i] and backtester.position_size > 0:
            backtester.place_order('SELL', backtester.position_size, "Signal Exit")
            self.last_entry_price = None
            self.entry_count = 0
    
    def run_backtest(self, data, 
                    initial_capital=10000,
                    commission_pct=0.001,
                    slippage_ticks=1,
                    margin_requirement=0.2,
                    max_leverage=5.0,
                    verbose=True):
        """
        Run backtest using the AdvancedBacktester
        
        Args:
            data: OHLC data with indicators
            initial_capital: Starting capital
            commission_pct: Commission percentage (0.001 = 0.1%)
            slippage_ticks: Number of ticks of slippage
            margin_requirement: Margin requirement (0.2 = 20%)
            max_leverage: Maximum leverage allowed
            verbose: Print trade details
        """
        
        # Initialize backtester
        backtester = AdvancedBacktester(
            initial_capital=initial_capital,
            commission_pct=commission_pct,
            slippage_ticks=slippage_ticks,
            margin_requirement=margin_requirement,
            max_leverage=max_leverage,
            verbose=verbose
        )
        
        # Reset strategy state
        self.reset_state()
        
        # Ensure sufficient data
        start_idx = max(self.period + 25, self.sma_length + 1)
        
        # Run backtest bar by bar
        for i in range(start_idx, len(data) - 1):  # -1 to allow next bar execution
            current_date = data.index[i]
            
            # Execute pending orders from previous bar
            if i > start_idx:  # Skip first bar
                next_open = data['Open'].iloc[i]
                high_price = data['High'].iloc[i]
                low_price = data['Low'].iloc[i]
                close_price = data['Close'].iloc[i]
                
                backtester.execute_pending_orders(current_date, next_open, high_price, low_price, close_price)
            
            # Generate new signals for this bar
            self.generate_signals(data, i, backtester)
        
        return backtester
    
    def calculate_performance_metrics(self, backtester):
        """Calculate performance metrics from backtester results"""
        if not backtester.equity_curve:
            return {
                'Total Return (%)': 0.0,
                'CAGR (%)': 0.0,
                'Max Drawdown (%)': 0.0,
                'Volatility (%)': 0.0,
                'Sharpe Ratio': 0.0,
                'Sortino Ratio': 0.0,
                'Skewness': 0.0,
                'Win Rate (%)': 0.0,
                'Number of Trades': 0,
                'Avg Trade Duration (days)': 0.0,
                'Final Cash': backtester.cash,
                'Final Equity': backtester.cash,
                'Margin Calls Triggered': 0,
                'Forced Liquidations': 0,
                'Total Slippage Cost': 0.0
            }
        
        equity_df = pd.DataFrame(backtester.equity_curve)
        equity_df.set_index('date', inplace=True)
        
        # Basic returns
        total_return = (equity_df['equity'].iloc[-1] / backtester.initial_capital - 1) * 100
        
        # Daily returns
        equity_df['daily_return'] = equity_df['equity'].pct_change()
        daily_returns = equity_df['daily_return'].dropna()
        
        # Risk metrics
        volatility = daily_returns.std() * np.sqrt(252) * 100 if len(daily_returns) > 1 else 0
        sharpe_ratio = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
        
        # Downside deviation
        downside_returns = daily_returns[daily_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 1 else 0
        sortino_ratio = (daily_returns.mean() * 252) / downside_deviation if downside_deviation > 0 else 0
        
        # Maximum drawdown
        equity_df['peak'] = equity_df['equity'].expanding(min_periods=1).max()
        equity_df['drawdown'] = (equity_df['equity'] / equity_df['peak'] - 1) * 100
        max_drawdown = equity_df['drawdown'].min()
        
        # Trade statistics
        trades_df = pd.DataFrame(backtester.trade_log)
        if not trades_df.empty:
            buy_trades = trades_df[trades_df['action'] == 'BUY']
            sell_trades = trades_df[trades_df['action'] == 'SELL']
            num_trades = len(buy_trades)
            
            # Calculate total slippage cost
            total_slippage = trades_df['slippage'].abs().sum() if 'slippage' in trades_df.columns else 0
            
            if len(sell_trades) > 0 and len(buy_trades) > 0:
                # Calculate win rate from completed trades
                completed_trades = sell_trades[sell_trades['total_pnl'].notna()]
                winning_trades = len(completed_trades[completed_trades['total_pnl'] > 0])
                win_rate = (winning_trades / len(completed_trades) * 100) if len(completed_trades) > 0 else 0
            else:
                win_rate = 0
        else:
            num_trades = 0
            total_slippage = 0
            win_rate = 0
        
        # CAGR calculation
        years = (equity_df.index[-1] - equity_df.index[0]).days / 365.25
        cagr = (equity_df['equity'].iloc[-1] / backtester.initial_capital) ** (1/years) - 1 if years > 0 else 0
        cagr *= 100
        
        # Additional metrics
        skewness = stats.skew(daily_returns) if len(daily_returns) > 3 else 0
        
        # Margin call statistics
        margin_calls_triggered = len([log for log in backtester.margin_call_log if log['event'] == 'MARGIN_CALL_TRIGGERED'])
        forced_liquidations = len([log for log in backtester.margin_call_log if log['event'] == 'FORCED_LIQUIDATION'])
        
        return {
            'Total Return (%)': round(total_return, 2),
            'CAGR (%)': round(cagr, 2),
            'Max Drawdown (%)': round(max_drawdown, 2),
            'Volatility (%)': round(volatility, 2),
            'Sharpe Ratio': round(sharpe_ratio, 2),
            'Sortino Ratio': round(sortino_ratio, 2),
            'Skewness': round(skewness, 2),
            'Win Rate (%)': round(win_rate, 2),
            'Number of Trades': num_trades,
            'Final Cash': round(backtester.cash, 2),
            'Final Equity': round(equity_df['equity'].iloc[-1], 2),
            'Margin Calls Triggered': margin_calls_triggered,
            'Forced Liquidations': forced_liquidations,
            'Total Slippage Cost': round(total_slippage, 2)
        }
    
    def plot_results(self, data, backtester, save_path=None):
        """Create comprehensive plotting of results"""
        fig, axes = plt.subplots(4, 1, figsize=(15, 16))
        
        # Plot 1: Price with Gaussian Channel
        ax1 = axes[0]
        
        # Price and bands
        ax1.plot(data.index, data['Close'], label='Close Price', color='black', linewidth=1)
        ax1.plot(data.index, data['filt'], label='Gaussian Filter', color='blue', linewidth=2)
        ax1.plot(data.index, data['hband'], label='Upper Band', color='red', linewidth=1, alpha=0.7)
        ax1.plot(data.index, data['lband'], label='Lower Band', color='red', linewidth=1, alpha=0.7)
        
        # Channel fill with color coding
        for i in range(1, len(data)):
            if pd.notna(data['filt'].iloc[i]) and pd.notna(data['filt'].iloc[i-1]):
                color = 'green' if data['filt'].iloc[i] > data['filt'].iloc[i-1] else 'red'
                alpha = 0.1
                ax1.fill_between([data.index[i-1], data.index[i]], 
                               [data['lband'].iloc[i-1], data['lband'].iloc[i]], 
                               [data['hband'].iloc[i-1], data['hband'].iloc[i]], 
                               color=color, alpha=alpha)
        
        # Trade markers
        for trade in backtester.trade_log:
            if trade['action'] == 'BUY':
                ax1.scatter(trade['date'], trade['price'], color='green', marker='^', s=100, zorder=5)
            else:
                ax1.scatter(trade['date'], trade['price'], color='red', marker='v', s=100, zorder=5)
        
        ax1.set_title('Gaussian Channel Strategy - Price Action & Signals')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Equity Curve
        ax2 = axes[1]
        if backtester.equity_curve:
            equity_df = pd.DataFrame(backtester.equity_curve)
            equity_df.set_index('date', inplace=True)
            
            ax2.plot(equity_df.index, equity_df['equity'], label='Strategy Equity', color='blue', linewidth=2)
            
            # Buy and hold comparison
            initial_shares = backtester.initial_capital / data['Close'].iloc[0]
            buy_hold_value = initial_shares * data['Close']
            ax2.plot(data.index, buy_hold_value, label='Buy & Hold', color='gray', linewidth=1, alpha=0.7)
            
            ax2.set_title('Equity Curve Comparison')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Drawdown
        ax3 = axes[2]
        if backtester.equity_curve:
            equity_df['peak'] = equity_df['equity'].expanding(min_periods=1).max()
            equity_df['drawdown'] = (equity_df['equity'] / equity_df['peak'] - 1) * 100
            
            ax3.fill_between(equity_df.index, equity_df['drawdown'], 0, 
                           color='red', alpha=0.3, label='Drawdown')
            ax3.plot(equity_df.index, equity_df['drawdown'], color='red', linewidth=1)
            ax3.set_title('Strategy Drawdown (%)')
            ax3.set_ylabel('Drawdown (%)')
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Margin Level
        ax4 = axes[3]
        if backtester.equity_curve:
            ax4.plot(equity_df.index, equity_df['margin_level_pct'], label='Margin Level %', color='orange', linewidth=2)
            ax4.axhline(y=100, color='green', linestyle='--', alpha=0.7, label='Safe Level (100%)')
            ax4.axhline(y=75, color='red', linestyle='--', alpha=0.7, label='Margin Call (75%)')
            
            # Mark margin call events
            for event in backtester.margin_call_log:
                if event['event'] == 'MARGIN_CALL_TRIGGERED':
                    ax4.scatter(event['date'], event.get('margin_level_pct', 75), 
                              color='red', marker='v', s=100, zorder=5)
                elif event['event'] == 'FORCED_LIQUIDATION':
                    ax4.scatter(event['date'], event.get('margin_level_pct', 50), 
                              color='darkred', marker='X', s=150, zorder=5)
            
            ax4.set_title('Margin Level Monitoring')
            ax4.set_ylabel('Margin Level (%)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def run_strategy(self, symbol, plot=True, save_path=None, **backtest_kwargs):
        """Run the complete strategy pipeline with advanced backtester"""
        print(f"Running Gaussian Channel Strategy on {symbol}")
        print(f"Parameters: Poles={self.poles}, Period={self.period}, Multiplier={self.multiplier}")
        print(f"Date Range: {self.start_date} to {self.end_date}")
        print(f"Position Sizing: {self.position_size_pct*100}% of total equity")
        print("-" * 60)
        
        # Data should be provided externally - this method is deprecated
        print("Warning: run_strategy method requires data to be provided externally")
        return None
        
        # Prepare signals
        data = self.prepare_signals(data)
        
        # Run backtest with advanced backtester
        backtester = self.run_backtest(data, **backtest_kwargs)
        
        # Calculate performance
        metrics = self.calculate_performance_metrics(backtester)
        
        # Print results
        print("\n" + "="*50)
        print("PERFORMANCE METRICS")
        print("="*50)
        for metric, value in metrics.items():
            print(f"{metric:<25}: {value}")
        
        # Print margin call summary
        if backtester.margin_call_log:
            print(f"\n📊 MARGIN CALL SUMMARY:")
            margin_calls = len([e for e in backtester.margin_call_log if e['event'] == 'MARGIN_CALL_TRIGGERED'])
            liquidations = len([e for e in backtester.margin_call_log if e['event'] == 'FORCED_LIQUIDATION'])
            print(f"   Margin Calls: {margin_calls}")
            print(f"   Forced Liquidations: {liquidations}")
        
        # Plot results
        if plot:
            self.plot_results(data, backtester, save_path)
        
        return {
            'data': data,
            'metrics': metrics,
            'backtester': backtester,
            'trade_log': backtester.trade_log,
            'equity_curve': backtester.equity_curve,
            'margin_call_log': backtester.margin_call_log
        }
    
    def run_csv_strategy(self, csv_path, symbol_name="CSV Data", date_column='Date', 
                        plot=True, save_path=None, **backtest_kwargs):
        """Run the strategy pipeline using CSV data with advanced backtester"""
        print(f"Running Gaussian Channel Strategy on {symbol_name}")
        print(f"Data Source: {csv_path}")
        print(f"Parameters: Poles={self.poles}, Period={self.period}, Multiplier={self.multiplier}")
        print(f"Date Range: {self.start_date} to {self.end_date}")
        print(f"Position Sizing: {self.position_size_pct*100}% of total equity")
        print("-" * 60)
        
        # Load CSV data
        data = self.load_csv_data(csv_path, date_column)
        if data is None:
            print("Failed to load CSV data")
            return None
        
        # Prepare signals
        data = self.prepare_signals(data)
        
        # Run backtest with advanced backtester
        backtester = self.run_backtest(data, **backtest_kwargs)
        
        # Calculate performance
        metrics = self.calculate_performance_metrics(backtester)
        
        # Print results
        print("\n" + "="*50)
        print("PERFORMANCE METRICS")
        print("="*50)
        for metric, value in metrics.items():
            print(f"{metric:<25}: {value}")
        
        # Print margin call summary
        if backtester.margin_call_log:
            print(f"\n📊 MARGIN CALL SUMMARY:")
            margin_calls = len([e for e in backtester.margin_call_log if e['event'] == 'MARGIN_CALL_TRIGGERED'])
            liquidations = len([e for e in backtester.margin_call_log if e['event'] == 'FORCED_LIQUIDATION'])
            print(f"   Margin Calls: {margin_calls}")
            print(f"   Forced Liquidations: {liquidations}")
        
        # Plot results
        if plot:
            self.plot_results(data, backtester, save_path)
        
        return {
            'data': data,
            'metrics': metrics,
            'backtester': backtester,
            'trade_log': backtester.trade_log,
            'equity_curve': backtester.equity_curve,
            'margin_call_log': backtester.margin_call_log
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize strategy with default parameters
    strategy = GaussianChannelStrategy(
        poles=4,
        period=144,
        multiplier=1.414,
        mode_lag=False,
        mode_fast=False,
        atr_spacing=0.4,
        sma_length=200,
        enable_sma_filter=False,
        max_pyramids=5,
        position_size_pct=0.65,
        start_date='2020-01-01',
        end_date='2024-01-01'
    )
    
    # Test with SPY using advanced backtester
    print("Testing Gaussian Channel Strategy with Advanced Backtester...")
    results = strategy.run_strategy('SPY', 
                                   plot=True,
                                   initial_capital=10000,
                                   commission_pct=0.001,
                                   slippage_ticks=1,
                                   margin_requirement=0.2,
                                   max_leverage=5.0,
                                   verbose=True)
    
    if results:
        print("\nStrategy test completed successfully!")
        print(f"Final equity: ${results['backtester'].cash:,.2f}")
    else:
        print("Strategy test failed!") 