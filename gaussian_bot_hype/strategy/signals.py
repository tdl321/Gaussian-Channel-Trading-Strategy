#!/usr/bin/env python3
"""
Signal Generation Module for Gaussian Channel Strategy

This module handles all signal generation logic for the Gaussian Channel strategy,
supporting both backtesting and live trading scenarios.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignalGenerator:
    """
    Signal generator for Gaussian Channel strategy
    
    Handles signal generation for both backtesting and live trading,
    including entry/exit conditions, pyramiding, and risk management.
    """
    
    def __init__(self, gaussian_filter, config):
        """
        Initialize signal generator
        
        Args:
            gaussian_filter: GaussianChannelFilter instance
            config: Config instance with strategy parameters
        """
        self.gaussian_filter = gaussian_filter
        self.config = config
        
        # Strategy parameters from config
        self.poles = config.POLES
        self.period = config.PERIOD
        self.multiplier = config.MULTIPLIER
        self.atr_spacing = config.ATR_SPACING
        self.max_pyramids = config.MAX_PYRAMIDS
        self.position_size_pct = config.POSITION_SIZE_PCT
        self.start_date = config.START_DATE
        self.end_date = config.END_DATE
        
        # Strategy state
        self.reset_state()
    
    def reset_state(self):
        """Reset strategy state variables"""
        self.entry_count = 0
        self.last_entry_price = None
        self.last_signal_check = None
    
    def load_csv_data(self, csv_path: str, date_column: str = 'Date', 
                     start_date: Optional[str] = None, end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Load and prepare CSV data for signal generation
        
        Args:
            csv_path: Path to CSV file
            date_column: Name of date column
            start_date: Start date filter (optional)
            end_date: End date filter (optional)
            
        Returns:
            DataFrame with prepared data or None if error
        """
        try:
            # Load CSV data
            data = pd.read_csv(csv_path)
            
            # Convert date column
            data[date_column] = pd.to_datetime(data[date_column])
            data.set_index(date_column, inplace=True)
            
            # Apply date filters
            if start_date:
                data = data[data.index >= pd.to_datetime(start_date)]
            if end_date:
                data = data[data.index <= pd.to_datetime(end_date)]
            
            # Ensure required columns exist
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return None
            
            # Calculate additional required data
            data = self._prepare_data(data)
            
            logger.info(f"Loaded {len(data)} bars from {csv_path}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading CSV data: {e}")
            return None
    
    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data with all required indicators for signal generation
        
        Args:
            data: Raw OHLCV DataFrame
            
        Returns:
            DataFrame with all indicators added
        """
        # Calculate HLC3 (High, Low, Close average)
        data['hlc3'] = (data['High'] + data['Low'] + data['Close']) / 3
        
        # Calculate True Range
        data['tr1'] = data['High'] - data['Low']
        data['tr2'] = abs(data['High'] - data['Close'].shift(1))
        data['tr3'] = abs(data['Low'] - data['Close'].shift(1))
        data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Calculate ATR using RMA (Relative Moving Average)
        data['atr'] = self._calculate_rma(data['true_range'], 14)
        
        # No SMA filter needed - removed from strategy
        
        # Clean up temporary columns
        data.drop(['tr1', 'tr2', 'tr3'], axis=1, inplace=True)
        
        return data
    
    def _calculate_rma(self, series: pd.Series, period: int) -> pd.Series:
        """
        Calculate Relative Moving Average (RMA)
        
        Args:
            series: Input series
            period: RMA period
            
        Returns:
            RMA series
        """
        alpha = 1.0 / period
        rma = series.copy()
        rma.iloc[0] = series.iloc[0]
        
        for i in range(1, len(series)):
            rma.iloc[i] = alpha * series.iloc[i] + (1 - alpha) * rma.iloc[i-1]
        
        return rma
    
    def prepare_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare trading signals based on Gaussian Channel (Updated for accurate visualization)
        
        Uses current bar data for both plotting and entries to ensure visual accuracy.
        The plotted bands now represent the actual bands used for entry/exit decisions.
        
        Args:
            data: DataFrame with OHLC and indicators
            
        Returns:
            DataFrame with signal columns added
        """
        # === CURRENT BARS (for plotting AND entries) - ACCURATE VISUALIZATION ===
        # Use current bar data for both plotting and entries to ensure visual accuracy
        src_current = data['hlc3']  # Current bar's hlc3
        tr_current = data['true_range']  # Current bar's true range
        
        # Apply Gaussian filter to current bars
        filt_current, hband_current, lband_current = self.gaussian_filter.apply_filter(src_current, tr_current)
        
        # Add current bar data to dataframe (for plotting AND entries)
        data['filt'] = filt_current  # Use current data for plotting
        data['hband'] = hband_current  # Use current data for plotting
        data['lband'] = lband_current  # Use current data for plotting
        
        # Green channel condition (filter rising) - Based on current data
        data['green_channel'] = (data['filt'] > data['filt'].shift(1)).fillna(False)
        
        # === ENTRY CONDITIONS (using current bar data) ===
        # Green channel entry: Current bar close above current band (true 0-bar delay continuation)
        data['green_entry'] = (
            data['green_channel'] & 
            (data['Close'] > data['hband']) &
            (data.index >= pd.to_datetime(self.start_date)) &
            (data.index <= pd.to_datetime(self.end_date))
        )
        
        # Red channel entry: Current bar close above current band (true 0-bar delay reversal)
        data['red_entry'] = (
            ~data['green_channel'] & 
            (data['Close'] > data['hband']) &
            (data.index >= pd.to_datetime(self.start_date)) &
            (data.index <= pd.to_datetime(self.end_date))
        )
        
        # === EXIT CONDITION (using current bar data) ===
        # Exit condition (closes all positions) - FAST EXIT using current bar data
        data['exit_signal'] = (data['Close'] < data['hband'])
        
        # === SUFFICIENT DATA CHECK ===
        # Ensure sufficient data for Gaussian Channel calculation
        sufficient_data = len(data) >= (self.period + 25)
        data['sufficient_data'] = sufficient_data
        
        # Apply sufficient data filter to entries
        data['green_entry'] = data['green_entry'] & sufficient_data
        data['red_entry'] = data['red_entry'] & sufficient_data
        
        return data
    
    def generate_live_signals(self, current_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generate trading signals for live trading
        
        Args:
            current_data: DataFrame with current market data and indicators
            
        Returns:
            List of signal dictionaries with action, size, and reason
        """
        signals = []
        
        if len(current_data) < 2:
            return signals
        
        # Get current bar index (last row)
        i = len(current_data) - 1
        current_date = current_data.index[i]
        current_price = current_data['Close'].iloc[i]
        atr_val = current_data['atr'].iloc[i]
        
        # Check date range
        if current_date < pd.to_datetime(self.start_date) or current_date > pd.to_datetime(self.end_date):
            return signals
        
        # Entry logic based on current bar's signal
        can_enter = (current_data['green_entry'].iloc[i] or current_data['red_entry'].iloc[i])
        
        # Base entry (first position) - matches Pine Script logic
        if can_enter:
            signals.append({
                'action': 'BUY',
                'size': self.position_size_pct,
                'reason': 'Base Entry',
                'price': current_price,
                'timestamp': current_date
            })
        
        # Exit logic
        if current_data['exit_signal'].iloc[i]:
            signals.append({
                'action': 'SELL',
                'size': 1.0,  # Close entire position
                'reason': 'Signal Exit',
                'price': current_price,
                'timestamp': current_date
            })
        
        return signals
    
    def generate_backtest_signals(self, data: pd.DataFrame, i: int, backtester) -> None:
        """
        Generate trading signals for backtesting (called by backtester)
        
        Args:
            data: DataFrame with OHLC and indicators
            i: Current bar index
            backtester: Backtester instance
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
    
    def get_signal_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary of signals in the data
        
        Args:
            data: DataFrame with signals prepared
            
        Returns:
            Dictionary with signal statistics
        """
        if 'green_entry' not in data.columns:
            return {}
        
        total_bars = len(data)
        green_entries = data['green_entry'].sum()
        red_entries = data['red_entry'].sum()
        exits = data['exit_signal'].sum()
        
        return {
            'total_bars': total_bars,
            'green_entries': green_entries,
            'red_entries': red_entries,
            'total_entries': green_entries + red_entries,
            'exits': exits,
            'entry_rate': (green_entries + red_entries) / total_bars if total_bars > 0 else 0
        } 