#!/usr/bin/env python3
"""
Signal Generation Module for Gaussian Channel Strategy
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignalGenerator:
    """
    Signal generator for Gaussian Channel strategy
    """
    
    def __init__(self, gaussian_filter, config_params):
        """
        Initialize signal generator
        
        Args:
            gaussian_filter: GaussianChannelFilter instance
            config_params: Dictionary with strategy parameters
        """
        self.gaussian_filter = gaussian_filter
        
        # Strategy parameters from config
        self.poles = config_params['POLES']
        self.period = config_params['PERIOD']
        self.multiplier = config_params['MULTIPLIER']
        
        # Strategy state
        self.reset_state()
    
    def reset_state(self):
        """Reset strategy state variables"""
        self.current_position_size = 0.0  # 0 = no position, >0 = long position
        self.position_entry_price = None
        self.position_entry_date = None
    
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
        Prepare data with required indicators for signal generation
        
        Args:
            data: Raw OHLCV DataFrame
            
        Returns:
            DataFrame with indicators added
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
        
        # Clean up temporary columns
        data.drop(['tr1', 'tr2', 'tr3'], axis=1, inplace=True)
        
        return data
    
    def _calculate_rma(self, series: pd.Series, period: int) -> pd.Series:
        """
        Calculate Relative Moving Average (RMA)
        """
        alpha = 1.0 / period
        rma = series.copy()
        rma.iloc[0] = series.iloc[0]
        
        for i in range(1, len(series)):
            rma.iloc[i] = alpha * series.iloc[i] + (1 - alpha) * rma.iloc[i-1]
        
        return rma
    
    def prepare_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare trading signals based on Gaussian Channel
        
        Args:
            data: DataFrame with OHLC and indicators
            
        Returns:
            DataFrame with signal columns added
        """
        # Prepare data with indicators
        data = self._prepare_data(data)
        
        # === CONFIRMED DATA (non-repainting) ===
        # Use previous bar data for trend determination to avoid repainting
        src_confirmed = data['hlc3'].shift(1)  # Previous bar's hlc3
        tr_confirmed = data['true_range'].shift(1)  # Previous bar's true range
        
        # Apply Gaussian filter to confirmed bars
        filt_confirmed, hband_confirmed, lband_confirmed = self.gaussian_filter.apply_filter(src_confirmed, tr_confirmed)
        
        # Add confirmed data to dataframe
        data['filt_confirmed'] = filt_confirmed
        data['hband_confirmed'] = hband_confirmed
        data['lband_confirmed'] = lband_confirmed
        
        # Green channel condition (non-repainting) - Based on confirmed data
        data['green_channel'] = (filt_confirmed > filt_confirmed.shift(1)).fillna(False)
        
        # === CURRENT BAR DATA (for entries only) ===
        # Use current bar data for entry/exit decisions (0-bar delay)
        src_current = data['hlc3']  # Current bar's hlc3
        tr_current = data['true_range']  # Current bar's true range
        
        # Apply Gaussian filter to current bars
        filt_current, hband_current, lband_current = self.gaussian_filter.apply_filter(src_current, tr_current)
        
        # Add current bar data to dataframe
        data['filt_current'] = filt_current
        data['hband_current'] = hband_current
        data['lband_current'] = lband_current
        
        # === CHANNEL BULLISH CONDITION (Current Bar) ===
        # Define "bullish channel" when current filter is rising (vs previous current bar)
        data['channel_bullish'] = (filt_current > filt_current.shift(1)).fillna(False)  # Optional: for trend analysis/plotting
        
        # === ENTRY CONDITIONS (Simplified) ===
        # Entry: Current bar close above current band (regardless of channel color)
        data['entry_signal'] = (data['Close'] > hband_current)
        
        # === EXIT CONDITION ===
        # Exit condition (closes all positions)
        data['exit_signal'] = (data['Close'] < hband_current)
        
        # === SUFFICIENT DATA CHECK ===
        # Ensure sufficient data for Gaussian Channel calculation
        sufficient_data = len(data) >= (self.period + 25)
        data['sufficient_data'] = sufficient_data
        
        # Apply sufficient data filter to entries
        data['entry_signal'] = data['entry_signal'] & sufficient_data
        
        return data
    
    def generate_live_signals(self, current_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generate trading signals for live trading
        
        Args:
            current_data: DataFrame with current market data and indicators
            
        Returns:
            List of signal dictionaries with action and reason
        """
        signals = []
        
        if len(current_data) < 2:
            return signals
        
        # Get current bar index (last row)
        i = len(current_data) - 1
        current_date = current_data.index[i]
        current_price = current_data['Close'].iloc[i]
        
        # Check signals
        entry_signal = current_data['entry_signal'].iloc[i]
        exit_signal = current_data['exit_signal'].iloc[i]
        
        # === ENTRY LOGIC ===
        # Entry condition: entry signal and no current position
        if entry_signal and self.current_position_size == 0:
            # Update position state
            self.current_position_size = 1.0  # Full position
            self.position_entry_price = current_price
            self.position_entry_date = current_date
            
            signals.append({
                'action': 'BUY',
                'reason': 'Gaussian Channel Entry Signal',
                'price': current_price,
                'timestamp': current_date
            })
        
        # === EXIT LOGIC ===
        # Exit condition: exit signal and have position
        if exit_signal and self.current_position_size > 0:
            # Reset position state
            self.current_position_size = 0.0
            self.position_entry_price = None
            self.position_entry_date = None
            
            signals.append({
                'action': 'SELL',
                'reason': 'Gaussian Channel Exit Signal',
                'price': current_price,
                'timestamp': current_date
            })
        
        return signals
    
    def get_signal_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary of signals in the data
        
        Args:
            data: DataFrame with signals prepared
            
        Returns:
            Dictionary with signal statistics
        """
        if 'entry_signal' not in data.columns:
            return {}
        
        total_bars = len(data)
        entries = data['entry_signal'].sum()
        exits = data['exit_signal'].sum()
        
        return {
            'total_bars': total_bars,
            'entries': entries,
            'exits': exits,
            'entry_rate': entries / total_bars if total_bars > 0 else 0
        }
    
    def get_position_state(self) -> Dict[str, Any]:
        """
        Get current position state for monitoring
        
        Returns:
            Dictionary with current position information
        """
        return {
            'position_size': self.current_position_size,
            'entry_price': self.position_entry_price,
            'entry_date': self.position_entry_date,
            'is_in_position': self.current_position_size > 0
        } 