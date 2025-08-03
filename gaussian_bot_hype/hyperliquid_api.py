#!/usr/bin/env python3
"""
Hyperliquid API Wrapper

This module provides a wrapper around the official Hyperliquid SDK to:
1. Convert Hyperliquid data formats to pandas DataFrame
2. Handle string-to-float conversions for all prices
3. Convert millisecond timestamps to datetime
4. Standardize column names to ['Open', 'High', 'Low', 'Close', 'Volume']

Based on the .cursorrules specifications for the Gaussian Channel Trading Bot.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging

# Import official Hyperliquid SDK
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils.constants import MAINNET_API_URL, TESTNET_API_URL
from eth_account import Account

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HyperliquidAPI:
    """
    Wrapper for Hyperliquid SDK with data conversion utilities
    
    Provides standardized data formats for the Gaussian Channel strategy
    """
    
    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None, 
                 base_url: str = MAINNET_API_URL, skip_ws: bool = True):
        """
        Initialize Hyperliquid API wrapper
        
        Args:
            api_key: API key for trading (optional for data-only access)
            secret_key: Secret key for trading (optional for data-only access)
            base_url: API base URL (mainnet or testnet)
            skip_ws: Skip WebSocket initialization for data-only access
        """
        self.base_url = base_url
        self.api_key = api_key
        self.secret_key = secret_key
        
        # Initialize Info client for market data
        self.info = Info(base_url=base_url, skip_ws=skip_ws)
        
        # Initialize Exchange client for trading (if credentials provided)
        self.exchange = None
        if api_key and secret_key:
            try:
                # Create wallet from private key
                wallet = Account.from_key(secret_key)
                self.exchange = Exchange(wallet=wallet, base_url=base_url)
                logger.info("Trading client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize trading client: {e}")
                logger.info("Running in data-only mode")
        
        logger.info(f"Hyperliquid API wrapper initialized for {base_url}")
    
    def get_current_prices(self) -> Dict[str, float]:
        """
        Get current mid prices for all symbols
        
        Returns:
            Dictionary mapping symbol names to current prices
        """
        try:
            prices = self.info.all_mids()
            # Convert string prices to float
            return {symbol: float(price) for symbol, price in prices.items()}
        except Exception as e:
            logger.error(f"Error fetching current prices: {e}")
            return {}
    
    def get_historical_data(self, symbol: str, interval: str = "1h", 
                          start_time: Optional[int] = None, 
                          end_time: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Get historical candle data for a symbol
        
        Args:
            symbol: Trading symbol (e.g., 'BTC')
            interval: Candle interval ('1m', '5m', '15m', '1h', '4h', '1d')
            start_time: Start timestamp in milliseconds (optional)
            end_time: End timestamp in milliseconds (optional)
            
        Returns:
            DataFrame with standardized columns: ['Open', 'High', 'Low', 'Close', 'Volume']
        """
        try:
            # Set default time range if not provided
            if end_time is None:
                end_time = int(datetime.now().timestamp() * 1000)
            if start_time is None:
                # Default to 30 days of data
                start_time = end_time - (30 * 24 * 60 * 60 * 1000)
            
            # Fetch candle data from Hyperliquid
            candles = self.info.candles_snapshot(symbol, interval, start_time, end_time)
            
            if not candles:
                logger.warning(f"No candle data found for {symbol}")
                return None
            
            # Convert to DataFrame with standardized format
            df = self._convert_candles_to_dataframe(candles)
            
            logger.info(f"Retrieved {len(df)} candles for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return None
    
    def _convert_candles_to_dataframe(self, candles: List[Dict]) -> pd.DataFrame:
        """
        Convert Hyperliquid candle format to pandas DataFrame
        
        Args:
            candles: List of candle dictionaries from Hyperliquid API
            
        Returns:
            DataFrame with standardized columns
        """
        # Extract data from Hyperliquid format
        data = []
        for candle in candles:
            data.append({
                'timestamp': candle['T'],  # End timestamp in milliseconds
                'Open': float(candle['o']),
                'High': float(candle['h']),
                'Low': float(candle['l']),
                'Close': float(candle['c']),
                'Volume': float(candle['v']),
                'symbol': candle['s'],
                'interval': candle['i'],
                'trades': candle['n']
            })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Convert timestamp to datetime
        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('Date', inplace=True)
        
        # Select and reorder standard columns
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Sort by timestamp
        df.sort_index(inplace=True)
        
        return df
    
    def get_user_state(self, address: str) -> Optional[Dict]:
        """
        Get user account state including positions and balance
        
        Args:
            address: User wallet address
            
        Returns:
            Dictionary with user state information
        """
        try:
            state = self.info.user_state(address)
            return state
        except Exception as e:
            logger.error(f"Error fetching user state: {e}")
            return None
    
    def get_user_positions(self, address: str) -> List[Dict]:
        """
        Get user's current positions
        
        Args:
            address: User wallet address
            
        Returns:
            List of position dictionaries
        """
        try:
            state = self.info.user_state(address)
            if state and 'assetPositions' in state:
                return state['assetPositions']
            return []
        except Exception as e:
            logger.error(f"Error fetching user positions: {e}")
            return []
    
    def get_account_balance(self, address: str) -> Optional[float]:
        """
        Get user's account balance
        
        Args:
            address: User wallet address
            
        Returns:
            Account balance as float
        """
        try:
            state = self.info.user_state(address)
            if state and 'marginSummary' in state:
                return float(state['marginSummary']['accountValue'])
            return None
        except Exception as e:
            logger.error(f"Error fetching account balance: {e}")
            return None
    
    def place_market_order(self, symbol: str, is_buy: bool, size: float, 
                          slippage: float = 0.05) -> Optional[Dict]:
        """
        Place a market order
        
        Args:
            symbol: Trading symbol
            is_buy: True for buy, False for sell
            size: Order size
            slippage: Maximum slippage tolerance
            
        Returns:
            Order response dictionary or None if failed
        """
        if not self.exchange:
            logger.error("Trading client not initialized")
            return None
        
        try:
            result = self.exchange.market_open(symbol, is_buy, size, slippage=slippage)
            logger.info(f"Market order placed: {symbol} {'BUY' if is_buy else 'SELL'} {size}")
            return result
        except Exception as e:
            logger.error(f"Error placing market order: {e}")
            return None
    
    def close_position(self, symbol: str, size: Optional[float] = None,
                      slippage: float = 0.05) -> Optional[Dict]:
        """
        Close a position
        
        Args:
            symbol: Trading symbol
            size: Position size to close (None for entire position)
            slippage: Maximum slippage tolerance
            
        Returns:
            Order response dictionary or None if failed
        """
        if not self.exchange:
            logger.error("Trading client not initialized")
            return None
        
        try:
            result = self.exchange.market_close(symbol, size, slippage=slippage)
            logger.info(f"Position closed: {symbol} {size if size else 'ALL'}")
            return result
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return None
    
    def get_available_symbols(self) -> List[str]:
        """
        Get list of available trading symbols
        
        Returns:
            List of symbol names
        """
        try:
            # Get current prices to see available symbols
            prices = self.get_current_prices()
            return list(prices.keys())
        except Exception as e:
            logger.error(f"Error fetching available symbols: {e}")
            return []
    
    def test_connection(self) -> bool:
        """
        Test API connection
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            prices = self.get_current_prices()
            return len(prices) > 0
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False


# Example usage and testing
if __name__ == "__main__":
    # Initialize API wrapper (data-only mode)
    api = HyperliquidAPI()
    
    # Test connection
    if api.test_connection():
        print("✅ API connection successful")
        
        # Get current prices
        prices = api.get_current_prices()
        print(f"Available symbols: {len(prices)}")
        print(f"BTC price: ${prices.get('BTC', 'N/A')}")
        
        # Get historical data
        btc_data = api.get_historical_data('BTC', interval='1h')
        if btc_data is not None:
            print(f"BTC historical data: {len(btc_data)} candles")
            print(btc_data.head())
    else:
        print("❌ API connection failed") 