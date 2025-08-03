#!/usr/bin/env python3
"""
Configuration settings for the Gaussian Channel Trading Bot
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """
    Configuration class for the Gaussian Channel Trading Bot
    """
    
    def __init__(self, config_path=None):
        """Initialize configuration with environment variables and defaults"""
        
        # === HYPERLIQUID API CONFIGURATION ===
        self.HYPERLIQUID_API_KEY = os.getenv('HYPERLIQUID_API_KEY', '')
        self.HYPERLIQUID_SECRET_KEY = os.getenv('HYPERLIQUID_SECRET_KEY', '')
        self.HYPERLIQUID_BASE_URL = os.getenv('HYPERLIQUID_BASE_URL', 'https://api.hyperliquid.xyz')
        self.HYPERLIQUID_WS_URL = os.getenv('HYPERLIQUID_WS_URL', 'wss://api.hyperliquid.xyz/ws')
        
        # === TRADING PARAMETERS ===
        self.SYMBOL = os.getenv('TRADING_SYMBOL', 'BTC-PERP')  # Default to Bitcoin perpetual
        self.INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', '10000'))
        self.POSITION_SIZE_PCT = float(os.getenv('POSITION_SIZE_PCT', '0.65'))  # 65% of equity per trade
        self.MAX_LEVERAGE = float(os.getenv('MAX_LEVERAGE', '5.0'))
        self.MARGIN_REQUIREMENT = float(os.getenv('MARGIN_REQUIREMENT', '0.2'))  # 20%
        self.MAINTENANCE_MARGIN_PCT = float(os.getenv('MAINTENANCE_MARGIN_PCT', '0.75'))  # 75%
        
        # === GAUSSIAN CHANNEL STRATEGY PARAMETERS ===
        self.POLES = int(os.getenv('GAUSSIAN_POLES', '4'))
        self.PERIOD = int(os.getenv('GAUSSIAN_PERIOD', '144'))
        self.MULTIPLIER = float(os.getenv('GAUSSIAN_MULTIPLIER', '1.414'))
        self.MODE_LAG = os.getenv('GAUSSIAN_MODE_LAG', 'false').lower() == 'true'
        self.MODE_FAST = os.getenv('GAUSSIAN_MODE_FAST', 'false').lower() == 'true'
        
        # === ENTRY/EXIT PARAMETERS ===
        self.ATR_SPACING = float(os.getenv('ATR_SPACING', '0.4'))
        self.MAX_PYRAMIDS = int(os.getenv('MAX_PYRAMIDS', '5'))
        self.SMA_LENGTH = int(os.getenv('SMA_LENGTH', '200'))
        self.ENABLE_SMA_FILTER = os.getenv('ENABLE_SMA_FILTER', 'false').lower() == 'true'
        
        # === EXECUTION PARAMETERS ===
        self.COMMISSION_PCT = float(os.getenv('COMMISSION_PCT', '0.001'))  # 0.1%
        self.SLIPPAGE_TICKS = int(os.getenv('SLIPPAGE_TICKS', '1'))
        self.SLIPPAGE_PER_TICK = float(os.getenv('SLIPPAGE_PER_TICK', '0.0001'))  # 0.01%
        
        # === TIMING PARAMETERS ===
        self.TRADING_INTERVAL = int(os.getenv('TRADING_INTERVAL', '86400'))  # 24 hours (daily)
        self.DATA_RETRY_DELAY = int(os.getenv('DATA_RETRY_DELAY', '5'))  # 5 seconds
        self.ERROR_RETRY_DELAY = int(os.getenv('ERROR_RETRY_DELAY', '30'))  # 30 seconds
        
        # === DATE RANGE FILTERS ===
        self.START_DATE = os.getenv('START_DATE', '2018-01-01')
        self.END_DATE = os.getenv('END_DATE', '2069-12-31')
        
        # === DIRECTORY PATHS ===
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.DATA_DIR = os.path.join(self.BASE_DIR, 'data')
        self.LOGS_DIR = os.path.join(self.BASE_DIR, 'logs')
        self.RESULTS_DIR = os.path.join(self.BASE_DIR, 'results')
        
        # Create directories if they don't exist
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.LOGS_DIR, exist_ok=True)
        os.makedirs(self.RESULTS_DIR, exist_ok=True)
        
        # === LOGGING CONFIGURATION ===
        self.LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
        self.LOG_FILE = os.path.join(self.LOGS_DIR, 'trading.log')
        self.TRADE_LOG_FILE = os.path.join(self.LOGS_DIR, 'trades.csv')
        self.PERFORMANCE_LOG_FILE = os.path.join(self.LOGS_DIR, 'performance.csv')
        
        # === RISK MANAGEMENT ===
        self.MAX_DRAWDOWN_PCT = float(os.getenv('MAX_DRAWDOWN_PCT', '20.0'))  # 20%
        self.DAILY_LOSS_LIMIT = float(os.getenv('DAILY_LOSS_LIMIT', '5.0'))  # 5% of equity
        self.FORCED_LIQUIDATION_BUFFER = float(os.getenv('FORCED_LIQUIDATION_BUFFER', '0.05'))  # 5%
        
        # === BACKTESTING PARAMETERS ===
        self.BACKTEST_START_DATE = os.getenv('BACKTEST_START_DATE', '2020-01-01')
        self.BACKTEST_END_DATE = os.getenv('BACKTEST_END_DATE', '2024-01-01')
        self.BACKTEST_INITIAL_CAPITAL = float(os.getenv('BACKTEST_INITIAL_CAPITAL', '10000'))
        
        # === RENDER DEPLOYMENT ===
        self.IS_RENDER_DEPLOYMENT = os.getenv('IS_RENDER_DEPLOYMENT', 'false').lower() == 'true'
        self.RENDER_SERVICE_URL = os.getenv('RENDER_SERVICE_URL', '')
        
        # === DEBUGGING ===
        self.DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
        self.VERBOSE_LOGGING = os.getenv('VERBOSE_LOGGING', 'true').lower() == 'true'
        
        # Validate critical configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate critical configuration parameters"""
        errors = []
        
        # Check API credentials
        if not self.HYPERLIQUID_API_KEY:
            errors.append("HYPERLIQUID_API_KEY is required")
        if not self.HYPERLIQUID_SECRET_KEY:
            errors.append("HYPERLIQUID_SECRET_KEY is required")
        
        # Check trading parameters
        if self.POSITION_SIZE_PCT <= 0 or self.POSITION_SIZE_PCT > 1:
            errors.append("POSITION_SIZE_PCT must be between 0 and 1")
        if self.MAX_LEVERAGE <= 0:
            errors.append("MAX_LEVERAGE must be positive")
        if self.INITIAL_CAPITAL <= 0:
            errors.append("INITIAL_CAPITAL must be positive")
        
        # Check strategy parameters
        if self.POLES < 1 or self.POLES > 9:
            errors.append("GAUSSIAN_POLES must be between 1 and 9")
        if self.PERIOD < 2:
            errors.append("GAUSSIAN_PERIOD must be at least 2")
        if self.MULTIPLIER <= 0:
            errors.append("GAUSSIAN_MULTIPLIER must be positive")
        
        # Check execution parameters
        if self.COMMISSION_PCT < 0:
            errors.append("COMMISSION_PCT must be non-negative")
        if self.SLIPPAGE_TICKS < 0:
            errors.append("SLIPPAGE_TICKS must be non-negative")
        
        # Check timing parameters
        if self.TRADING_INTERVAL <= 0:
            errors.append("TRADING_INTERVAL must be positive")
        
        if errors:
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors))
    
    def get_api_config(self):
        """Get API configuration as a dictionary"""
        return {
            'api_key': self.HYPERLIQUID_API_KEY,
            'secret_key': self.HYPERLIQUID_SECRET_KEY,
            'base_url': self.HYPERLIQUID_BASE_URL,
            'ws_url': self.HYPERLIQUID_WS_URL
        }
    
    def get_strategy_config(self):
        """Get strategy configuration as a dictionary"""
        return {
            'poles': self.POLES,
            'period': self.PERIOD,
            'multiplier': self.MULTIPLIER,
            'mode_lag': self.MODE_LAG,
            'mode_fast': self.MODE_FAST,
            'atr_spacing': self.ATR_SPACING,
            'max_pyramids': self.MAX_PYRAMIDS,
            'sma_length': self.SMA_LENGTH,
            'enable_sma_filter': self.ENABLE_SMA_FILTER,
            'position_size_pct': self.POSITION_SIZE_PCT,
            'start_date': self.START_DATE,
            'end_date': self.END_DATE
        }
    
    def get_execution_config(self):
        """Get execution configuration as a dictionary"""
        return {
            'initial_capital': self.INITIAL_CAPITAL,
            'commission_pct': self.COMMISSION_PCT,
            'slippage_ticks': self.SLIPPAGE_TICKS,
            'slippage_per_tick': self.SLIPPAGE_PER_TICK,
            'margin_requirement': self.MARGIN_REQUIREMENT,
            'maintenance_margin_pct': self.MAINTENANCE_MARGIN_PCT,
            'max_leverage': self.MAX_LEVERAGE,
            'forced_liquidation_buffer': self.FORCED_LIQUIDATION_BUFFER
        }
    
    def print_config(self):
        """Print current configuration (without sensitive data)"""
        print("="*50)
        print("GAUSSIAN CHANNEL BOT CONFIGURATION")
        print("="*50)
        print(f"Symbol: {self.SYMBOL}")
        print(f"Initial Capital: ${self.INITIAL_CAPITAL:,.2f}")
        print(f"Position Size: {self.POSITION_SIZE_PCT*100:.1f}% of equity")
        print(f"Max Leverage: {self.MAX_LEVERAGE}x")
        print(f"Commission: {self.COMMISSION_PCT*100:.3f}%")
        print(f"Slippage: {self.SLIPPAGE_TICKS} ticks")
        print()
        print("Strategy Parameters:")
        print(f"  Poles: {self.POLES}")
        print(f"  Period: {self.PERIOD}")
        print(f"  Multiplier: {self.MULTIPLIER}")
        print(f"  Mode Lag: {self.MODE_LAG}")
        print(f"  Mode Fast: {self.MODE_FAST}")
        print(f"  ATR Spacing: {self.ATR_SPACING}")
        print(f"  Max Pyramids: {self.MAX_PYRAMIDS}")
        print()
        print("Risk Management:")
        print(f"  Max Drawdown: {self.MAX_DRAWDOWN_PCT}%")
        print(f"  Daily Loss Limit: {self.DAILY_LOSS_LIMIT}%")
        print(f"  Margin Requirement: {self.MARGIN_REQUIREMENT*100}%")
        print(f"  Maintenance Margin: {self.MAINTENANCE_MARGIN_PCT*100}%")
        print()
        print("Timing:")
        print(f"  Trading Interval: {self.TRADING_INTERVAL} seconds")
        print(f"  Date Range: {self.START_DATE} to {self.END_DATE}")
        print()
        print("Directories:")
        print(f"  Data: {self.DATA_DIR}")
        print(f"  Logs: {self.LOGS_DIR}")
        print(f"  Results: {self.RESULTS_DIR}")
        print("="*50)


# Example usage
if __name__ == "__main__":
    config = Config()
    config.print_config() 