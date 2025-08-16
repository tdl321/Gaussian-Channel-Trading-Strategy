#!/usr/bin/env python3
"""
Minimal Configuration for Gaussian Channel Trading Bot
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# === HYPERLIQUID API CONFIGURATION ===
HYPERLIQUID_API_KEY = os.getenv('HYPERLIQUID_API_KEY', '')
HYPERLIQUID_SECRET_KEY = os.getenv('HYPERLIQUID_SECRET_KEY', '')
HYPERLIQUID_BASE_URL = os.getenv('HYPERLIQUID_BASE_URL', 'https://api.hyperliquid.xyz')

# === TRADING PARAMETERS ===
SYMBOL = os.getenv('TRADING_SYMBOL', 'BTC')
LEVERAGE = int(os.getenv('LEVERAGE', '5'))

# === GAUSSIAN CHANNEL STRATEGY PARAMETERS ===
# Optimized parameters from Bayesian optimization
POLES = int(os.getenv('GAUSSIAN_POLES', '5'))
PERIOD = int(os.getenv('GAUSSIAN_PERIOD', '135'))
MULTIPLIER = float(os.getenv('GAUSSIAN_MULTIPLIER', '2.859'))

# === TIMING PARAMETERS ===
TRADING_INTERVAL = int(os.getenv('TRADING_INTERVAL', '3600'))   # 1 hour

# === LOGGING ===
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

def validate_config():
    """Validate essential configuration parameters"""
    errors = []
    
    # Check API credentials
    if not HYPERLIQUID_API_KEY:
        errors.append("HYPERLIQUID_API_KEY is required")
    if not HYPERLIQUID_SECRET_KEY:
        errors.append("HYPERLIQUID_SECRET_KEY is required")
    
    # Check trading parameters
    if LEVERAGE <= 0:
        errors.append("LEVERAGE must be positive")
    
    # Check strategy parameters
    if POLES < 1 or POLES > 9:
        errors.append("GAUSSIAN_POLES must be between 1 and 9")
    if PERIOD < 2:
        errors.append("GAUSSIAN_PERIOD must be at least 2")
    if MULTIPLIER <= 0:
        errors.append("GAUSSIAN_MULTIPLIER must be positive")
    
    if errors:
        raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors))

def print_config():
    """Print current configuration"""
    print("="*50)
    print("GAUSSIAN CHANNEL BOT CONFIGURATION")
    print("="*50)
    print(f"Symbol: {SYMBOL}")
    print(f"Leverage: {LEVERAGE}x")
    print()
    print("Strategy Parameters:")
    print(f"  Poles: {POLES}")
    print(f"  Period: {PERIOD}")
    print(f"  Multiplier: {MULTIPLIER}")
    print()
    print(f"Trading Interval: {TRADING_INTERVAL} seconds ({TRADING_INTERVAL//3600} hour{'s' if TRADING_INTERVAL//3600 != 1 else ''})")
    print("="*50)

# Validate configuration on import
validate_config()

# Example usage
if __name__ == "__main__":
    print_config() 