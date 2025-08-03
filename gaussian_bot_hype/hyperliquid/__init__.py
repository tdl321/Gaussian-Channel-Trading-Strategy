"""
Hyperliquid SDK for Python

This module provides a complete interface to the Hyperliquid exchange API,
including market data, trading execution, and real-time data streaming.
"""

# Import main classes using relative imports
from .api import API
from .info import Info
from .exchange import Exchange
from .websocket_manager import WebsocketManager

# Export main classes
__all__ = [
    'API',
    'Info', 
    'Exchange',
    'WebsocketManager'
]

# Version info
__version__ = '1.0.0'
