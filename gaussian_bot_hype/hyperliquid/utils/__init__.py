"""
Hyperliquid SDK Utilities

This module contains utility classes and constants for the Hyperliquid SDK.
"""

from .constants import MAINNET_API_URL, TESTNET_API_URL, LOCAL_API_URL
from .error import Error, ClientError, ServerError
from .types import (
    Meta, 
    SpotMeta, 
    Subscription, 
    Cloid,
    BuilderInfo,
    OrderRequest,
    OrderType
)
from .signing import (
    get_timestamp_ms,
    float_to_wire,
    order_request_to_order_wire,
    sign_l1_action
)

# Export main utilities
__all__ = [
    # Constants
    'MAINNET_API_URL',
    'TESTNET_API_URL', 
    'LOCAL_API_URL',
    
    # Error classes
    'Error',
    'ClientError',
    'ServerError',
    
    # Types
    'Meta',
    'SpotMeta', 
    'Subscription',
    'Cloid',
    'BuilderInfo',
    'OrderRequest',
    'OrderType',
    
    # Signing utilities
    'get_timestamp_ms',
    'float_to_wire',
    'order_request_to_order_wire',
    'sign_l1_action'
]
