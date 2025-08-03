"""
Execution module for Gaussian Channel Trading Bot.

Contains order execution and position management components.
"""

from .position_manager import PositionManager, Position, AccountState
from .executor import OrderExecutor, OrderResult, OrderRequest

__all__ = ['PositionManager', 'Position', 'AccountState', 'OrderExecutor', 'OrderResult', 'OrderRequest'] 