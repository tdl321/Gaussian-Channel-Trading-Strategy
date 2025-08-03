"""
Position Manager for Gaussian Channel Trading Bot

Handles position tracking, margin monitoring, and risk management
using Hyperliquid SDK patterns.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from eth_account.signers.local import LocalAccount

from config import Config


@dataclass
class Position:
    """Position data structure matching Hyperliquid API format"""
    symbol: str
    size: float  # Positive = long, negative = short
    entry_price: float
    unrealized_pnl: float
    margin_used: float
    liquidation_price: float
    leverage: Dict[str, any]
    position_value: float
    return_on_equity: float
    last_updated: datetime


@dataclass
class AccountState:
    """Account state data structure"""
    account_value: float
    total_margin_used: float
    total_notional_position: float
    total_raw_usd: float
    last_updated: datetime


class PositionManager:
    """
    Manages position tracking and risk monitoring for the trading bot.
    
    Follows Hyperliquid SDK patterns for position data retrieval and
    provides real-time position tracking with risk management.
    """
    
    def __init__(self, config: Config, info: Info, exchange: Exchange, account: LocalAccount):
        """
        Initialize the position manager.
        
        Args:
            config: Configuration object with risk parameters
            info: Hyperliquid Info instance for data retrieval
            exchange: Hyperliquid Exchange instance for trading
            account: Ethereum account for authentication
        """
        self.config = config
        self.info = info
        self.exchange = exchange
        self.account = account
        self.address = account.address
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        # Position tracking
        self.positions: Dict[str, Position] = {}
        self.account_state: Optional[AccountState] = None
        
        # Risk parameters from config
        self.max_leverage = config.MAX_LEVERAGE
        self.max_position_size = config.MAX_POSITION_SIZE
        self.max_drawdown = config.MAX_DRAWDOWN
        self.daily_loss_limit = config.DAILY_LOSS_LIMIT
        
        self.logger.info(f"PositionManager initialized for account: {self.address}")
    
    def update_positions(self) -> Dict[str, Position]:
        """
        Update position data from Hyperliquid API.
        
        Returns:
            Dictionary of current positions by symbol
        """
        try:
            # Get user state following SDK pattern
            user_state = self.info.user_state(self.address)
            
            # Update account state
            self._update_account_state(user_state)
            
            # Parse positions following SDK pattern
            new_positions = {}
            for position_data in user_state["assetPositions"]:
                position = position_data["position"]
                
                # Convert string values to float (following SDK data format)
                pos = Position(
                    symbol=position["coin"],
                    size=float(position["szi"]),
                    entry_price=float(position["entryPx"]),
                    unrealized_pnl=float(position["unrealizedPnl"]),
                    margin_used=float(position["marginUsed"]),
                    liquidation_price=float(position["liquidationPx"]),
                    leverage=position["leverage"],
                    position_value=float(position["positionValue"]),
                    return_on_equity=float(position["returnOnEquity"]),
                    last_updated=datetime.now()
                )
                
                new_positions[pos.symbol] = pos
                
                # Log position updates
                if pos.symbol not in self.positions:
                    self.logger.info(f"New position opened: {pos.symbol} {pos.size} @ {pos.entry_price}")
                else:
                    old_pos = self.positions[pos.symbol]
                    if abs(pos.size - old_pos.size) > 0.001:  # Significant change
                        self.logger.info(f"Position updated: {pos.symbol} {pos.size} (was {old_pos.size})")
            
            # Check for closed positions
            closed_positions = set(self.positions.keys()) - set(new_positions.keys())
            for symbol in closed_positions:
                self.logger.info(f"Position closed: {symbol}")
            
            self.positions = new_positions
            return new_positions
            
        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")
            return self.positions
    
    def _update_account_state(self, user_state: Dict) -> None:
        """Update account state from user state data."""
        try:
            margin_summary = user_state["marginSummary"]
            
            self.account_state = AccountState(
                account_value=float(margin_summary["accountValue"]),
                total_margin_used=float(margin_summary["totalMarginUsed"]),
                total_notional_position=float(margin_summary["totalNtlPos"]),
                total_raw_usd=float(margin_summary["totalRawUsd"]),
                last_updated=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error updating account state: {e}")
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for a specific symbol.
        
        Args:
            symbol: Trading symbol (e.g., "BTC", "ETH")
            
        Returns:
            Position object or None if no position exists
        """
        return self.positions.get(symbol)
    
    def get_total_exposure(self) -> float:
        """
        Get total notional exposure across all positions.
        
        Returns:
            Total notional value of all positions
        """
        if self.account_state:
            return self.account_state.total_notional_position
        return 0.0
    
    def get_account_value(self) -> float:
        """
        Get current account value.
        
        Returns:
            Account value in USD
        """
        if self.account_state:
            return self.account_state.account_value
        return 0.0
    
    def calculate_margin_ratio(self) -> float:
        """
        Calculate current margin ratio.
        
        Returns:
            Margin ratio (margin used / account value)
        """
        if self.account_state and self.account_state.account_value > 0:
            return self.account_state.total_margin_used / self.account_state.account_value
        return 0.0
    
    def check_liquidation_risk(self, symbol: str) -> Tuple[bool, float]:
        """
        Check liquidation risk for a specific position.
        
        Args:
            symbol: Trading symbol to check
            
        Returns:
            Tuple of (is_at_risk, risk_percentage)
        """
        position = self.get_position(symbol)
        if not position:
            return False, 0.0
        
        # Get current price
        try:
            current_price = self._get_current_price(symbol)
            if current_price is None:
                return False, 0.0
            
            # Calculate distance to liquidation
            if position.size > 0:  # Long position
                distance_to_liquidation = (current_price - position.liquidation_price) / current_price
            else:  # Short position
                distance_to_liquidation = (position.liquidation_price - current_price) / current_price
            
            # Consider at risk if within 10% of liquidation
            is_at_risk = distance_to_liquidation < 0.10
            risk_percentage = max(0.0, (0.10 - distance_to_liquidation) * 1000)  # Scale to percentage
            
            return is_at_risk, risk_percentage
            
        except Exception as e:
            self.logger.error(f"Error checking liquidation risk for {symbol}: {e}")
            return False, 0.0
    
    def get_unrealized_pnl(self, symbol: str) -> float:
        """
        Get unrealized P&L for a specific position.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Unrealized P&L in USD
        """
        position = self.get_position(symbol)
        return position.unrealized_pnl if position else 0.0
    
    def get_total_unrealized_pnl(self) -> float:
        """
        Get total unrealized P&L across all positions.
        
        Returns:
            Total unrealized P&L in USD
        """
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    def should_reduce_position(self, symbol: str, risk_level: str = "medium") -> bool:
        """
        Determine if a position should be reduced based on risk parameters.
        
        Args:
            symbol: Trading symbol to check
            risk_level: Risk level ("low", "medium", "high")
            
        Returns:
            True if position should be reduced
        """
        position = self.get_position(symbol)
        if not position:
            return False
        
        # Check margin ratio
        margin_ratio = self.calculate_margin_ratio()
        margin_threshold = {
            "low": 0.5,
            "medium": 0.7,
            "high": 0.8
        }.get(risk_level, 0.7)
        
        if margin_ratio > margin_threshold:
            self.logger.warning(f"High margin ratio ({margin_ratio:.2%}) - consider reducing {symbol}")
            return True
        
        # Check liquidation risk
        is_at_risk, risk_percentage = self.check_liquidation_risk(symbol)
        if is_at_risk:
            self.logger.warning(f"High liquidation risk for {symbol} ({risk_percentage:.1f}%)")
            return True
        
        # Check daily loss limit
        total_pnl = self.get_total_unrealized_pnl()
        account_value = self.get_account_value()
        if account_value > 0 and total_pnl < -(account_value * self.daily_loss_limit):
            self.logger.warning(f"Daily loss limit exceeded - consider reducing positions")
            return True
        
        return False
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current price or None if unavailable
        """
        try:
            # Get all mid prices
            all_mids = self.info.all_mids()
            for mid_data in all_mids:
                if mid_data["coin"] == symbol:
                    return float(mid_data["mid"])
            return None
        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    def get_position_summary(self) -> Dict:
        """
        Get summary of all positions and account state.
        
        Returns:
            Dictionary with position and account summary
        """
        return {
            "account_value": self.get_account_value(),
            "total_margin_used": self.account_state.total_margin_used if self.account_state else 0.0,
            "margin_ratio": self.calculate_margin_ratio(),
            "total_exposure": self.get_total_exposure(),
            "total_unrealized_pnl": self.get_total_unrealized_pnl(),
            "position_count": len(self.positions),
            "positions": {
                symbol: {
                    "size": pos.size,
                    "entry_price": pos.entry_price,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "margin_used": pos.margin_used
                }
                for symbol, pos in self.positions.items()
            },
            "last_updated": self.account_state.last_updated if self.account_state else None
        }
    
    def log_position_summary(self) -> None:
        """Log current position summary for monitoring."""
        summary = self.get_position_summary()
        
        self.logger.info(f"Position Summary:")
        self.logger.info(f"  Account Value: ${summary['account_value']:,.2f}")
        self.logger.info(f"  Total Exposure: ${summary['total_exposure']:,.2f}")
        self.logger.info(f"  Margin Ratio: {summary['margin_ratio']:.2%}")
        self.logger.info(f"  Unrealized P&L: ${summary['total_unrealized_pnl']:,.2f}")
        self.logger.info(f"  Active Positions: {summary['position_count']}")
        
        for symbol, pos_data in summary['positions'].items():
            self.logger.info(f"    {symbol}: {pos_data['size']} @ ${pos_data['entry_price']:.2f} "
                           f"(P&L: ${pos_data['unrealized_pnl']:.2f})") 