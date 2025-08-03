"""
Order Executor for Gaussian Channel Trading Bot

Handles order placement, confirmation, and risk management
using Hyperliquid SDK patterns.
"""

import logging
import time
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime

from config import Config
from hyperliquid_api import HyperliquidAPI
from .position_manager import PositionManager


@dataclass
class OrderResult:
    """Order execution result structure"""
    success: bool
    order_id: Optional[str] = None
    filled_size: Optional[float] = None
    avg_price: Optional[float] = None
    error: Optional[str] = None
    timestamp: Optional[datetime] = None


@dataclass
class OrderRequest:
    """Order request structure"""
    symbol: str
    side: str  # "buy" or "sell"
    size: float
    order_type: str = "market"  # "market" or "limit"
    limit_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class OrderExecutor:
    """
    Manages order execution and risk management for the trading bot.
    
    Follows Hyperliquid SDK patterns for order placement and
    provides comprehensive error handling and retry logic.
    """
    
    def __init__(self, api: HyperliquidAPI, config: Config, position_manager: PositionManager):
        """
        Initialize the order executor.
        
        Args:
            api: HyperliquidAPI wrapper instance
            config: Configuration object with risk parameters
            position_manager: Position manager for risk checks
        """
        self.api = api
        self.config = config
        self.position_manager = position_manager
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        # Order tracking
        self.pending_orders: Dict[str, OrderRequest] = {}
        self.order_history: Dict[str, OrderResult] = {}
        
        # Risk parameters from config
        self.max_leverage = config.MAX_LEVERAGE
        self.position_size_pct = config.POSITION_SIZE_PCT
        self.max_drawdown_pct = config.MAX_DRAWDOWN_PCT
        self.daily_loss_limit = config.DAILY_LOSS_LIMIT
        
        # Execution parameters
        self.max_retries = 3
        self.retry_delay = 1.0  # seconds
        self.slippage_tolerance = 0.001  # 0.1%
        
        self.logger.info("OrderExecutor initialized")
    
    def calculate_position_size(self, signal_strength: float, account_value: float) -> float:
        """
        Calculate position size based on signal strength and risk parameters.
        
        Args:
            signal_strength: Signal strength (0.0 to 1.0)
            account_value: Current account value
            
        Returns:
            Position size in USD
        """
        # Base position size as percentage of account value
        base_size = account_value * self.position_size_pct
        
        # Adjust for signal strength
        adjusted_size = base_size * signal_strength
        
        # Apply minimum and maximum constraints
        min_size = 10.0  # Minimum $10 position
        max_size = account_value * 0.8  # Maximum 80% of account value
        
        position_size = max(min_size, min(adjusted_size, max_size))
        
        self.logger.info(f"Calculated position size: ${position_size:,.2f} "
                        f"(signal: {signal_strength:.2f}, account: ${account_value:,.2f})")
        
        return position_size
    
    def place_market_order(self, symbol: str, side: str, size: float) -> OrderResult:
        """
        Place a market order following Hyperliquid SDK patterns.
        
        Args:
            symbol: Trading symbol (e.g., "BTC", "ETH")
            side: Order side ("buy" or "sell")
            size: Order size in USD
            
        Returns:
            OrderResult with execution details
        """
        try:
            # Validate inputs
            if side.lower() not in ["buy", "sell"]:
                return OrderResult(
                    success=False,
                    error=f"Invalid side: {side}. Must be 'buy' or 'sell'",
                    timestamp=datetime.now()
                )
            
            if size <= 0:
                return OrderResult(
                    success=False,
                    error=f"Invalid size: {size}. Must be positive",
                    timestamp=datetime.now()
                )
            
            # Log order attempt
            self.logger.info(f"Placing market {side} order: {symbol} ${size:,.2f}")
            
            # Convert side to boolean for SDK (True = buy, False = sell)
            is_buy = side.lower() == "buy"
            
            # Place order following SDK pattern from basic_market_order.py
            order_result = self.api.market_open(symbol, is_buy, size, None, self.slippage_tolerance)
            
            # Process order result following SDK pattern
            if order_result["status"] == "ok":
                for status in order_result["response"]["data"]["statuses"]:
                    try:
                        filled = status["filled"]
                        result = OrderResult(
                            success=True,
                            order_id=filled["oid"],
                            filled_size=float(filled["totalSz"]),
                            avg_price=float(filled["avgPx"]),
                            timestamp=datetime.now()
                        )
                        
                        self.logger.info(f"Order filled: {symbol} {side} {result.filled_size} @ ${result.avg_price}")
                        
                        # Update position manager
                        self.position_manager.update_positions()
                        
                        return result
                        
                    except KeyError:
                        # Order failed
                        error_msg = status.get("error", "Unknown error")
                        result = OrderResult(
                            success=False,
                            error=error_msg,
                            timestamp=datetime.now()
                        )
                        
                        self.logger.error(f"Order failed: {symbol} {side} - {error_msg}")
                        return result
            
            # Order failed at API level
            error_msg = order_result.get("error", "Order failed")
            result = OrderResult(
                success=False,
                error=error_msg,
                timestamp=datetime.now()
            )
            
            self.logger.error(f"Order failed: {symbol} {side} - {error_msg}")
            return result
            
        except Exception as e:
            self.logger.error(f"Exception placing market order: {e}")
            return OrderResult(
                success=False,
                error=str(e),
                timestamp=datetime.now()
            )
    
    def place_market_order_with_retry(self, symbol: str, side: str, size: float) -> OrderResult:
        """
        Place a market order with retry logic for resilience.
        
        Args:
            symbol: Trading symbol
            side: Order side ("buy" or "sell")
            size: Order size in USD
            
        Returns:
            OrderResult with execution details
        """
        for attempt in range(self.max_retries):
            try:
                result = self.place_market_order(symbol, side, size)
                
                if result.success:
                    return result
                
                # If order failed, wait before retry
                if attempt < self.max_retries - 1:
                    self.logger.warning(f"Order failed, retrying in {self.retry_delay}s (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(self.retry_delay)
                
            except Exception as e:
                self.logger.error(f"Exception on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        
        # All retries failed
        return OrderResult(
            success=False,
            error=f"All {self.max_retries} attempts failed",
            timestamp=datetime.now()
        )
    
    def close_position(self, symbol: str) -> OrderResult:
        """
        Close all positions for a symbol using market order.
        
        Args:
            symbol: Trading symbol to close
            
        Returns:
            OrderResult with execution details
        """
        try:
            # Check if position exists
            position = self.position_manager.get_position(symbol)
            if not position or position.size == 0:
                return OrderResult(
                    success=True,
                    error="No position to close",
                    timestamp=datetime.now()
                )
            
            self.logger.info(f"Closing position: {symbol} {position.size}")
            
            # Close position following SDK pattern from basic_market_order.py
            order_result = self.api.market_close(symbol)
            
            # Process order result
            if order_result["status"] == "ok":
                for status in order_result["response"]["data"]["statuses"]:
                    try:
                        filled = status["filled"]
                        result = OrderResult(
                            success=True,
                            order_id=filled["oid"],
                            filled_size=float(filled["totalSz"]),
                            avg_price=float(filled["avgPx"]),
                            timestamp=datetime.now()
                        )
                        
                        self.logger.info(f"Position closed: {symbol} {result.filled_size} @ ${result.avg_price}")
                        
                        # Update position manager
                        self.position_manager.update_positions()
                        
                        return result
                        
                    except KeyError:
                        error_msg = status.get("error", "Unknown error")
                        result = OrderResult(
                            success=False,
                            error=error_msg,
                            timestamp=datetime.now()
                        )
                        
                        self.logger.error(f"Close failed: {symbol} - {error_msg}")
                        return result
            
            # Close failed at API level
            error_msg = order_result.get("error", "Close failed")
            result = OrderResult(
                success=False,
                error=error_msg,
                timestamp=datetime.now()
            )
            
            self.logger.error(f"Close failed: {symbol} - {error_msg}")
            return result
            
        except Exception as e:
            self.logger.error(f"Exception closing position: {e}")
            return OrderResult(
                success=False,
                error=str(e),
                timestamp=datetime.now()
            )
    
    def execute_signal(self, symbol: str, signal: str, signal_strength: float = 1.0) -> OrderResult:
        """
        Execute a trading signal with risk management.
        
        Args:
            symbol: Trading symbol
            signal: Signal type ("buy", "sell", "close")
            signal_strength: Signal strength (0.0 to 1.0)
            
        Returns:
            OrderResult with execution details
        """
        try:
            # Update position data
            self.position_manager.update_positions()
            
            # Get current account value
            account_value = self.position_manager.get_account_value()
            if account_value <= 0:
                return OrderResult(
                    success=False,
                    error="Insufficient account value",
                    timestamp=datetime.now()
                )
            
            # Check risk limits
            if not self._check_risk_limits(symbol):
                return OrderResult(
                    success=False,
                    error="Risk limits exceeded",
                    timestamp=datetime.now()
                )
            
            # Execute based on signal type
            if signal.lower() == "buy":
                # Calculate position size
                position_size = self.calculate_position_size(signal_strength, account_value)
                
                # Check if we already have a position
                current_position = self.position_manager.get_position(symbol)
                if current_position and current_position.size > 0:
                    self.logger.info(f"Already have long position in {symbol}, skipping buy signal")
                    return OrderResult(
                        success=True,
                        error="Already have long position",
                        timestamp=datetime.now()
                    )
                
                # Place buy order
                return self.place_market_order_with_retry(symbol, "buy", position_size)
                
            elif signal.lower() == "sell":
                # Close existing position
                return self.close_position(symbol)
                
            elif signal.lower() == "close":
                # Close existing position
                return self.close_position(symbol)
                
            else:
                return OrderResult(
                    success=False,
                    error=f"Invalid signal: {signal}",
                    timestamp=datetime.now()
                )
                
        except Exception as e:
            self.logger.error(f"Exception executing signal: {e}")
            return OrderResult(
                success=False,
                error=str(e),
                timestamp=datetime.now()
            )
    
    def _check_risk_limits(self, symbol: str) -> bool:
        """
        Check if order would exceed risk limits.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if order is within risk limits
        """
        try:
            # Check margin ratio
            margin_ratio = self.position_manager.calculate_margin_ratio()
            if margin_ratio > 0.8:  # 80% margin ratio limit
                self.logger.warning(f"High margin ratio ({margin_ratio:.2%}) - rejecting order")
                return False
            
            # Check daily loss limit
            total_pnl = self.position_manager.get_total_unrealized_pnl()
            account_value = self.position_manager.get_account_value()
            if account_value > 0 and total_pnl < -(account_value * self.daily_loss_limit):
                self.logger.warning(f"Daily loss limit exceeded - rejecting order")
                return False
            
            # Check liquidation risk
            is_at_risk, risk_percentage = self.position_manager.check_liquidation_risk(symbol)
            if is_at_risk:
                self.logger.warning(f"High liquidation risk for {symbol} ({risk_percentage:.1f}%) - rejecting order")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")
            return False  # Fail safe - reject order if risk check fails
    
    def cancel_all_orders(self, symbol: str) -> bool:
        """
        Cancel all pending orders for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if cancellation was successful
        """
        try:
            # Get open orders (following SDK pattern from basic_order.py)
            open_orders = self.api.open_orders()
            
            cancelled_count = 0
            for order in open_orders:
                if order["coin"] == symbol:
                    cancel_result = self.api.cancel(symbol, order["oid"])
                    if cancel_result["status"] == "ok":
                        cancelled_count += 1
                        self.logger.info(f"Cancelled order: {symbol} {order['oid']}")
                    else:
                        self.logger.error(f"Failed to cancel order: {symbol} {order['oid']}")
            
            self.logger.info(f"Cancelled {cancelled_count} orders for {symbol}")
            return cancelled_count > 0
            
        except Exception as e:
            self.logger.error(f"Exception cancelling orders: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> Dict:
        """
        Get status of a specific order.
        
        Args:
            order_id: Order ID to check
            
        Returns:
            Order status dictionary
        """
        try:
            # Query order by ID (following SDK pattern from basic_order.py)
            order_status = self.api.query_order_by_oid(order_id)
            return order_status
            
        except Exception as e:
            self.logger.error(f"Error getting order status: {e}")
            return {"error": str(e)}
    
    def log_execution_summary(self) -> None:
        """Log execution summary for monitoring."""
        summary = self.position_manager.get_position_summary()
        
        self.logger.info(f"Execution Summary:")
        self.logger.info(f"  Account Value: ${summary['account_value']:,.2f}")
        self.logger.info(f"  Total Exposure: ${summary['total_exposure']:,.2f}")
        self.logger.info(f"  Margin Ratio: {summary['margin_ratio']:.2%}")
        self.logger.info(f"  Unrealized P&L: ${summary['total_unrealized_pnl']:,.2f}")
        self.logger.info(f"  Active Positions: {summary['position_count']}")
        
        # Log recent orders
        recent_orders = [order for order in self.order_history.values() 
                        if order.timestamp and (datetime.now() - order.timestamp).seconds < 3600]
        
        if recent_orders:
            self.logger.info(f"  Recent Orders (last hour): {len(recent_orders)}")
            for order in recent_orders[-5:]:  # Last 5 orders
                status = "SUCCESS" if order.success else "FAILED"
                self.logger.info(f"    {status}: {order.order_id or 'N/A'} - {order.error or 'Executed'}") 