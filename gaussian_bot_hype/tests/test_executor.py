"""
Test file for OrderExecutor component.
"""

import unittest
from unittest.mock import Mock, patch
from datetime import datetime

from execution.executor import OrderExecutor, OrderResult, OrderRequest
from execution.position_manager import PositionManager, Position, AccountState


class TestOrderExecutor(unittest.TestCase):
    """Test cases for OrderExecutor."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock configuration
        self.mock_config = Mock()
        self.mock_config.MAX_LEVERAGE = 5.0
        self.mock_config.POSITION_SIZE_PCT = 0.65
        self.mock_config.MAX_DRAWDOWN_PCT = 0.20
        self.mock_config.DAILY_LOSS_LIMIT = 0.05
    
        # Mock HyperliquidAPI
        self.mock_api = Mock()
        
        # Create mock position manager
        self.position_manager = Mock()
    
        # Create order executor
        self.executor = OrderExecutor(
            self.mock_api,
            self.mock_config,
            self.position_manager
        )
    
    def test_calculate_position_size(self):
        """Test position size calculation."""
        # Test with different signal strengths
        account_value = 10000.0
        
        # Full signal strength
        size = self.executor.calculate_position_size(1.0, account_value)
        expected_size = account_value * 0.65  # MAX_POSITION_SIZE
        self.assertAlmostEqual(size, expected_size, places=2)
        
        # Half signal strength
        size = self.executor.calculate_position_size(0.5, account_value)
        expected_size = account_value * 0.65 * 0.5
        self.assertAlmostEqual(size, expected_size, places=2)
        
        # Minimum size constraint
        size = self.executor.calculate_position_size(0.001, account_value)
        self.assertEqual(size, 10.0)  # Minimum size
        
        # Maximum size constraint
        size = self.executor.calculate_position_size(2.0, account_value)
        expected_size = account_value * 0.8  # Maximum 80%
        self.assertAlmostEqual(size, expected_size, places=2)
    
    def test_place_market_order_validation(self):
        """Test market order input validation."""
        # Test invalid side
        result = self.executor.place_market_order("BTC", "invalid", 100.0)
        self.assertFalse(result.success)
        self.assertIn("Invalid side", result.error)
        
        # Test invalid size
        result = self.executor.place_market_order("BTC", "buy", -100.0)
        self.assertFalse(result.success)
        self.assertIn("Invalid size", result.error)
        
        result = self.executor.place_market_order("BTC", "buy", 0.0)
        self.assertFalse(result.success)
        self.assertIn("Invalid size", result.error)
    
    def test_place_market_order_success(self):
        """Test successful market order placement."""
        # Mock successful order response
        mock_order_result = {
            "status": "ok",
            "response": {
                "data": {
                    "statuses": [
                        {
                            "filled": {
                                "oid": "order123",
                                "totalSz": "100.0",
                                "avgPx": "45000.0"
                            }
                        }
                    ]
                }
            }
        }
        
        self.mock_api.market_open.return_value = mock_order_result
        
        # Place order
        result = self.executor.place_market_order("BTC", "buy", 100.0)
        
        # Verify result
        self.assertTrue(result.success)
        self.assertEqual(result.order_id, "order123")
        self.assertEqual(result.filled_size, 100.0)
        self.assertEqual(result.avg_price, 45000.0)
        
        # Verify SDK call
        self.mock_api.market_open.assert_called_once_with("BTC", True, 100.0, None, 0.001)
    
    def test_place_market_order_failure(self):
        """Test failed market order placement."""
        # Mock failed order response
        mock_order_result = {
            "status": "ok",
            "response": {
                "data": {
                    "statuses": [
                        {
                            "error": "Insufficient margin"
                        }
                    ]
                }
            }
        }
        
        self.mock_api.market_open.return_value = mock_order_result
        
        # Place order
        result = self.executor.place_market_order("BTC", "buy", 100.0)
        
        # Verify result
        self.assertFalse(result.success)
        self.assertIn("Insufficient margin", result.error)
    
    def test_close_position_no_position(self):
        """Test closing position when none exists."""
        # Mock no position
        self.position_manager.get_position.return_value = None
        
        result = self.executor.close_position("BTC")
        
        self.assertTrue(result.success)
        self.assertIn("No position to close", result.error)
    
    def test_close_position_success(self):
        """Test successful position closure."""
        # Mock existing position
        mock_position = Position(
            symbol="BTC",
            size=0.1,
            entry_price=45000.0,
            unrealized_pnl=500.0,
            margin_used=100.0,
            liquidation_price=40000.0,
            leverage={"type": "cross", "value": 10},
            position_value=4500.0,
            return_on_equity=0.05,
            last_updated=datetime.now()
        )
        
        self.position_manager.get_position.return_value = mock_position
        
        # Mock successful close response
        mock_close_result = {
            "status": "ok",
            "response": {
                "data": {
                    "statuses": [
                        {
                            "filled": {
                                "oid": "close123",
                                "totalSz": "0.1",
                                "avgPx": "46000.0"
                            }
                        }
                    ]
                }
            }
        }
        
        self.mock_api.market_close.return_value = mock_close_result
        
        # Close position
        result = self.executor.close_position("BTC")
        
        # Verify result
        self.assertTrue(result.success)
        self.assertEqual(result.order_id, "close123")
        self.assertEqual(result.filled_size, 0.1)
        self.assertEqual(result.avg_price, 46000.0)
        
        # Verify SDK call
        self.mock_api.market_close.assert_called_once_with("BTC")
    
    def test_execute_signal_buy(self):
        """Test executing buy signal."""
        # Mock account state
        self.position_manager.get_account_value.return_value = 10000.0
        self.position_manager.get_position.return_value = None  # No existing position
        
        # Mock risk checks
        self.executor._check_risk_limits = Mock(return_value=True)
        
        # Mock successful order
        mock_order_result = {
            "status": "ok",
            "response": {
                "data": {
                    "statuses": [
                        {
                            "filled": {
                                "oid": "buy123",
                                "totalSz": "6500.0",
                                "avgPx": "45000.0"
                            }
                        }
                    ]
                }
            }
        }
        
        self.mock_api.market_open.return_value = mock_order_result
        
        # Execute buy signal
        result = self.executor.execute_signal("BTC", "buy", 1.0)
        
        # Verify result
        self.assertTrue(result.success)
        self.assertEqual(result.order_id, "buy123")
        self.assertEqual(result.filled_size, 6500.0)
    
    def test_execute_signal_buy_existing_position(self):
        """Test executing buy signal when position already exists."""
        # Mock existing position
        mock_position = Position(
            symbol="BTC",
            size=0.1,
            entry_price=45000.0,
            unrealized_pnl=500.0,
            margin_used=100.0,
            liquidation_price=40000.0,
            leverage={"type": "cross", "value": 10},
            position_value=4500.0,
            return_on_equity=0.05,
            last_updated=datetime.now()
        )
        
        self.position_manager.get_account_value.return_value = 10000.0
        self.position_manager.get_position.return_value = mock_position
        
        # Mock risk checks to pass
        self.executor._check_risk_limits = Mock(return_value=True)
        
        # Execute buy signal
        result = self.executor.execute_signal("BTC", "buy", 1.0)
        
        # Should skip buy due to existing position
        self.assertTrue(result.success)
        self.assertIn("Already have long position", result.error)
    
    def test_execute_signal_sell(self):
        """Test executing sell signal."""
        # Mock existing position
        mock_position = Position(
            symbol="BTC",
            size=0.1,
            entry_price=45000.0,
            unrealized_pnl=500.0,
            margin_used=100.0,
            liquidation_price=40000.0,
            leverage={"type": "cross", "value": 10},
            position_value=4500.0,
            return_on_equity=0.05,
            last_updated=datetime.now()
        )
        
        self.position_manager.get_position.return_value = mock_position
        self.position_manager.get_account_value.return_value = 10000.0
        
        # Mock successful close
        mock_close_result = {
            "status": "ok",
            "response": {
                "data": {
                    "statuses": [
                        {
                            "filled": {
                                "oid": "sell123",
                                "totalSz": "0.1",
                                "avgPx": "46000.0"
                            }
                        }
                    ]
                }
            }
        }
        
        self.mock_api.market_close.return_value = mock_close_result
        
        # Mock risk checks to pass
        self.executor._check_risk_limits = Mock(return_value=True)
        
        # Execute sell signal
        result = self.executor.execute_signal("BTC", "sell", 1.0)
        
        # Verify result
        self.assertTrue(result.success)
        self.assertEqual(result.order_id, "sell123")
    
    def test_check_risk_limits(self):
        """Test risk limit checking."""
        # Mock position manager methods
        self.position_manager.calculate_margin_ratio.return_value = 0.5
        self.position_manager.get_total_unrealized_pnl.return_value = 100.0
        self.position_manager.get_account_value.return_value = 10000.0
        self.position_manager.check_liquidation_risk.return_value = (False, 0.0)
        
        # Test normal conditions
        result = self.executor._check_risk_limits("BTC")
        self.assertTrue(result)
        
        # Test high margin ratio
        self.position_manager.calculate_margin_ratio.return_value = 0.9
        result = self.executor._check_risk_limits("BTC")
        self.assertFalse(result)
        
        # Test daily loss limit
        self.position_manager.calculate_margin_ratio.return_value = 0.5
        self.position_manager.get_total_unrealized_pnl.return_value = -600.0  # 6% loss
        result = self.executor._check_risk_limits("BTC")
        self.assertFalse(result)
        
        # Test liquidation risk
        self.position_manager.get_total_unrealized_pnl.return_value = 100.0
        self.position_manager.check_liquidation_risk.return_value = (True, 15.0)
        result = self.executor._check_risk_limits("BTC")
        self.assertFalse(result)
    
    def test_retry_logic(self):
        """Test order retry logic."""
        # Mock failed order responses
        failed_result = {
            "status": "ok",
            "response": {
                "data": {
                    "statuses": [
                        {
                            "error": "Temporary error"
                        }
                    ]
                }
            }
        }
        
        success_result = {
            "status": "ok",
            "response": {
                "data": {
                    "statuses": [
                        {
                            "filled": {
                                "oid": "retry123",
                                "totalSz": "100.0",
                                "avgPx": "45000.0"
                            }
                        }
                    ]
                }
            }
        }
        
        # First two calls fail, third succeeds
        self.mock_api.market_open.side_effect = [failed_result, failed_result, success_result]
        
        # Place order with retry
        result = self.executor.place_market_order_with_retry("BTC", "buy", 100.0)
        
        # Verify success after retries
        self.assertTrue(result.success)
        self.assertEqual(result.order_id, "retry123")
        
        # Verify 3 calls were made
        self.assertEqual(self.mock_api.market_open.call_count, 3)


if __name__ == "__main__":
    unittest.main() 