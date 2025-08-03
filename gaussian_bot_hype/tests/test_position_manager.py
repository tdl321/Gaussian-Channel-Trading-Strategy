"""
Test file for PositionManager component.
"""

import unittest
from unittest.mock import Mock, patch
from datetime import datetime

from execution.position_manager import PositionManager, Position, AccountState


class TestPositionManager(unittest.TestCase):
    """Test cases for PositionManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock configuration
        self.mock_config = Mock()
        self.mock_config.MAX_LEVERAGE = 5.0
        self.mock_config.MAX_POSITION_SIZE = 0.65
        self.mock_config.MAX_DRAWDOWN = 0.20
        self.mock_config.DAILY_LOSS_LIMIT = 0.05
        
        # Mock Hyperliquid objects
        self.mock_info = Mock()
        self.mock_exchange = Mock()
        self.mock_account = Mock()
        self.mock_account.address = "0x1234567890abcdef"
        
        # Create position manager instance
        self.position_manager = PositionManager(
            self.mock_config,
            self.mock_info,
            self.mock_exchange,
            self.mock_account
        )
    
    def test_position_parsing(self):
        """Test position data parsing from Hyperliquid API format."""
        # Mock user state data (following SDK format)
        mock_user_state = {
            "assetPositions": [
                {
                    "position": {
                        "coin": "BTC",
                        "szi": "0.1",
                        "entryPx": "45000.0",
                        "unrealizedPnl": "500.0",
                        "marginUsed": "100.0",
                        "liquidationPx": "40000.0",
                        "leverage": {"type": "cross", "value": 10},
                        "positionValue": "4500.0",
                        "returnOnEquity": "0.05"
                    }
                }
            ],
            "marginSummary": {
                "accountValue": "5000.0",
                "totalMarginUsed": "100.0",
                "totalNtlPos": "4500.0",
                "totalRawUsd": "5000.0"
            }
        }
        
        # Mock the info.user_state method
        self.mock_info.user_state.return_value = mock_user_state
        
        # Update positions
        positions = self.position_manager.update_positions()
        
        # Verify position was parsed correctly
        self.assertIn("BTC", positions)
        btc_position = positions["BTC"]
        
        self.assertEqual(btc_position.symbol, "BTC")
        self.assertEqual(btc_position.size, 0.1)
        self.assertEqual(btc_position.entry_price, 45000.0)
        self.assertEqual(btc_position.unrealized_pnl, 500.0)
        self.assertEqual(btc_position.margin_used, 100.0)
        self.assertEqual(btc_position.liquidation_price, 40000.0)
    
    def test_account_state_parsing(self):
        """Test account state parsing from Hyperliquid API format."""
        mock_user_state = {
            "assetPositions": [],
            "marginSummary": {
                "accountValue": "10000.0",
                "totalMarginUsed": "2000.0",
                "totalNtlPos": "8000.0",
                "totalRawUsd": "10000.0"
            }
        }
        
        self.mock_info.user_state.return_value = mock_user_state
        self.position_manager.update_positions()
        
        # Verify account state
        self.assertIsNotNone(self.position_manager.account_state)
        self.assertEqual(self.position_manager.get_account_value(), 10000.0)
        self.assertEqual(self.position_manager.get_total_exposure(), 8000.0)
        self.assertEqual(self.position_manager.calculate_margin_ratio(), 0.2)
    
    def test_position_queries(self):
        """Test position query methods."""
        # Set up mock positions
        self.position_manager.positions = {
            "BTC": Position(
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
        }
        
        # Test position queries
        btc_position = self.position_manager.get_position("BTC")
        self.assertIsNotNone(btc_position)
        self.assertEqual(btc_position.size, 0.1)
        
        eth_position = self.position_manager.get_position("ETH")
        self.assertIsNone(eth_position)
        
        # Test P&L queries
        self.assertEqual(self.position_manager.get_unrealized_pnl("BTC"), 500.0)
        self.assertEqual(self.position_manager.get_unrealized_pnl("ETH"), 0.0)
        self.assertEqual(self.position_manager.get_total_unrealized_pnl(), 500.0)
    
    def test_risk_management(self):
        """Test risk management methods."""
        # Set up mock account state
        self.position_manager.account_state = AccountState(
            account_value=5000.0,
            total_margin_used=4000.0,  # 80% margin ratio
            total_notional_position=8000.0,
            total_raw_usd=5000.0,
            last_updated=datetime.now()
        )
        
        # Set up a mock position
        self.position_manager.positions = {
            "BTC": Position(
                symbol="BTC",
                size=0.1,
                entry_price=45000.0,
                unrealized_pnl=-100.0,  # Small negative P&L (2% loss)
                margin_used=100.0,
                liquidation_price=40000.0,
                leverage={"type": "cross", "value": 10},
                position_value=4500.0,
                return_on_equity=-0.02,
                last_updated=datetime.now()
            )
        }
        
        # Test margin ratio calculation
        margin_ratio = self.position_manager.calculate_margin_ratio()
        self.assertEqual(margin_ratio, 0.8)
        
        # Mock the _get_current_price method to avoid API calls
        self.position_manager._get_current_price = Mock(return_value=45000.0)
        
        # Test risk assessment (should reduce due to high margin ratio)
        should_reduce = self.position_manager.should_reduce_position("BTC", "medium")
        self.assertTrue(should_reduce)  # Should reduce due to high margin ratio
        
        # Test with lower margin ratio (should not reduce)
        self.position_manager.account_state.total_margin_used = 1000.0  # 20% margin ratio
        should_reduce_low = self.position_manager.should_reduce_position("BTC", "medium")
        self.assertFalse(should_reduce_low)  # Should not reduce due to low margin ratio
    
    def test_position_summary(self):
        """Test position summary generation."""
        # Set up mock data
        self.position_manager.positions = {
            "BTC": Position(
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
        }
        
        self.position_manager.account_state = AccountState(
            account_value=5000.0,
            total_margin_used=100.0,
            total_notional_position=4500.0,
            total_raw_usd=5000.0,
            last_updated=datetime.now()
        )
        
        # Get summary
        summary = self.position_manager.get_position_summary()
        
        # Verify summary structure
        self.assertEqual(summary["account_value"], 5000.0)
        self.assertEqual(summary["total_exposure"], 4500.0)
        self.assertEqual(summary["total_unrealized_pnl"], 500.0)
        self.assertEqual(summary["position_count"], 1)
        self.assertIn("BTC", summary["positions"])


if __name__ == "__main__":
    unittest.main() 