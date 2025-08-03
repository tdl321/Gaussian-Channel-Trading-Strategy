"""
Test file for PerformanceTracker component.
"""

import unittest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
import tempfile
import os

from utils.performance import PerformanceTracker, Trade, PerformanceMetrics
from execution.executor import OrderResult


class TestPerformanceTracker(unittest.TestCase):
    """Test cases for PerformanceTracker."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock configuration
        self.mock_config = Mock()
        
        # Create performance tracker
        self.tracker = PerformanceTracker(self.mock_config)
        
        # Override file paths to use temp directory
        self.tracker.trades_file = os.path.join(self.temp_dir, "trades.csv")
        self.tracker.metrics_file = os.path.join(self.temp_dir, "performance_metrics.json")
        self.tracker.equity_file = os.path.join(self.temp_dir, "equity_curve.csv")
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary directory
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_add_trade(self):
        """Test adding a new trade."""
        # Create mock order result
        order_result = OrderResult(
            success=True,
            order_id="order123",
            filled_size=100.0,
            avg_price=45000.0,
            timestamp=datetime.now()
        )
        
        # Add trade
        trade_id = self.tracker.add_trade(
            order_result=order_result,
            symbol="BTC",
            side="buy",
            size=100.0,
            price=45000.0,
            commission=5.0,
            slippage=2.0
        )
        
        # Verify trade was added
        self.assertIn(trade_id, self.tracker.trades)
        trade = self.tracker.trades[trade_id]
        
        self.assertEqual(trade.symbol, "BTC")
        self.assertEqual(trade.side, "buy")
        self.assertEqual(trade.size, 100.0)
        self.assertEqual(trade.entry_price, 45000.0)
        self.assertEqual(trade.commission, 5.0)
        self.assertEqual(trade.slippage, 2.0)
        self.assertEqual(trade.status, "open")
    
    def test_add_failed_trade(self):
        """Test that failed trades are not added."""
        # Create failed order result
        order_result = OrderResult(
            success=False,
            error="Insufficient margin",
            timestamp=datetime.now()
        )
        
        # Try to add failed trade
        trade_id = self.tracker.add_trade(
            order_result=order_result,
            symbol="BTC",
            side="buy",
            size=100.0,
            price=45000.0
        )
        
        # Should return empty string and not add trade
        self.assertEqual(trade_id, "")
        self.assertEqual(len(self.tracker.trades), 0)
    
    def test_close_trade(self):
        """Test closing a trade."""
        # Add a trade first
        order_result = OrderResult(
            success=True,
            order_id="order123",
            filled_size=100.0,
            avg_price=45000.0,
            timestamp=datetime.now()
        )
        
        trade_id = self.tracker.add_trade(
            order_result=order_result,
            symbol="BTC",
            side="buy",
            size=100.0,
            price=45000.0
        )
        
        # Close the trade
        success = self.tracker.close_trade(trade_id, 46000.0)
        
        # Verify trade was closed
        self.assertTrue(success)
        trade = self.tracker.trades[trade_id]
        
        self.assertEqual(trade.status, "closed")
        self.assertEqual(trade.exit_price, 46000.0)
        self.assertIsNotNone(trade.pnl)
        self.assertIsNotNone(trade.pnl_percent)
        
        # Verify P&L calculation (buy trade)
        expected_pnl = 100.0 * (46000.0 - 45000.0) / 45000.0  # USD-based calculation
        self.assertAlmostEqual(trade.pnl, expected_pnl, places=2)
    
    def test_close_nonexistent_trade(self):
        """Test closing a trade that doesn't exist."""
        success = self.tracker.close_trade("nonexistent", 46000.0)
        self.assertFalse(success)
    
    def test_calculate_metrics_empty(self):
        """Test metrics calculation with no trades."""
        metrics = self.tracker.calculate_metrics()
        
        self.assertEqual(metrics.total_trades, 0)
        self.assertEqual(metrics.win_rate, 0.0)
        self.assertEqual(metrics.total_pnl, 0.0)
        self.assertEqual(metrics.profit_factor, 0.0)
    
    def test_calculate_metrics_with_trades(self):
        """Test metrics calculation with trades."""
        # Add winning trade
        order_result1 = OrderResult(
            success=True,
            order_id="order1",
            filled_size=100.0,
            avg_price=45000.0,
            timestamp=datetime.now()
        )
        
        trade_id1 = self.tracker.add_trade(
            order_result=order_result1,
            symbol="BTC",
            side="buy",
            size=100.0,
            price=45000.0
        )
        
        self.tracker.close_trade(trade_id1, 46000.0)  # $1000 profit (100 * (46000-45000))
        
        # Add losing trade
        order_result2 = OrderResult(
            success=True,
            order_id="order2",
            filled_size=50.0,
            avg_price=45000.0,
            timestamp=datetime.now()
        )
        
        trade_id2 = self.tracker.add_trade(
            order_result=order_result2,
            symbol="ETH",
            side="buy",
            size=50.0,
            price=45000.0
        )
        
        self.tracker.close_trade(trade_id2, 44000.0)  # $500 loss (50 * (45000-44000))
        
        # Calculate metrics
        metrics = self.tracker.calculate_metrics()
        
        # Verify metrics
        self.assertEqual(metrics.total_trades, 2)
        self.assertEqual(metrics.winning_trades, 1)
        self.assertEqual(metrics.losing_trades, 1)
        self.assertEqual(metrics.win_rate, 0.5)
        # P&L calculation: size is in USD, so we need to calculate differently
        # Trade 1: 100 USD at 45000, exit at 46000 = 100 * (46000-45000)/45000 = 2.22 USD profit
        # Trade 2: 50 USD at 45000, exit at 44000 = 50 * (45000-44000)/45000 = 1.11 USD loss
        expected_total_pnl = 2.22 - 1.11  # Approximately 1.11
        self.assertAlmostEqual(metrics.total_pnl, expected_total_pnl, places=1)
        self.assertAlmostEqual(metrics.gross_profit, 2.22, places=1)
        self.assertAlmostEqual(metrics.gross_loss, 1.11, places=1)
        self.assertAlmostEqual(metrics.profit_factor, 2.0, places=1)  # 2.22/1.11
        self.assertAlmostEqual(metrics.average_win, 2.22, places=1)
        self.assertAlmostEqual(metrics.average_loss, 1.11, places=1)  # Absolute value of loss
        self.assertAlmostEqual(metrics.largest_win, 2.22, places=1)
        self.assertAlmostEqual(metrics.largest_loss, 1.11, places=1)  # Absolute value of loss
    
    def test_metrics_caching(self):
        """Test that metrics are cached and not recalculated unnecessarily."""
        # Add a trade
        order_result = OrderResult(
            success=True,
            order_id="order123",
            filled_size=100.0,
            avg_price=45000.0,
            timestamp=datetime.now()
        )
        
        trade_id = self.tracker.add_trade(
            order_result=order_result,
            symbol="BTC",
            side="buy",
            size=100.0,
            price=45000.0
        )
        
        self.tracker.close_trade(trade_id, 46000.0)
        
        # Calculate metrics twice
        metrics1 = self.tracker.calculate_metrics()
        metrics2 = self.tracker.calculate_metrics()
        
        # Should be the same object (cached)
        self.assertIs(metrics1, metrics2)
        
        # Force recalculation
        metrics3 = self.tracker.calculate_metrics(force_recalculate=True)
        
        # Should be different object
        self.assertIsNot(metrics1, metrics3)
    
    def test_trade_summary(self):
        """Test trade summary generation."""
        # Add some trades
        order_result1 = OrderResult(
            success=True,
            order_id="order1",
            filled_size=100.0,
            avg_price=45000.0,
            timestamp=datetime.now()
        )
        
        trade_id1 = self.tracker.add_trade(
            order_result=order_result1,
            symbol="BTC",
            side="buy",
            size=100.0,
            price=45000.0
        )
        
        self.tracker.close_trade(trade_id1, 46000.0)
        
        # Add open trade
        order_result2 = OrderResult(
            success=True,
            order_id="order2",
            filled_size=50.0,
            avg_price=45000.0,
            timestamp=datetime.now()
        )
        
        self.tracker.add_trade(
            order_result=order_result2,
            symbol="ETH",
            side="buy",
            size=50.0,
            price=45000.0
        )
        
        # Get summary
        summary = self.tracker.get_trade_summary()
        
        # Verify summary
        self.assertEqual(summary["total_trades"], 2)
        self.assertEqual(summary["open_trades"], 1)
        self.assertEqual(summary["closed_trades"], 1)
        self.assertAlmostEqual(summary["total_pnl"], 2.22, places=2)  # Only closed trade: 100 * (46000-45000)/45000 = 2.22
        self.assertEqual(len(summary["trades"]), 2)
    
    def test_file_saving(self):
        """Test that trades are saved to files."""
        # Add a trade
        order_result = OrderResult(
            success=True,
            order_id="order123",
            filled_size=100.0,
            avg_price=45000.0,
            timestamp=datetime.now()
        )
        
        trade_id = self.tracker.add_trade(
            order_result=order_result,
            symbol="BTC",
            side="buy",
            size=100.0,
            price=45000.0
        )
        
        self.tracker.close_trade(trade_id, 46000.0)
        
        # Save performance data
        self.tracker.save_performance_data()
        
        # Verify files were created
        self.assertTrue(os.path.exists(self.tracker.metrics_file))
        self.assertTrue(os.path.exists(self.tracker.equity_file))
        
        # Verify trades file exists
        self.assertTrue(os.path.exists(self.tracker.trades_file))
    
    def test_drawdown_calculation(self):
        """Test maximum drawdown calculation."""
        # Simulate equity curve with drawdown
        self.tracker.equity_curve = [
            (datetime.now(), 1000.0),   # Peak
            (datetime.now(), 800.0),    # Drawdown
            (datetime.now(), 900.0),    # Recovery
            (datetime.now(), 600.0),    # Deeper drawdown
            (datetime.now(), 700.0),    # Partial recovery
        ]
        
        max_dd, max_dd_percent = self.tracker._calculate_max_drawdown()
        
        # Maximum drawdown should be 400 (1000 - 600)
        self.assertEqual(max_dd, 400.0)
        # Maximum drawdown percentage should be 40% (400/1000)
        self.assertEqual(max_dd_percent, 40.0)
    
    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation."""
        # Simulate daily returns
        self.tracker.daily_returns = [
            (datetime.now(), 0.01),   # 1% return
            (datetime.now(), 0.02),   # 2% return
            (datetime.now(), -0.01),  # -1% return
            (datetime.now(), 0.015),  # 1.5% return
        ]
        
        sharpe = self.tracker._calculate_sharpe_ratio()
        
        # Should be a reasonable Sharpe ratio
        self.assertIsInstance(sharpe, float)
        self.assertGreater(sharpe, -10)  # Not extremely negative
        self.assertLess(sharpe, 10)      # Not extremely positive


if __name__ == "__main__":
    unittest.main() 