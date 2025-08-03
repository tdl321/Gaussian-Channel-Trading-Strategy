#!/usr/bin/env python3
"""
Performance Tracker Demo Script

Demonstrates the functionality of the PerformanceTracker class
with realistic trading scenarios.
"""

import sys
import os
from datetime import datetime, timedelta
from unittest.mock import Mock

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from utils.performance import PerformanceTracker
from execution.executor import OrderResult


def create_mock_order_result(success=True, order_id="", filled_size=0.0, avg_price=0.0, timestamp=None):
    """Create a mock OrderResult for testing."""
    return OrderResult(
        success=success,
        order_id=order_id,
        filled_size=filled_size,
        avg_price=avg_price,
        timestamp=timestamp or datetime.now()
    )


def create_mock_config():
    """Create a mock configuration for demo purposes."""
    config = Mock()
    config.HYPERLIQUID_API_KEY = "demo_key"
    config.HYPERLIQUID_SECRET_KEY = "demo_secret"
    config.HYPERLIQUID_BASE_URL = "https://api.hyperliquid.xyz"
    config.HYPERLIQUID_WS_URL = "wss://api.hyperliquid.xyz/ws"
    config.SYMBOL = "BTC-PERP"
    config.INITIAL_CAPITAL = 10000
    config.POSITION_SIZE_PCT = 0.65
    config.MAX_LEVERAGE = 5.0
    config.MARGIN_REQUIREMENT = 0.2
    config.MAINTENANCE_MARGIN_PCT = 0.75
    config.POLES = 4
    config.PERIOD = 144
    config.MULTIPLIER = 1.414
    config.MODE_LAG = False
    config.MODE_FAST = False
    config.ATR_SPACING = 0.4
    config.MAX_PYRAMIDS = 5
    config.SMA_LENGTH = 200
    config.ENABLE_SMA_FILTER = False
    config.COMMISSION_PCT = 0.001
    config.SLIPPAGE_TICKS = 1
    config.SLIPPAGE_PER_TICK = 0.0001
    config.TRADING_INTERVAL = 60
    config.DATA_RETRY_DELAY = 5
    config.ERROR_RETRY_DELAY = 30
    config.START_DATE = "2018-01-01"
    config.END_DATE = "2069-12-31"
    config.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    config.DATA_DIR = os.path.join(config.BASE_DIR, 'data')
    config.LOGS_DIR = os.path.join(config.BASE_DIR, 'logs')
    config.RESULTS_DIR = os.path.join(config.BASE_DIR, 'results')
    config.LOG_LEVEL = "INFO"
    config.LOG_FILE = os.path.join(config.LOGS_DIR, 'trading.log')
    config.TRADE_LOG_FILE = os.path.join(config.LOGS_DIR, 'trades.csv')
    config.PERFORMANCE_LOG_FILE = os.path.join(config.LOGS_DIR, 'performance.csv')
    config.MAX_DRAWDOWN_PCT = 20.0
    config.DAILY_LOSS_LIMIT = 5.0
    config.FORCED_LIQUIDATION_BUFFER = 0.05
    config.BACKTEST_START_DATE = "2020-01-01"
    config.BACKTEST_END_DATE = "2024-01-01"
    config.BACKTEST_INITIAL_CAPITAL = 10000
    config.IS_RENDER_DEPLOYMENT = False
    config.RENDER_SERVICE_URL = ""
    config.DEBUG_MODE = False
    config.VERBOSE_LOGGING = True
    
    # Create directories if they don't exist
    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    return config


def demo_basic_trading():
    """Demonstrate basic trading performance tracking."""
    print("=" * 60)
    print("BASIC TRADING PERFORMANCE DEMO")
    print("=" * 60)
    
    # Initialize configuration and performance tracker
    config = create_mock_config()
    tracker = PerformanceTracker(config)
    
    # Simulate a series of trades
    trades = [
        # Winning trades
        {"symbol": "BTC", "side": "buy", "size": 1000, "entry_price": 45000, "exit_price": 46000, "commission": 1.0},
        {"symbol": "ETH", "side": "buy", "size": 500, "entry_price": 3000, "exit_price": 3150, "commission": 0.5},
        {"symbol": "SOL", "side": "buy", "size": 200, "entry_price": 100, "exit_price": 110, "commission": 0.2},
        
        # Losing trades
        {"symbol": "BTC", "side": "buy", "size": 800, "entry_price": 46000, "exit_price": 45000, "commission": 0.8},
        {"symbol": "ETH", "side": "buy", "size": 300, "entry_price": 3150, "exit_price": 3000, "commission": 0.3},
    ]
    
    # Execute trades
    for i, trade_data in enumerate(trades):
        # Add trade
        order_result = create_mock_order_result(
            order_id=f"order_{i+1:03d}",
            filled_size=trade_data["size"],
            avg_price=trade_data["entry_price"]
        )
        
        trade_id = tracker.add_trade(
            order_result=order_result,
            symbol=trade_data["symbol"],
            side=trade_data["side"],
            size=trade_data["size"],
            price=trade_data["entry_price"],
            commission=trade_data["commission"]
        )
        
        # Close trade after a delay
        tracker.close_trade(trade_id, trade_data["exit_price"])
        
        print(f"Trade {i+1}: {trade_data['symbol']} {trade_data['side'].upper()} "
              f"${trade_data['size']:,.0f} @ ${trade_data['entry_price']:,.2f} → ${trade_data['exit_price']:,.2f}")
    
    # Calculate and display metrics
    metrics = tracker.calculate_metrics()
    
    print("\n" + "=" * 40)
    print("PERFORMANCE METRICS")
    print("=" * 40)
    print(f"Total Trades: {metrics.total_trades}")
    print(f"Win Rate: {metrics.win_rate:.2%}")
    print(f"Total P&L: ${metrics.total_pnl:,.2f}")
    print(f"Profit Factor: {metrics.profit_factor:.2f}")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"Max Drawdown: ${metrics.max_drawdown:,.2f} ({metrics.max_drawdown_percent:.2f}%)")
    print(f"Average Win: ${metrics.average_win:,.2f}")
    print(f"Average Loss: ${metrics.average_loss:,.2f}")
    print(f"Total Commission: ${metrics.total_commission:,.2f}")


def demo_advanced_features():
    """Demonstrate advanced performance tracking features."""
    print("\n" + "=" * 60)
    print("ADVANCED FEATURES DEMO")
    print("=" * 60)
    
    # Initialize configuration and performance tracker
    config = create_mock_config()
    tracker = PerformanceTracker(config)
    
    # Simulate more complex trading scenario
    base_time = datetime.now() - timedelta(days=30)
    
    # Add trades with different timestamps
    for i in range(10):
        trade_time = base_time + timedelta(days=i*3)
        
        # Simulate winning and losing trades
        if i % 3 == 0:  # Losing trade
            entry_price = 45000 + i * 100
            exit_price = entry_price - 500
            size = 500 + i * 50
        else:  # Winning trade
            entry_price = 45000 + i * 100
            exit_price = entry_price + 300
            size = 500 + i * 50
        
        order_result = create_mock_order_result(
            order_id=f"order_{i+1:03d}",
            filled_size=size,
            avg_price=entry_price,
            timestamp=trade_time
        )
        
        trade_id = tracker.add_trade(
            order_result=order_result,
            symbol="BTC",
            side="buy",
            size=size,
            price=entry_price,
            commission=size * 0.001,  # 0.1% commission
            slippage=size * 0.0005    # 0.05% slippage
        )
        
        # Close trade
        tracker.close_trade(trade_id, exit_price, trade_time + timedelta(hours=2))
    
    # Generate performance report
    print("\nGenerating comprehensive performance report...")
    report = tracker.generate_performance_report()
    print(report)
    
    # Get symbol performance breakdown
    print("\n" + "=" * 40)
    print("SYMBOL PERFORMANCE BREAKDOWN")
    print("=" * 40)
    symbol_perf = tracker.get_symbol_performance()
    for symbol, stats in symbol_perf.items():
        print(f"{symbol}:")
        print(f"  Trades: {stats['trades']}")
        print(f"  Win Rate: {stats['win_rate']:.2%}")
        print(f"  Total P&L: ${stats['total_pnl']:,.2f}")
        print(f"  Average P&L: ${stats['avg_pnl']:,.2f}")
        print(f"  Total Volume: ${stats['total_volume']:,.2f}")
        print()
    
    # Get monthly returns
    print("=" * 40)
    print("MONTHLY RETURNS")
    print("=" * 40)
    monthly_returns = tracker.get_monthly_returns()
    for month, pnl in sorted(monthly_returns.items()):
        print(f"{month}: ${pnl:,.2f}")
    
    # Save performance data
    print("\nSaving performance data to files...")
    tracker.save_performance_data()
    print("Performance data saved successfully!")


def demo_risk_metrics():
    """Demonstrate risk metrics calculation."""
    print("\n" + "=" * 60)
    print("RISK METRICS DEMO")
    print("=" * 60)
    
    # Initialize configuration and performance tracker
    config = create_mock_config()
    tracker = PerformanceTracker(config)
    
    # Simulate equity curve with drawdowns
    base_time = datetime.now() - timedelta(days=100)
    equity_values = [10000]  # Start with $10,000
    
    # Simulate daily equity changes
    for i in range(100):
        # Add some volatility
        daily_return = 0.001 + (i % 7 - 3) * 0.002  # Weekly pattern
        new_equity = equity_values[-1] * (1 + daily_return)
        equity_values.append(new_equity)
        
        # Add to equity curve
        tracker.equity_curve.append((base_time + timedelta(days=i), new_equity))
        
        # Calculate daily returns
        if i > 0:
            daily_return = (new_equity - equity_values[-2]) / equity_values[-2]
            tracker.daily_returns.append((base_time + timedelta(days=i), daily_return))
    
    # Calculate risk metrics
    sharpe = tracker._calculate_sharpe_ratio()
    sortino = tracker._calculate_sortino_ratio()
    max_dd, max_dd_percent = tracker._calculate_max_drawdown()
    
    print(f"Sharpe Ratio: {sharpe:.3f}")
    print(f"Sortino Ratio: {sortino:.3f}")
    print(f"Maximum Drawdown: ${max_dd:,.2f} ({max_dd_percent:.2f}%)")
    
    # Show equity curve summary
    print(f"\nEquity Curve Summary:")
    print(f"Starting Equity: ${equity_values[0]:,.2f}")
    print(f"Ending Equity: ${equity_values[-1]:,.2f}")
    print(f"Total Return: {((equity_values[-1] / equity_values[0]) - 1) * 100:.2f}%")


if __name__ == "__main__":
    print("GAUSSIAN CHANNEL TRADING BOT - PERFORMANCE TRACKER DEMO")
    print("=" * 70)
    
    try:
        # Run demonstrations
        demo_basic_trading()
        demo_advanced_features()
        demo_risk_metrics()
        
        print("\n" + "=" * 70)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc() 