"""
Performance Metrics for Gaussian Channel Trading Bot

Handles trade tracking, performance calculations, and reporting
for monitoring trading strategy effectiveness.
"""

import logging
import json
import csv
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os

from execution.executor import OrderResult
from execution.position_manager import Position


@dataclass
class Trade:
    """Trade data structure for performance tracking"""
    trade_id: str
    symbol: str
    side: str  # "buy" or "sell"
    size: float
    entry_price: float
    exit_price: Optional[float] = None
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    commission: float = 0.0
    slippage: float = 0.0
    order_id: Optional[str] = None
    status: str = "open"  # "open", "closed", "cancelled"


@dataclass
class PerformanceMetrics:
    """Performance metrics structure"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    gross_profit: float
    gross_loss: float
    profit_factor: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_percent: float
    average_trade_duration: float
    total_commission: float
    total_slippage: float
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


class PerformanceTracker:
    """
    Tracks trading performance and calculates key metrics.
    
    Provides comprehensive performance analysis including
    risk-adjusted returns, drawdown analysis, and trade statistics.
    """
    
    def __init__(self, config):
        """
        Initialize the performance tracker.
        
        Args:
            config: Configuration object with performance parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Trade tracking
        self.trades: Dict[str, Trade] = {}
        self.trade_counter = 0
        
        # Performance data
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.daily_returns: List[Tuple[datetime, float]] = []
        
        # Metrics cache
        self._metrics_cache: Optional[PerformanceMetrics] = None
        self._last_calculation = None
        
        # File paths
        self.trades_file = "logs/trades.csv"
        self.metrics_file = "logs/performance_metrics.json"
        self.equity_file = "logs/equity_curve.csv"
        
        self.logger.info("PerformanceTracker initialized")
    
    def add_trade(self, order_result: OrderResult, symbol: str, side: str, 
                  size: float, price: float, commission: float = 0.0, 
                  slippage: float = 0.0) -> str:
        """
        Add a new trade to the performance tracker.
        
        Args:
            order_result: Order execution result
            symbol: Trading symbol
            side: Trade side ("buy" or "sell")
            size: Trade size
            price: Execution price
            commission: Commission paid
            slippage: Slippage incurred
            
        Returns:
            Trade ID for tracking
        """
        if not order_result.success:
            self.logger.warning(f"Not adding failed trade: {order_result.error}")
            return ""
        
        self.trade_counter += 1
        trade_id = f"trade_{self.trade_counter:06d}"
        
        trade = Trade(
            trade_id=trade_id,
            symbol=symbol,
            side=side,
            size=size,
            entry_price=price,
            entry_time=order_result.timestamp or datetime.now(),
            commission=commission,
            slippage=slippage,
            order_id=order_result.order_id,
            status="open"
        )
        
        self.trades[trade_id] = trade
        
        # Update equity curve
        self._update_equity_curve()
        
        # Clear metrics cache
        self._metrics_cache = None
        
        self.logger.info(f"Added trade: {trade_id} {symbol} {side} {size} @ ${price:.2f}")
        
        # Save to file
        self._save_trade_to_file(trade)
        
        return trade_id
    
    def close_trade(self, trade_id: str, exit_price: float, 
                   exit_time: Optional[datetime] = None) -> bool:
        """
        Close an existing trade.
        
        Args:
            trade_id: Trade ID to close
            exit_price: Exit price
            exit_time: Exit time (defaults to now)
            
        Returns:
            True if trade was successfully closed
        """
        if trade_id not in self.trades:
            self.logger.error(f"Trade not found: {trade_id}")
            return False
        
        trade = self.trades[trade_id]
        if trade.status != "open":
            self.logger.warning(f"Trade {trade_id} is not open (status: {trade.status})")
            return False
        
        # Update trade
        trade.exit_price = exit_price
        trade.exit_time = exit_time or datetime.now()
        trade.status = "closed"
        
        # Calculate P&L for USD-based position sizing
        # size = USD amount traded, not number of units
        if trade.side.lower() == "buy":
            # For long positions: P&L = size * (exit_price - entry_price) / entry_price
            trade.pnl = trade.size * (exit_price - trade.entry_price) / trade.entry_price
        else:  # sell (short)
            # For short positions: P&L = size * (entry_price - exit_price) / entry_price
            trade.pnl = trade.size * (trade.entry_price - exit_price) / trade.entry_price
        
        # Subtract costs
        trade.pnl -= trade.commission + trade.slippage
        
        # Calculate percentage return
        trade.pnl_percent = (trade.pnl / trade.size) * 100
        
        # Update equity curve
        self._update_equity_curve()
        
        # Clear metrics cache
        self._metrics_cache = None
        
        self.logger.info(f"Closed trade: {trade_id} P&L: ${trade.pnl:.2f} ({trade.pnl_percent:.2f}%)")
        
        # Update file
        self._update_trade_in_file(trade)
        
        return True
    
    def calculate_metrics(self, force_recalculate: bool = False) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            force_recalculate: Force recalculation even if cache is valid
            
        Returns:
            PerformanceMetrics object
        """
        # Check cache
        if (not force_recalculate and self._metrics_cache is not None and 
            self._last_calculation and 
            (datetime.now() - self._last_calculation).seconds < 60):
            return self._metrics_cache
        
        if not self.trades:
            return self._create_empty_metrics()
        
        # Get closed trades only
        closed_trades = [t for t in self.trades.values() if t.status == "closed"]
        
        if not closed_trades:
            return self._create_empty_metrics()
        
        # Basic trade statistics
        total_trades = len(closed_trades)
        winning_trades = len([t for t in closed_trades if t.pnl > 0])
        losing_trades = len([t for t in closed_trades if t.pnl < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # P&L calculations
        total_pnl = sum(t.pnl for t in closed_trades)
        gross_profit = sum(t.pnl for t in closed_trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in closed_trades if t.pnl < 0))
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Average calculations
        wins = [t.pnl for t in closed_trades if t.pnl > 0]
        losses = [t.pnl for t in closed_trades if t.pnl < 0]
        
        average_win = np.mean(wins) if wins else 0.0
        average_loss = abs(np.mean(losses)) if losses else 0.0  # Use absolute value for average loss
        largest_win = max(wins) if wins else 0.0
        largest_loss = abs(min(losses)) if losses else 0.0  # Use absolute value for largest loss
        
        # Risk metrics
        sharpe_ratio = self._calculate_sharpe_ratio()
        sortino_ratio = self._calculate_sortino_ratio()
        max_drawdown, max_drawdown_percent = self._calculate_max_drawdown()
        
        # Trade duration
        durations = []
        for trade in closed_trades:
            if trade.entry_time and trade.exit_time:
                duration = (trade.exit_time - trade.entry_time).total_seconds() / 3600  # hours
                durations.append(duration)
        
        average_trade_duration = np.mean(durations) if durations else 0.0
        
        # Cost analysis
        total_commission = sum(t.commission for t in closed_trades)
        total_slippage = sum(t.slippage for t in closed_trades)
        
        # Date range
        start_date = min(t.entry_time for t in closed_trades if t.entry_time)
        end_date = max(t.exit_time for t in closed_trades if t.exit_time)
        
        # Create metrics object
        metrics = PerformanceMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            profit_factor=profit_factor,
            average_win=average_win,
            average_loss=average_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_percent=max_drawdown_percent,
            average_trade_duration=average_trade_duration,
            total_commission=total_commission,
            total_slippage=total_slippage,
            start_date=start_date,
            end_date=end_date
        )
        
        # Update cache
        self._metrics_cache = metrics
        self._last_calculation = datetime.now()
        
        return metrics
    
    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio using daily returns."""
        if not self.daily_returns:
            return 0.0
        
        returns = [r[1] for r in self.daily_returns]
        if not returns:
            return 0.0
        
        # Need at least 2 returns for meaningful calculation
        if len(returns) < 2:
            return 0.0
        
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # For small sample sizes, use a more conservative annualization
        # Only annualize if we have enough data points
        if len(returns) >= 30:  # At least 30 days of data
            # Annualized Sharpe ratio
            sharpe = (avg_return - risk_free_rate/252) / std_return * np.sqrt(252)
        else:
            # Use simple Sharpe ratio without annualization for small samples
            sharpe = (avg_return - risk_free_rate/252) / std_return
        
        # Cap extreme values
        return max(min(sharpe, 10.0), -10.0)
    
    def _calculate_sortino_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio using downside deviation."""
        if not self.daily_returns:
            return 0.0
        
        returns = [r[1] for r in self.daily_returns]
        if not returns:
            return 0.0
        
        # Need at least 2 returns for meaningful calculation
        if len(returns) < 2:
            return 0.0
        
        avg_return = np.mean(returns)
        
        # Calculate downside deviation
        downside_returns = [r for r in returns if r < avg_return]
        if not downside_returns:
            return float('inf')
        
        downside_deviation = np.std(downside_returns)
        
        if downside_deviation == 0:
            return float('inf')
        
        # For small sample sizes, use a more conservative annualization
        if len(returns) >= 30:  # At least 30 days of data
            # Annualized Sortino ratio
            sortino = (avg_return - risk_free_rate/252) / downside_deviation * np.sqrt(252)
        else:
            # Use simple Sortino ratio without annualization for small samples
            sortino = (avg_return - risk_free_rate/252) / downside_deviation
        
        # Cap extreme values
        return max(min(sortino, 10.0), -10.0)
    
    def _calculate_max_drawdown(self) -> Tuple[float, float]:
        """Calculate maximum drawdown and percentage."""
        if not self.equity_curve:
            return 0.0, 0.0
        
        equity_values = [e[1] for e in self.equity_curve]
        if not equity_values:
            return 0.0, 0.0
        
        peak = equity_values[0]
        max_dd = 0.0
        max_dd_percent = 0.0
        
        for value in equity_values:
            if value > peak:
                peak = value
            
            drawdown = peak - value
            drawdown_percent = (drawdown / peak) * 100 if peak > 0 else 0.0
            
            if drawdown > max_dd:
                max_dd = drawdown
                max_dd_percent = drawdown_percent
        
        return max_dd, max_dd_percent
    
    def _update_equity_curve(self) -> None:
        """Update equity curve based on current trades."""
        # This would typically be called with current account value
        # For now, we'll calculate from trades
        if not self.trades:
            return
        
        # Calculate current equity from closed trades
        total_pnl = sum(t.pnl for t in self.trades.values() if t.status == "closed" and t.pnl is not None)
        
        # Add to equity curve
        self.equity_curve.append((datetime.now(), total_pnl))
        
        # Calculate daily returns
        if len(self.equity_curve) >= 2:
            prev_equity = self.equity_curve[-2][1]
            current_equity = self.equity_curve[-1][1]
            daily_return = (current_equity - prev_equity) / abs(prev_equity) if prev_equity != 0 else 0.0
            self.daily_returns.append((datetime.now(), daily_return))
    
    def _create_empty_metrics(self) -> PerformanceMetrics:
        """Create empty metrics when no trades exist."""
        return PerformanceMetrics(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_pnl=0.0,
            gross_profit=0.0,
            gross_loss=0.0,
            profit_factor=0.0,
            average_win=0.0,
            average_loss=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            max_drawdown_percent=0.0,
            average_trade_duration=0.0,
            total_commission=0.0,
            total_slippage=0.0
        )
    
    def get_trade_summary(self) -> Dict:
        """Get summary of all trades."""
        return {
            "total_trades": len(self.trades),
            "open_trades": len([t for t in self.trades.values() if t.status == "open"]),
            "closed_trades": len([t for t in self.trades.values() if t.status == "closed"]),
            "total_pnl": sum(t.pnl for t in self.trades.values() if t.pnl is not None),
            "trades": [asdict(t) for t in self.trades.values()]
        }
    
    def log_performance_summary(self) -> None:
        """Log comprehensive performance summary."""
        metrics = self.calculate_metrics()
        
        self.logger.info("=" * 60)
        self.logger.info("PERFORMANCE SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Total Trades: {metrics.total_trades}")
        self.logger.info(f"Win Rate: {metrics.win_rate:.2%}")
        self.logger.info(f"Total P&L: ${metrics.total_pnl:,.2f}")
        self.logger.info(f"Profit Factor: {metrics.profit_factor:.2f}")
        self.logger.info(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        self.logger.info(f"Sortino Ratio: {metrics.sortino_ratio:.2f}")
        self.logger.info(f"Max Drawdown: ${metrics.max_drawdown:,.2f} ({metrics.max_drawdown_percent:.2f}%)")
        self.logger.info(f"Average Win: ${metrics.average_win:,.2f}")
        self.logger.info(f"Average Loss: ${metrics.average_loss:,.2f}")
        self.logger.info(f"Largest Win: ${metrics.largest_win:,.2f}")
        self.logger.info(f"Largest Loss: ${metrics.largest_loss:,.2f}")
        self.logger.info(f"Total Commission: ${metrics.total_commission:,.2f}")
        self.logger.info(f"Total Slippage: ${metrics.total_slippage:,.2f}")
        self.logger.info(f"Average Trade Duration: {metrics.average_trade_duration:.1f} hours")
        
        if metrics.start_date and metrics.end_date:
            duration = metrics.end_date - metrics.start_date
            self.logger.info(f"Trading Period: {duration.days} days")
        
        self.logger.info("=" * 60)
    
    def save_performance_data(self) -> None:
        """Save performance data to files."""
        try:
            # Save metrics
            metrics = self.calculate_metrics()
            with open(self.metrics_file, 'w') as f:
                json.dump(asdict(metrics), f, indent=2, default=str)
            
            # Save equity curve
            with open(self.equity_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'equity'])
                for timestamp, equity in self.equity_curve:
                    writer.writerow([timestamp.isoformat(), equity])
            
            self.logger.info(f"Performance data saved to {self.metrics_file} and {self.equity_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving performance data: {e}")
    
    def _save_trade_to_file(self, trade: Trade) -> None:
        """Save trade to CSV file."""
        try:
            file_exists = False
            try:
                with open(self.trades_file, 'r') as f:
                    file_exists = True
            except FileNotFoundError:
                pass
            
            with open(self.trades_file, 'a', newline='') as f:
                writer = csv.writer(f)
                
                if not file_exists:
                    writer.writerow([
                        'trade_id', 'symbol', 'side', 'size', 'entry_price', 'exit_price',
                        'entry_time', 'exit_time', 'pnl', 'pnl_percent', 'commission',
                        'slippage', 'order_id', 'status'
                    ])
                
                writer.writerow([
                    trade.trade_id, trade.symbol, trade.side, trade.size,
                    trade.entry_price, trade.exit_price, trade.entry_time,
                    trade.exit_time, trade.pnl, trade.pnl_percent,
                    trade.commission, trade.slippage, trade.order_id, trade.status
                ])
                
        except Exception as e:
            self.logger.error(f"Error saving trade to file: {e}")
    
    def _update_trade_in_file(self, trade: Trade) -> None:
        """Update trade in CSV file."""
        # For simplicity, we'll just append the updated trade
        # In a production system, you might want to update the specific row
        self._save_trade_to_file(trade)
    
    def load_historical_data(self) -> bool:
        """
        Load historical performance data from files.
        
        Returns:
            True if data was loaded successfully
        """
        try:
            # Load trades from CSV
            if os.path.exists(self.trades_file):
                trades_df = pd.read_csv(self.trades_file)
                for _, row in trades_df.iterrows():
                    trade = Trade(
                        trade_id=row['trade_id'],
                        symbol=row['symbol'],
                        side=row['side'],
                        size=float(row['size']),
                        entry_price=float(row['entry_price']),
                        exit_price=float(row['exit_price']) if pd.notna(row['exit_price']) else None,
                        entry_time=pd.to_datetime(row['entry_time']) if pd.notna(row['entry_time']) else None,
                        exit_time=pd.to_datetime(row['exit_time']) if pd.notna(row['exit_time']) else None,
                        pnl=float(row['pnl']) if pd.notna(row['pnl']) else None,
                        pnl_percent=float(row['pnl_percent']) if pd.notna(row['pnl_percent']) else None,
                        commission=float(row['commission']),
                        slippage=float(row['slippage']),
                        order_id=row['order_id'] if pd.notna(row['order_id']) else None,
                        status=row['status']
                    )
                    self.trades[trade.trade_id] = trade
                
                # Update trade counter
                if self.trades:
                    max_id = max(int(tid.split('_')[1]) for tid in self.trades.keys())
                    self.trade_counter = max_id
            
            # Load equity curve
            if os.path.exists(self.equity_file):
                equity_df = pd.read_csv(self.equity_file)
                for _, row in equity_df.iterrows():
                    timestamp = pd.to_datetime(row['timestamp'])
                    equity = float(row['equity'])
                    self.equity_curve.append((timestamp, equity))
            
            self.logger.info(f"Loaded {len(self.trades)} historical trades")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading historical data: {e}")
            return False
    
    def generate_performance_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate a comprehensive performance report.
        
        Args:
            output_file: Optional file path to save the report
            
        Returns:
            Report content as string
        """
        metrics = self.calculate_metrics()
        
        # Calculate additional metrics
        total_trades = len(self.trades)
        open_trades = len([t for t in self.trades.values() if t.status == "open"])
        closed_trades = len([t for t in self.trades.values() if t.status == "closed"])
        
        # Calculate average trade duration
        durations = []
        for trade in self.trades.values():
            if trade.entry_time and trade.exit_time and trade.status == "closed":
                duration = (trade.exit_time - trade.entry_time).total_seconds() / 3600  # hours
                durations.append(duration)
        
        avg_duration_hours = np.mean(durations) if durations else 0.0
        avg_duration_days = avg_duration_hours / 24
        
        # Calculate best and worst months
        monthly_pnl = {}
        for trade in self.trades.values():
            if trade.exit_time and trade.pnl is not None:
                month_key = trade.exit_time.strftime("%Y-%m")
                monthly_pnl[month_key] = monthly_pnl.get(month_key, 0) + trade.pnl
        
        best_month = max(monthly_pnl.items(), key=lambda x: x[1]) if monthly_pnl else (None, 0)
        worst_month = min(monthly_pnl.items(), key=lambda x: x[1]) if monthly_pnl else (None, 0)
        
        # Generate report
        report = []
        report.append("=" * 80)
        report.append("GAUSSIAN CHANNEL TRADING BOT - PERFORMANCE REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Trading Summary
        report.append("TRADING SUMMARY")
        report.append("-" * 40)
        report.append(f"Total Trades: {total_trades}")
        report.append(f"Open Trades: {open_trades}")
        report.append(f"Closed Trades: {closed_trades}")
        report.append(f"Win Rate: {metrics.win_rate:.2%}")
        report.append(f"Total P&L: ${metrics.total_pnl:,.2f}")
        report.append(f"Profit Factor: {metrics.profit_factor:.2f}")
        report.append("")
        
        # Risk Metrics
        report.append("RISK METRICS")
        report.append("-" * 40)
        report.append(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        report.append(f"Sortino Ratio: {metrics.sortino_ratio:.2f}")
        report.append(f"Max Drawdown: ${metrics.max_drawdown:,.2f} ({metrics.max_drawdown_percent:.2f}%)")
        report.append("")
        
        # Trade Analysis
        report.append("TRADE ANALYSIS")
        report.append("-" * 40)
        report.append(f"Average Win: ${metrics.average_win:,.2f}")
        report.append(f"Average Loss: ${metrics.average_loss:,.2f}")
        report.append(f"Largest Win: ${metrics.largest_win:,.2f}")
        report.append(f"Largest Loss: ${metrics.largest_loss:,.2f}")
        report.append(f"Average Trade Duration: {avg_duration_days:.1f} days ({avg_duration_hours:.1f} hours)")
        report.append("")
        
        # Cost Analysis
        report.append("COST ANALYSIS")
        report.append("-" * 40)
        report.append(f"Total Commission: ${metrics.total_commission:,.2f}")
        report.append(f"Total Slippage: ${metrics.total_slippage:,.2f}")
        report.append(f"Total Costs: ${metrics.total_commission + metrics.total_slippage:,.2f}")
        report.append("")
        
        # Monthly Performance
        if monthly_pnl:
            report.append("MONTHLY PERFORMANCE")
            report.append("-" * 40)
            report.append(f"Best Month: {best_month[0]} (${best_month[1]:,.2f})")
            report.append(f"Worst Month: {worst_month[0]} (${worst_month[1]:,.2f})")
            report.append("")
        
        # Trading Period
        if metrics.start_date and metrics.end_date:
            duration = metrics.end_date - metrics.start_date
            report.append("TRADING PERIOD")
            report.append("-" * 40)
            report.append(f"Start Date: {metrics.start_date.strftime('%Y-%m-%d')}")
            report.append(f"End Date: {metrics.end_date.strftime('%Y-%m-%d')}")
            report.append(f"Duration: {duration.days} days")
            report.append("")
        
        # Recent Trades (last 10)
        if self.trades:
            report.append("RECENT TRADES (Last 10)")
            report.append("-" * 40)
            sorted_trades = sorted(self.trades.values(), key=lambda t: t.exit_time or t.entry_time, reverse=True)
            for trade in sorted_trades[:10]:
                status_icon = "🟢" if trade.status == "closed" else "🟡" if trade.status == "open" else "🔴"
                pnl_str = f"${trade.pnl:.2f}" if trade.pnl is not None else "N/A"
                report.append(f"{status_icon} {trade.trade_id}: {trade.symbol} {trade.side.upper()} "
                            f"${trade.size:,.0f} @ ${trade.entry_price:.2f} → {pnl_str}")
        
        report.append("")
        report.append("=" * 80)
        
        report_content = "\n".join(report)
        
        # Save to file if specified
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    f.write(report_content)
                self.logger.info(f"Performance report saved to {output_file}")
            except Exception as e:
                self.logger.error(f"Error saving performance report: {e}")
        
        return report_content
    
    def get_monthly_returns(self) -> Dict[str, float]:
        """
        Get monthly returns for analysis.
        
        Returns:
            Dictionary with month keys (YYYY-MM) and P&L values
        """
        monthly_pnl = {}
        for trade in self.trades.values():
            if trade.exit_time and trade.pnl is not None and trade.status == "closed":
                month_key = trade.exit_time.strftime("%Y-%m")
                monthly_pnl[month_key] = monthly_pnl.get(month_key, 0) + trade.pnl
        
        return monthly_pnl
    
    def get_symbol_performance(self) -> Dict[str, Dict]:
        """
        Get performance breakdown by symbol.
        
        Returns:
            Dictionary with symbol keys and performance metrics
        """
        symbol_stats = {}
        
        for trade in self.trades.values():
            if trade.status != "closed":
                continue
                
            symbol = trade.symbol
            if symbol not in symbol_stats:
                symbol_stats[symbol] = {
                    'trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'total_pnl': 0.0,
                    'total_volume': 0.0
                }
            
            symbol_stats[symbol]['trades'] += 1
            symbol_stats[symbol]['total_pnl'] += trade.pnl
            symbol_stats[symbol]['total_volume'] += trade.size
            
            if trade.pnl > 0:
                symbol_stats[symbol]['wins'] += 1
            else:
                symbol_stats[symbol]['losses'] += 1
        
        # Calculate additional metrics
        for symbol, stats in symbol_stats.items():
            stats['win_rate'] = stats['wins'] / stats['trades'] if stats['trades'] > 0 else 0.0
            stats['avg_pnl'] = stats['total_pnl'] / stats['trades'] if stats['trades'] > 0 else 0.0
        
        return symbol_stats
    
    def get_trade_analysis(self) -> Dict:
        """
        Get detailed trade analysis including trade patterns and statistics.
        
        Returns:
            Dictionary with detailed trade analysis
        """
        if not self.trades:
            return {}
        
        closed_trades = [t for t in self.trades.values() if t.status == "closed"]
        if not closed_trades:
            return {}
        
        # Trade duration analysis
        durations = []
        for trade in closed_trades:
            if trade.entry_time and trade.exit_time:
                duration = (trade.exit_time - trade.entry_time).total_seconds() / 3600  # hours
                durations.append(duration)
        
        # P&L distribution analysis
        pnl_values = [t.pnl for t in closed_trades if t.pnl is not None]
        
        # Time-based analysis
        hourly_pnl = {}
        daily_pnl = {}
        for trade in closed_trades:
            if trade.exit_time and trade.pnl is not None:
                hour = trade.exit_time.hour
                day = trade.exit_time.strftime("%A")
                
                hourly_pnl[hour] = hourly_pnl.get(hour, 0) + trade.pnl
                daily_pnl[day] = daily_pnl.get(day, 0) + trade.pnl
        
        # Best/worst performing hours and days
        best_hour = max(hourly_pnl.items(), key=lambda x: x[1]) if hourly_pnl else (None, 0)
        worst_hour = min(hourly_pnl.items(), key=lambda x: x[1]) if hourly_pnl else (None, 0)
        best_day = max(daily_pnl.items(), key=lambda x: x[1]) if daily_pnl else (None, 0)
        worst_day = min(daily_pnl.items(), key=lambda x: x[1]) if daily_pnl else (None, 0)
        
        return {
            "total_trades": len(closed_trades),
            "avg_trade_duration_hours": np.mean(durations) if durations else 0.0,
            "min_trade_duration_hours": np.min(durations) if durations else 0.0,
            "max_trade_duration_hours": np.max(durations) if durations else 0.0,
            "pnl_std": np.std(pnl_values) if pnl_values else 0.0,
            "pnl_skewness": self._calculate_skewness(pnl_values) if len(pnl_values) > 2 else 0.0,
            "best_hour": best_hour,
            "worst_hour": worst_hour,
            "best_day": best_day,
            "worst_day": worst_day,
            "hourly_pnl": hourly_pnl,
            "daily_pnl": daily_pnl
        }
    
    def _calculate_skewness(self, values: List[float]) -> float:
        """Calculate skewness of a list of values."""
        if len(values) < 3:
            return 0.0
        
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return 0.0
        
        skewness = np.mean([((x - mean) / std) ** 3 for x in values])
        return skewness
    
    def get_consecutive_wins_losses(self) -> Dict:
        """
        Get analysis of consecutive wins and losses.
        
        Returns:
            Dictionary with consecutive win/loss statistics
        """
        if not self.trades:
            return {}
        
        closed_trades = [t for t in self.trades.values() if t.status == "closed"]
        if not closed_trades:
            return {}
        
        # Sort trades by exit time
        sorted_trades = sorted(closed_trades, key=lambda t: t.exit_time or t.entry_time)
        
        current_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        current_streak_type = None
        
        for trade in sorted_trades:
            if trade.pnl is None:
                continue
            
            is_win = trade.pnl > 0
            
            if current_streak_type is None:
                current_streak_type = "win" if is_win else "loss"
                current_streak = 1
            elif (is_win and current_streak_type == "win") or (not is_win and current_streak_type == "loss"):
                current_streak += 1
            else:
                # Streak broken
                if current_streak_type == "win":
                    max_win_streak = max(max_win_streak, current_streak)
                else:
                    max_loss_streak = max(max_loss_streak, current_streak)
                
                current_streak_type = "win" if is_win else "loss"
                current_streak = 1
        
        # Check final streak
        if current_streak_type == "win":
            max_win_streak = max(max_win_streak, current_streak)
        else:
            max_loss_streak = max(max_loss_streak, current_streak)
        
        return {
            "max_consecutive_wins": max_win_streak,
            "max_consecutive_losses": max_loss_streak,
            "current_streak": current_streak,
            "current_streak_type": current_streak_type
        }
    
    def export_trades_to_csv(self, filename: str = None) -> str:
        """
        Export all trades to a CSV file.
        
        Args:
            filename: Optional filename, defaults to 'trades_export_YYYYMMDD_HHMMSS.csv'
            
        Returns:
            Path to the exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trades_export_{timestamp}.csv"
        
        filepath = os.path.join(self.config.LOGS_DIR, filename)
        
        try:
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Write header
                writer.writerow([
                    'trade_id', 'symbol', 'side', 'size', 'entry_price', 'exit_price',
                    'entry_time', 'exit_time', 'pnl', 'pnl_percent', 'commission',
                    'slippage', 'order_id', 'status'
                ])
                
                # Write trades
                for trade in self.trades.values():
                    writer.writerow([
                        trade.trade_id, trade.symbol, trade.side, trade.size,
                        trade.entry_price, trade.exit_price, trade.entry_time,
                        trade.exit_time, trade.pnl, trade.pnl_percent,
                        trade.commission, trade.slippage, trade.order_id, trade.status
                    ])
            
            self.logger.info(f"Trades exported to {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error exporting trades: {e}")
            return ""
    
    def get_performance_summary_dict(self) -> Dict:
        """
        Get performance summary as a dictionary for easy serialization.
        
        Returns:
            Dictionary with all performance metrics
        """
        metrics = self.calculate_metrics()
        trade_analysis = self.get_trade_analysis()
        consecutive_stats = self.get_consecutive_wins_losses()
        symbol_perf = self.get_symbol_performance()
        monthly_returns = self.get_monthly_returns()
        
        return {
            "metrics": {
                "total_trades": metrics.total_trades,
                "winning_trades": metrics.winning_trades,
                "losing_trades": metrics.losing_trades,
                "win_rate": metrics.win_rate,
                "total_pnl": metrics.total_pnl,
                "gross_profit": metrics.gross_profit,
                "gross_loss": metrics.gross_loss,
                "profit_factor": metrics.profit_factor,
                "average_win": metrics.average_win,
                "average_loss": metrics.average_loss,
                "largest_win": metrics.largest_win,
                "largest_loss": metrics.largest_loss,
                "sharpe_ratio": metrics.sharpe_ratio,
                "sortino_ratio": metrics.sortino_ratio,
                "max_drawdown": metrics.max_drawdown,
                "max_drawdown_percent": metrics.max_drawdown_percent,
                "average_trade_duration": metrics.average_trade_duration,
                "total_commission": metrics.total_commission,
                "total_slippage": metrics.total_slippage,
                "start_date": metrics.start_date.isoformat() if metrics.start_date else None,
                "end_date": metrics.end_date.isoformat() if metrics.end_date else None
            },
            "trade_analysis": trade_analysis,
            "consecutive_stats": consecutive_stats,
            "symbol_performance": symbol_perf,
            "monthly_returns": monthly_returns,
            "generated_at": datetime.now().isoformat()
        } 