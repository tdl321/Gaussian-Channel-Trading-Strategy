"""
Logging Configuration for Gaussian Channel Trading Bot

Provides structured logging with file rotation, different log levels,
and comprehensive monitoring capabilities.
"""

import logging
import logging.handlers
import os
from datetime import datetime
from typing import Optional


class TradingBotLogger:
    """
    Comprehensive logging system for the trading bot.
    
    Provides structured logging with different levels and file rotation
    for monitoring and debugging.
    """
    
    def __init__(self, log_dir: str = "logs", log_level: str = "INFO"):
        """
        Initialize the logging system.
        
        Args:
            log_dir: Directory to store log files
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.log_dir = log_dir
        self.log_level = getattr(logging, log_level.upper())
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Configure logging
        self._setup_logging()
        
        # Get main logger
        self.logger = logging.getLogger("gaussian_bot")
        self.logger.info("Trading Bot Logger initialized")
    
    def _setup_logging(self) -> None:
        """Set up the logging configuration."""
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        # Clear any existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        root_logger.addHandler(console_handler)
        
        # Main log file handler (rotating)
        main_log_file = os.path.join(self.log_dir, "trading_bot.log")
        main_handler = logging.handlers.RotatingFileHandler(
            main_log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        main_handler.setLevel(self.log_level)
        main_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(main_handler)
        
        # Error log file handler
        error_log_file = os.path.join(self.log_dir, "errors.log")
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(error_handler)
        
        # Trade log file handler
        trade_log_file = os.path.join(self.log_dir, "trades.log")
        trade_handler = logging.handlers.RotatingFileHandler(
            trade_log_file,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        trade_handler.setLevel(logging.INFO)
        trade_handler.setFormatter(detailed_formatter)
        
        # Create trade logger
        trade_logger = logging.getLogger("trades")
        trade_logger.setLevel(logging.INFO)
        trade_logger.addHandler(trade_handler)
        trade_logger.propagate = False  # Don't propagate to root logger
        
        # Performance log file handler
        perf_log_file = os.path.join(self.log_dir, "performance.log")
        perf_handler = logging.handlers.RotatingFileHandler(
            perf_log_file,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        perf_handler.setLevel(logging.INFO)
        perf_handler.setFormatter(detailed_formatter)
        
        # Create performance logger
        perf_logger = logging.getLogger("performance")
        perf_logger.setLevel(logging.INFO)
        perf_logger.addHandler(perf_handler)
        perf_logger.propagate = False  # Don't propagate to root logger
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a logger with the specified name.
        
        Args:
            name: Logger name
            
        Returns:
            Configured logger
        """
        return logging.getLogger(name)
    
    def log_trade(self, message: str, level: str = "INFO") -> None:
        """
        Log a trade-related message.
        
        Args:
            message: Log message
            level: Log level
        """
        logger = logging.getLogger("trades")
        log_func = getattr(logger, level.lower())
        log_func(message)
    
    def log_performance(self, message: str, level: str = "INFO") -> None:
        """
        Log a performance-related message.
        
        Args:
            message: Log message
            level: Log level
        """
        logger = logging.getLogger("performance")
        log_func = getattr(logger, level.lower())
        log_func(message)
    
    def log_signal(self, symbol: str, signal: str, strength: float, 
                   price: float, timestamp: Optional[datetime] = None) -> None:
        """
        Log a trading signal.
        
        Args:
            symbol: Trading symbol
            signal: Signal type (buy/sell/close)
            strength: Signal strength (0.0 to 1.0)
            price: Current price
            timestamp: Signal timestamp
        """
        timestamp = timestamp or datetime.now()
        message = f"SIGNAL: {symbol} {signal.upper()} (strength: {strength:.2f}) @ ${price:.2f} - {timestamp}"
        self.log_trade(message, "INFO")
    
    def log_order(self, symbol: str, side: str, size: float, price: float,
                  order_id: str, status: str, timestamp: Optional[datetime] = None) -> None:
        """
        Log an order execution.
        
        Args:
            symbol: Trading symbol
            side: Order side (buy/sell)
            size: Order size
            price: Execution price
            order_id: Order ID
            status: Order status
            timestamp: Order timestamp
        """
        timestamp = timestamp or datetime.now()
        message = f"ORDER: {symbol} {side.upper()} {size} @ ${price:.2f} (ID: {order_id}) - {status} - {timestamp}"
        self.log_trade(message, "INFO")
    
    def log_position(self, symbol: str, size: float, entry_price: float,
                     current_price: float, pnl: float, timestamp: Optional[datetime] = None) -> None:
        """
        Log a position update.
        
        Args:
            symbol: Trading symbol
            size: Position size
            entry_price: Entry price
            current_price: Current price
            pnl: Unrealized P&L
            timestamp: Update timestamp
        """
        timestamp = timestamp or datetime.now()
        message = f"POSITION: {symbol} {size} @ ${entry_price:.2f} (current: ${current_price:.2f}, P&L: ${pnl:.2f}) - {timestamp}"
        self.log_trade(message, "INFO")
    
    def log_risk_alert(self, alert_type: str, message: str, 
                       severity: str = "WARNING", timestamp: Optional[datetime] = None) -> None:
        """
        Log a risk alert.
        
        Args:
            alert_type: Type of risk alert
            message: Alert message
            severity: Alert severity
            timestamp: Alert timestamp
        """
        timestamp = timestamp or datetime.now()
        full_message = f"RISK ALERT [{alert_type}]: {message} - {timestamp}"
        
        # Log to main logger
        logger = logging.getLogger("gaussian_bot")
        log_func = getattr(logger, severity.lower())
        log_func(full_message)
        
        # Also log to trade logger for tracking
        self.log_trade(full_message, severity)
    
    def log_performance_metric(self, metric_name: str, value: float, 
                              unit: str = "", timestamp: Optional[datetime] = None) -> None:
        """
        Log a performance metric.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            unit: Unit of measurement
            timestamp: Metric timestamp
        """
        timestamp = timestamp or datetime.now()
        message = f"METRIC: {metric_name} = {value:.4f}{unit} - {timestamp}"
        self.log_performance(message, "INFO")
    
    def log_error(self, error: Exception, context: str = "", 
                  timestamp: Optional[datetime] = None) -> None:
        """
        Log an error with context.
        
        Args:
            error: Exception object
            context: Error context
            timestamp: Error timestamp
        """
        timestamp = timestamp or datetime.now()
        message = f"ERROR [{context}]: {str(error)} - {timestamp}"
        
        logger = logging.getLogger("gaussian_bot")
        logger.error(message, exc_info=True)
    
    def log_startup(self, config_info: dict) -> None:
        """
        Log startup information.
        
        Args:
            config_info: Configuration information
        """
        logger = logging.getLogger("gaussian_bot")
        
        logger.info("=" * 60)
        logger.info("GAUSSIAN CHANNEL TRADING BOT STARTING")
        logger.info("=" * 60)
        logger.info(f"Startup Time: {datetime.now()}")
        logger.info(f"Log Level: {logging.getLevelName(self.log_level)}")
        logger.info(f"Log Directory: {self.log_dir}")
        
        # Log configuration (without sensitive data)
        safe_config = {k: v for k, v in config_info.items() 
                      if not any(sensitive in k.lower() for sensitive in ['key', 'secret', 'password'])}
        
        for key, value in safe_config.items():
            logger.info(f"Config - {key}: {value}")
        
        logger.info("=" * 60)
    
    def log_shutdown(self, reason: str = "Normal shutdown") -> None:
        """
        Log shutdown information.
        
        Args:
            reason: Shutdown reason
        """
        logger = logging.getLogger("gaussian_bot")
        
        logger.info("=" * 60)
        logger.info("GAUSSIAN CHANNEL TRADING BOT SHUTTING DOWN")
        logger.info("=" * 60)
        logger.info(f"Shutdown Time: {datetime.now()}")
        logger.info(f"Reason: {reason}")
        logger.info("=" * 60)
    
    def get_log_files(self) -> dict:
        """
        Get information about log files.
        
        Returns:
            Dictionary with log file information
        """
        log_files = {}
        
        for filename in os.listdir(self.log_dir):
            if filename.endswith('.log'):
                filepath = os.path.join(self.log_dir, filename)
                try:
                    stat = os.stat(filepath)
                    log_files[filename] = {
                        'size': stat.st_size,
                        'modified': datetime.fromtimestamp(stat.st_mtime),
                        'path': filepath
                    }
                except OSError:
                    continue
        
        return log_files
    
    def cleanup_old_logs(self, days_to_keep: int = 30) -> None:
        """
        Clean up old log files.
        
        Args:
            days_to_keep: Number of days to keep log files
        """
        import time
        
        cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
        deleted_count = 0
        
        for filename in os.listdir(self.log_dir):
            if filename.endswith('.log'):
                filepath = os.path.join(self.log_dir, filename)
                try:
                    if os.path.getmtime(filepath) < cutoff_time:
                        os.remove(filepath)
                        deleted_count += 1
                        self.logger.info(f"Deleted old log file: {filename}")
                except OSError:
                    continue
        
        if deleted_count > 0:
            self.logger.info(f"Cleaned up {deleted_count} old log files")


# Global logger instance
_logger_instance: Optional[TradingBotLogger] = None


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger
    """
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = TradingBotLogger()
    
    return _logger_instance.get_logger(name)


def setup_logging(log_dir: str = "logs", log_level: str = "INFO") -> TradingBotLogger:
    """
    Set up the logging system.
    
    Args:
        log_dir: Directory to store log files
        log_level: Logging level
        
    Returns:
        TradingBotLogger instance
    """
    global _logger_instance
    _logger_instance = TradingBotLogger(log_dir, log_level)
    return _logger_instance 