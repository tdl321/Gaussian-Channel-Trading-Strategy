# Gaussian Channel Trading Bot for Hyperliquid

A fully-automated, trend-following trading bot using a Gaussian Channel strategy for the Hyperliquid exchange. The bot implements a long-only strategy with dynamic exit logic and supports both backtesting and live deployment.

## 🎯 Strategy Overview

- **Type**: Long-only trend-following strategy
- **Core Signal**: When price closes above dynamic Gaussian upper band and filter is trending up
- **Exit**: When price closes below the upper band (dynamic stop-loss)
- **Parameters**: Poles (4-6), Sampling period (144 days), Multiplier (1.414), ATR(14) for volatility scaling

## 📁 Project Structure

```
gaussian_bot_hype/
├── main.py                   # Entry point & live trading loop
├── config.py                 # Configuration & environment loading
├── hyperliquid_api.py        # SDK wrapper for Hyperliquid integration
├── data/
│   └── historical_data.csv   # Cached historical candles
├── strategy/
│   ├── gaussian_filter.py    # Gaussian filter logic (✅ COMPLETE)
│   ├── signals.py            # Entry/exit conditions
│   └── backtest.py           # Event-driven backtest engine (✅ COMPLETE)
├── execution/
│   ├── executor.py           # Order placement & risk logic
│   └── position_manager.py   # Track open trades
├── logs/
│   └── trade_log.txt         # Logged trades
├── utils/
│   └── performance.py        # Metrics: Sharpe, drawdown, profit factor
├── tests/                    # Test files
├── examples/                 # Example scripts and demos
└── docs/                     # Documentation
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+ (required for Hyperliquid SDK compatibility)
- Hyperliquid account with API access
- Ethereum wallet for trading

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd gaussian_bot_hype
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your Hyperliquid API credentials
   ```

4. **Set up configuration**
   ```bash
   # Edit config.py or set environment variables
   export HYPERLIQUID_API_KEY="your_api_key"
   export HYPERLIQUID_SECRET_KEY="your_secret_key"
   export TRADING_SYMBOL="BTC"
   export INITIAL_CAPITAL="10000"
   ```

### Running the Bot

#### Live Trading
```bash
python main.py --live
```

#### Backtesting
```bash
python main.py --backtest --start-date 2024-01-01 --end-date 2024-12-31
```

#### Performance Demo
```bash
python examples/test_performance_demo.py
```

## ⚙️ Configuration

### Core Strategy Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `POLES` | 4 | Gaussian filter poles (4-6 recommended) |
| `PERIOD` | 144 | Sampling period in days |
| `MULTIPLIER` | 1.414 | Channel multiplier |
| `ATR_SPACING` | 0.4 | ATR spacing for volatility scaling |
| `POSITION_SIZE_PCT` | 0.65 | Position size as % of equity |

### Risk Management

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_LEVERAGE` | 5.0 | Maximum leverage |
| `MAX_DRAWDOWN_PCT` | 20.0 | Maximum drawdown % |
| `DAILY_LOSS_LIMIT` | 5.0 | Daily loss limit % |
| `MARGIN_REQUIREMENT` | 0.2 | Initial margin requirement |

### Execution Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TRADING_INTERVAL` | 60 | Trading loop interval (seconds) |
| `COMMISSION_PCT` | 0.001 | Commission rate (0.1%) |
| `SLIPPAGE_TICKS` | 1 | Slippage in ticks |

## 🔧 API Integration

The bot uses the official [Hyperliquid Python SDK](https://github.com/tdl321/hyperliquid-python-sdk) with a custom wrapper (`hyperliquid_api.py`) that provides:

- **Data Conversion**: Converts Hyperliquid candle format to pandas DataFrame
- **Standardized Interface**: Consistent API for market data and trading
- **Error Handling**: Comprehensive error handling and retry logic
- **Position Management**: Real-time position tracking

### Data Format

The bot expects standardized OHLCV data:
```python
DataFrame with columns: ['Open', 'High', 'Low', 'Close', 'Volume']
```

### Trading Interface

```python
# Market orders
api.place_market_order(symbol, is_buy, size, slippage)

# Position management
api.get_user_positions(address)
api.get_account_balance(address)
```

## 📊 Performance Tracking

The bot includes comprehensive performance tracking with:

### Core Metrics
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss
- **Sortino Ratio**: Downside deviation ratio

### Advanced Analytics
- **Symbol Performance**: Breakdown by trading symbol
- **Time-based Analysis**: Best/worst performing hours and days
- **Trade Patterns**: Duration analysis, P&L distribution
- **Consecutive Streaks**: Maximum consecutive wins and losses

### Reporting
```python
# Generate comprehensive report
tracker.generate_performance_report("performance_report.txt")

# Export trade data
tracker.export_trades_to_csv("trades_export.csv")

# Get JSON summary
summary = tracker.get_performance_summary_dict()
```

## 🧪 Testing

Run the full test suite:
```bash
python -m pytest tests/ -v
```

Run specific test categories:
```bash
# Performance tracking tests
python -m pytest tests/test_performance.py -v

# Order execution tests
python -m pytest tests/test_executor.py -v

# Position management tests
python -m pytest tests/test_position_manager.py -v
```

## 📈 Backtesting

The bot includes a comprehensive backtesting engine with:

- **Event-driven architecture**: Realistic order execution simulation
- **Margin call handling**: Automatic liquidation simulation
- **Slippage modeling**: Realistic execution costs
- **Risk management**: Position sizing and drawdown controls

### Running Backtests

```python
# Simple backtest
bot.run_backtest(start_date="2024-01-01", end_date="2024-12-31")

# With custom parameters
bot.run_backtest(
    start_date="2024-01-01",
    end_date="2024-12-31",
    initial_capital=10000,
    commission=0.001,
    slippage=0.0005
)
```

## 🚀 Deployment

### Render Deployment

The bot is designed for deployment on Render with:

- **Environment Variables**: Secure credential management
- **Persistent Storage**: Historical data and logs
- **Health Monitoring**: Performance metrics and alerts
- **Auto-restart**: Automatic recovery from failures

### Local Deployment

For local deployment, ensure:
- **API Credentials**: Valid Hyperliquid API keys
- **Network Access**: Stable internet connection
- **Storage**: 10-50MB for historical data
- **Monitoring**: Log monitoring and alerting

## 🔒 Security

- **API Keys**: Stored in environment variables
- **Private Keys**: Never hardcoded in source code
- **Network Security**: HTTPS/WSS for all communications
- **Access Control**: Limited API permissions
- **Audit Trail**: Comprehensive logging

## 📝 Logging

The bot provides structured logging for:

- **Trade Execution**: Order placement and confirmation
- **Performance Metrics**: Real-time performance tracking
- **Error Handling**: Exception logging and recovery
- **System Health**: Connection status and API responses

Log files are stored in the `logs/` directory:
- `trading.log`: Main application logs
- `trades.csv`: Individual trade records
- `performance_metrics.json`: Calculated metrics
- `equity_curve.csv`: Equity curve over time

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Trading cryptocurrencies involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results. Always test thoroughly with small amounts before live trading.

## 🔗 Links

- [Hyperliquid Exchange](https://hyperliquid.xyz/)
- [Hyperliquid Python SDK](https://github.com/tdl321/hyperliquid-python-sdk)
- [Gaussian Channel Strategy Documentation](docs/gaussian_channel_strategy.md)
- [Performance Tracker Documentation](docs/performance_tracker_summary.md) 