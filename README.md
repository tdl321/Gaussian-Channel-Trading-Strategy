# Gaussian Channel Trading Bot

A fully-automated, trend-following trading bot using a Gaussian Channel strategy for Hyperliquid exchange.

## 🚀 Quick Start

### Running Backtests

The bot now supports backtesting on multiple cryptocurrencies with different data formats:

```bash
# Run BTC backtest (default)
python run_backtest.py

# Run ETH backtest
python run_backtest.py --symbol ETH

# Run with custom data path
python run_backtest.py --symbol BTC --data-path /path/to/custom/data.csv

# Help
python run_backtest.py --help
```

### Supported Data Formats

#### BTC Data Format
- **File**: `gaussian_bot_hype/data/btc_1d_data_2018_to_2025.csv`
- **Format**: `Open time,Open,High,Low,Close,Volume,Close time,...`
- **Period**: 2018-2025 (2772 days)

#### ETH Data Format  
- **File**: `gaussian_bot_hype/data/ETH-USD (2017-2024).csv`
- **Format**: `Date,Open,High,Low,Close,Adj Close,Volume`
- **Period**: 2017-2024 (2264 days)

### Strategy Parameters

#### BTC Parameters (Original, Stable)
- **Poles**: 6
- **Period**: 144 days
- **Multiplier**: 1.414
- **ATR Period**: 14
- **Position Size**: 100%

#### ETH Parameters (Optimized for Altcoin-like Behavior)
- **Poles**: 3 (faster response)
- **Period**: 72 (faster adaptation)
- **Multiplier**: 1.8 (wider channel)
- **ATR Period**: 7 (faster volatility)
- **Position Size**: 100%

## 📊 Project Structure

```
gaussian_bot_hype/
├── main.py                   # ✅ LIVE TRADING BOT
├── config.py                 # ✅ CONFIGURATION SYSTEM
├── strategy/
│   ├── gaussian_filter.py    # ✅ GAUSSIAN FILTER ENGINE
│   ├── signals.py            # ✅ SIGNAL GENERATION
│   └── backtest.py           # ✅ BACKTESTING ENGINE
├── hyperliquid/              # ✅ FULL SDK INTEGRATION
│   ├── exchange.py           # Trading execution
│   ├── info.py              # Market data
│   └── utils/               # Authentication & types
├── tests/                    # ✅ TEST SUITE
├── data/
│   ├── btc_1d_data_2018_to_2025.csv  # ✅ BTC Historical Data
│   └── ETH-USD (2017-2024).csv       # ✅ ETH Historical Data
├── logs/                    # ✅ Log files
└── results/                 # ✅ Backtest results
```

## 🎯 Core Strategy

### Gaussian Channel Strategy
- **Type**: Long-only trend-following strategy
- **Core Signal**: When price closes above dynamic Gaussian upper band
- **Exit**: When price closes below the upper band (dynamic stop-loss)
- **Position Size**: 100% of available collateral per trade
- **No Pyramiding**: Single position only

### Strategy Parameters
- **Poles**: Filter complexity (3-9, default: 6)
- **Period**: Sampling period for Gaussian filter (default: 144 days)
- **Multiplier**: Channel width multiplier (default: 1.414)
- **ATR Period**: Average True Range period (default: 14)

## 🔧 Configuration

### Environment Variables (.env)
```env
# API Credentials
HYPERLIQUID_API_KEY=your_api_key_here
HYPERLIQUID_SECRET_KEY=your_private_key_here
HYPERLIQUID_BASE_URL=https://api.hyperliquid.xyz

# Trading Parameters
TRADING_SYMBOL=BTC
LEVERAGE=5

# Strategy Parameters
GAUSSIAN_POLES=6
GAUSSIAN_PERIOD=144
GAUSSIAN_MULTIPLIER=1.414

# Timing
TRADING_INTERVAL=3600  # 1 hour

# Logging
LOG_LEVEL=INFO
```

## 📈 Performance Metrics

The backtest system provides comprehensive performance analysis:

- **Total Return**: Overall percentage gain/loss
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Average Duration**: Average trade holding period
- **Total Trades**: Number of completed trades

## 🧪 Testing

### Unit Tests
```bash
cd gaussian_bot_hype/tests
python run_tests.py
```

### Backtesting
```bash
# Test on BTC data
python run_backtest.py --symbol BTC

# Test on ETH data  
python run_backtest.py --symbol ETH
```

## 🚀 Live Trading

### Prerequisites
1. Hyperliquid account with API access
2. API key and secret configured in `.env`
3. Sufficient collateral for trading

### Start Live Trading
```bash
python gaussian_bot_hype/main.py
```

## 📝 Pine Script Strategies

The project includes optimized Pine Script strategies for TradingView:

- **`docs/BTC_gaussian_channel_strategy.pine`**: Original strategy for major cryptocurrencies
- **`docs/ALT_gaussian_channel_strategy.pine`**: Speed-optimized for altcoins

### Strategy Differences

#### BTC Strategy (Original)
- **Poles**: 6
- **Period**: 144 days
- **Multiplier**: 1.414
- **ATR**: 14 days
- **Entry/Exit**: Close-based for reliability

#### ALT Strategy (Speed Optimized)
- **Poles**: 3 (faster response)
- **Period**: 144 days
- **Multiplier**: 2.0 (wider channel)
- **ATR**: 7 days (faster volatility)
- **Entry/Exit**: Close-based for data reliability
- **No Repainting**: `calc_on_every_tick=false`

## 🔍 Data Quality Considerations

### Major Cryptocurrencies (BTC, ETH)
- **Data Quality**: High, consistent
- **Entry/Exit**: Close-based works reliably
- **Touch-based**: May work but less reliable

### Altcoins (SOL, etc.)
- **Data Quality**: Variable, may have gaps
- **Entry/Exit**: Close-based recommended for reliability
- **Touch-based**: Often unreliable due to data quality issues

## 📊 Recent Backtest Results

### BTC Performance (2018-2025)
- **Total Return**: +318,532% (from $10,000 to $31.8M)
- **Total Trades**: 30
- **Win Rate**: 46.7%
- **Average Duration**: 36 days

### ETH Performance (2017-2024) - OPTIMIZED PARAMETERS
- **Total Return**: +2,183% (from $10,000 to $228,290)
- **Total Trades**: 65
- **Win Rate**: 35.4%
- **Average Duration**: 9.0 days
- **Strategy**: Optimized parameters (poles=3, period=72, multiplier=1.8, atr=7)

### ETH Performance (2017-2024) - ORIGINAL PARAMETERS
- **Total Return**: -79.9% (from $10,000 to $2,014)
- **Total Trades**: 39
- **Win Rate**: 28.2%
- **Average Duration**: 21.6 days
- **Strategy**: Original parameters (poles=6, period=144, multiplier=1.414, atr=14)

*Note: ETH optimization shows +2,263% improvement in total return, demonstrating the critical importance of parameter optimization for different asset classes*

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Trading cryptocurrencies involves substantial risk of loss. Use at your own risk.
