# Gaussian Channel Trading Bot

A simple, direct implementation of the Gaussian Channel strategy using the Hyperliquid SDK.

## Overview

This bot implements a Gaussian Channel strategy that:
- Uses a Gaussian filter to create dynamic support/resistance bands
- Enters long positions when price closes above the upper band in an uptrend
- Exits positions when price closes below the upper band
- Uses the Hyperliquid SDK directly for all execution

## Features

- **Minimal Configuration**: Only essential parameters
- **Direct SDK Integration**: Uses Hyperliquid SDK without unnecessary abstractions
- **Simple Signal Generation**: Clean Gaussian Channel implementation
- **Real-time Trading**: Live trading with configurable intervals

## Configuration

Create a `.env` file with your Hyperliquid credentials:

```env
HYPERLIQUID_API_KEY=your_api_key
HYPERLIQUID_SECRET_KEY=your_private_key
TRADING_SYMBOL=BTC
LEVERAGE=5
GAUSSIAN_POLES=6
GAUSSIAN_PERIOD=144
GAUSSIAN_MULTIPLIER=1.414
TRADING_INTERVAL=86400
```

## Strategy Parameters

- **POLES**: Number of poles for the Gaussian filter (1-9)
- **PERIOD**: Sampling period for the filter (default: 144)
- **MULTIPLIER**: Multiplier for channel bands (default: 1.414)
- **LEVERAGE**: Leverage setting for Hyperliquid (default: 5)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

## Architecture

The bot consists of only essential components:

1. **Config**: Minimal configuration management
2. **Gaussian Filter**: Core strategy implementation
3. **Signal Generator**: Signal generation logic
4. **Main Bot**: Direct SDK integration and execution

No unnecessary abstractions or wrapper code - just the strategy and direct SDK usage.

## Files

- `config.py` - Minimal configuration
- `main.py` - Main bot with direct SDK integration
- `strategy/gaussian_filter.py` - Gaussian filter implementation
- `strategy/signals.py` - Signal generation logic
- `hyperliquid/` - Hyperliquid SDK (included)

## Trading Logic

1. **Entry**: When price closes above the Gaussian upper band in an uptrend
2. **Exit**: When price closes below the Gaussian upper band
3. **Position Size**: Full position size (100% of available collateral)
4. **Risk Management**: Handled automatically by Hyperliquid

## Disclaimer

This is for educational purposes. Trading involves risk. Use at your own discretion. 