#!/usr/bin/env python3
"""
Simple Gaussian Channel Trading Bot using Hyperliquid SDK directly
"""

import asyncio
import sys
import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import eth_account
from eth_account import Account

from config import (
    HYPERLIQUID_API_KEY, HYPERLIQUID_SECRET_KEY, HYPERLIQUID_BASE_URL,
    SYMBOL, LEVERAGE, POLES, PERIOD, MULTIPLIER, TRADING_INTERVAL,
    print_config
)
from strategy.gaussian_filter import GaussianChannelFilter
from strategy.signals import SignalGenerator
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GaussianChannelBot:
    """
    Simple Gaussian Channel Trading Bot using Hyperliquid SDK directly
    """
    
    def __init__(self):
        """Initialize the bot"""
        self.is_running = False
        
        # Initialize Hyperliquid SDK components
        self.wallet = None
        self.exchange = None
        self.info = None
        
        # Initialize strategy components
        self.gaussian_filter = None
        self.signal_generator = None
        
        logger.info("Gaussian Channel Bot initialized")
    
    def initialize(self) -> bool:
        """Initialize the bot components"""
        try:
            # Create wallet from private key
            self.wallet = Account.from_key(HYPERLIQUID_SECRET_KEY)
            logger.info(f"Wallet initialized: {self.wallet.address}")
            
            # Initialize Hyperliquid SDK
            self.info = Info(HYPERLIQUID_BASE_URL)
            self.exchange = Exchange(
                wallet=self.wallet,
                base_url=HYPERLIQUID_BASE_URL
            )
            logger.info("Hyperliquid SDK initialized")
            
            # Initialize strategy components
            self.gaussian_filter = GaussianChannelFilter(
                poles=POLES,
                period=PERIOD,
                multiplier=MULTIPLIER
            )
            
            self.signal_generator = SignalGenerator(
                gaussian_filter=self.gaussian_filter,
                config_params={
                    'POLES': POLES,
                    'PERIOD': PERIOD,
                    'MULTIPLIER': MULTIPLIER
                }
            )
            logger.info("Strategy components initialized")
            
            # Set leverage
            self.exchange.update_leverage(LEVERAGE, SYMBOL)
            logger.info(f"Leverage set to {LEVERAGE}x for {SYMBOL}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize bot: {e}")
            return False
    
    def get_current_position(self) -> Optional[dict]:
        """Get current position for the trading symbol"""
        try:
            user_state = self.info.user_state(self.wallet.address)
            
            for position_data in user_state["assetPositions"]:
                position = position_data["position"]
                if position["coin"] == SYMBOL:
                    return {
                        'size': float(position["szi"]),
                        'entry_price': float(position["entryPx"]),
                        'unrealized_pnl': float(position["unrealizedPnl"]),
                        'position_value': float(position["positionValue"])
                    }
            
            return None  # No position
            
        except Exception as e:
            logger.error(f"Error getting position: {e}")
            return None
    
    def get_current_price(self) -> Optional[float]:
        """Get current price for the trading symbol"""
        try:
            mids = self.info.all_mids()
            coin = self.info.name_to_coin[SYMBOL]
            return float(mids[coin])
        except Exception as e:
            logger.error(f"Error getting current price: {e}")
            return None
    
    def get_daily_data(self, days: int = 200) -> Optional[list]:
        """Get daily candle data for strategy calculation"""
        try:
            end_time = int(time.time() * 1000)  # Current time in milliseconds
            start_time = end_time - (days * 24 * 60 * 60 * 1000)  # days ago
            
            candles = self.info.candles_snapshot(
                name=SYMBOL,
                interval="1d",  # Daily candles for strategy calculation
                startTime=start_time,
                endTime=end_time
            )
            
            return candles
            
        except Exception as e:
            logger.error(f"Error getting daily data: {e}")
            return None
    
    def execute_signal(self, signal: dict) -> bool:
        """Execute a trading signal using the SDK directly"""
        try:
            action = signal['action']
            reason = signal.get('reason', 'No reason provided')
            price = signal.get('price', 0)
            
            logger.info(f"Executing {action} signal: {reason} @ ${price:,.2f}")
            
            if action == 'BUY':
                # Open long position
                result = self.exchange.market_open(
                    name=SYMBOL,
                    is_buy=True,
                    sz=1.0  # Use full position size
                )
                
            elif action == 'SELL':
                # Close position
                result = self.exchange.market_close(
                    coin=SYMBOL
                )
            else:
                logger.error(f"Unknown action: {action}")
                return False
            
            if result.get("status") == "ok":
                logger.info(f"✅ {action} order executed successfully")
                return True
            else:
                logger.error(f"❌ {action} order failed: {result}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
            return False
    
    def run_trading_cycle(self):
        """Run one trading cycle - check current price vs daily bands"""
        try:
            # Get current position and price
            position = self.get_current_position()
            current_price = self.get_current_price()
            
            if current_price is None:
                logger.warning("Could not get current price, skipping cycle")
                return
            
            logger.info(f"Current price: ${current_price:,.2f}")
            if position:
                logger.info(f"Current position: {position['size']} @ ${position['entry_price']:,.2f} (PnL: ${position['unrealized_pnl']:,.2f})")
            else:
                logger.info("No current position")
            
            # Get daily data for strategy calculation (144+ days needed for Gaussian filter)
            daily_candles = self.get_daily_data(days=200)
            if not daily_candles:
                logger.warning("Could not get daily data, skipping cycle")
                return
            
            # Calculate daily bands using strategy
            df_daily = self._candles_to_dataframe(daily_candles)
            df_with_signals = self.signal_generator.prepare_signals(df_daily)
            
            # Get latest daily bands
            if len(df_with_signals) == 0:
                logger.warning("No daily signals calculated, skipping cycle")
                return
                
            latest_bands = df_with_signals.iloc[-1]
            upper_band = latest_bands['hband_current']
            
            logger.info(f"Daily upper band: ${upper_band:,.2f}")
            
            # Check current price vs daily bands for entry/exit
            if current_price > upper_band and not position:
                # ENTER signal: Current price above daily upper band
                logger.info(f"🟢 ENTRY SIGNAL: Price ${current_price:,.2f} > Daily Band ${upper_band:,.2f}")
                self.execute_signal({
                    'action': 'BUY',
                    'reason': 'Price above daily Gaussian upper band',
                    'price': current_price
                })
                
            elif current_price < upper_band and position:
                # EXIT signal: Current price below daily upper band
                logger.info(f"🔴 EXIT SIGNAL: Price ${current_price:,.2f} < Daily Band ${upper_band:,.2f}")
                self.execute_signal({
                    'action': 'SELL',
                    'reason': 'Price below daily Gaussian upper band',
                    'price': current_price
                })
            else:
                logger.info(f"⏸️  No signal: Price ${current_price:,.2f} vs Band ${upper_band:,.2f}")
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
    
    def _candles_to_dataframe(self, candles: list) -> 'pd.DataFrame':
        """Convert Hyperliquid candle data to pandas DataFrame"""
        import pandas as pd
        
        data = []
        for candle in candles:
            data.append({
                'Date': pd.to_datetime(candle['T'], unit='ms'),
                'Open': float(candle['o']),
                'High': float(candle['h']),
                'Low': float(candle['l']),
                'Close': float(candle['c']),
                'Volume': float(candle['v'])
            })
        
        df = pd.DataFrame(data)
        df.set_index('Date', inplace=True)
        return df
    
    def run(self):
        """Main trading loop"""
        logger.info("Starting Gaussian Channel Bot...")
        self.is_running = True
        
        try:
            while self.is_running:
                self.run_trading_cycle()
                
                # Wait for next cycle
                logger.info(f"Waiting {TRADING_INTERVAL} seconds until next cycle...")
                time.sleep(TRADING_INTERVAL)
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down...")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            self.is_running = False
            logger.info("Bot stopped")


def main():
    """Main function"""
    bot = GaussianChannelBot()
    
    # Initialize bot
    if not bot.initialize():
        logger.error("Failed to initialize bot")
        return
    
    # Print configuration
    print_config()
    
    # Run the bot
    bot.run()


if __name__ == "__main__":
    main() 