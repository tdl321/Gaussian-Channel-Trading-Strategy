#!/usr/bin/env python3
"""
Gaussian Channel Trend-Following Bot for Hyperliquid on Render

Main entry point for the trading bot with both backtesting and live trading capabilities.
"""

import os
import sys
import asyncio
import schedule
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from hyperliquid_api import HyperliquidAPI
from strategy.gaussian_filter import GaussianChannelFilter
from strategy.signals import SignalGenerator
from execution.executor import OrderExecutor
from execution.position_manager import PositionManager
from utils.performance import PerformanceAnalyzer

# Load environment variables
load_dotenv()

class GaussianChannelBot:
    """
    Main trading bot class that orchestrates all components
    """
    
    def __init__(self, config_path=None):
        """Initialize the trading bot with configuration"""
        self.config = Config(config_path)
        self.api = HyperliquidAPI(self.config)
        self.gaussian_filter = GaussianChannelFilter(
            poles=self.config.POLES,
            period=self.config.PERIOD,
            multiplier=self.config.MULTIPLIER,
            mode_lag=self.config.MODE_LAG,
            mode_fast=self.config.MODE_FAST
        )
        self.signal_generator = SignalGenerator(self.gaussian_filter, self.config)
        self.executor = OrderExecutor(self.api, self.config)
        self.position_manager = PositionManager(self.config)
        self.performance_analyzer = PerformanceAnalyzer()
        
        # Bot state
        self.is_running = False
        self.last_signal_check = None
        
    async def initialize(self):
        """Initialize all components and establish connections"""
        print("🚀 Initializing Gaussian Channel Bot...")
        
        try:
            # Initialize API connection
            await self.api.connect()
            print("✅ API connection established")
            
            # Load historical data for warm-up
            await self.load_historical_data()
            print("✅ Historical data loaded")
            
            # Initialize position manager
            await self.position_manager.initialize(self.api)
            print("✅ Position manager initialized")
            
            print("🎯 Bot initialization complete!")
            return True
            
        except Exception as e:
            print(f"❌ Bot initialization failed: {e}")
            return False
    
    async def load_historical_data(self):
        """Load historical data for strategy warm-up"""
        try:
            # Load data from CSV if available, otherwise from API
            csv_path = os.path.join(self.config.DATA_DIR, "historical_data.csv")
            if os.path.exists(csv_path):
                data = self.signal_generator.load_csv_data(csv_path)
            else:
                # Load from API (last 300 days for warm-up)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=300)
                data = await self.api.get_historical_data(
                    self.config.SYMBOL, 
                    start_date, 
                    end_date
                )
            
            if data is not None and len(data) > 0:
                self.signal_generator.prepare_signals(data)
                print(f"📊 Loaded {len(data)} historical bars")
            else:
                print("⚠️  No historical data available")
                
        except Exception as e:
            print(f"❌ Error loading historical data: {e}")
    
    async def run_live_trading(self):
        """Main live trading loop"""
        print("🔄 Starting live trading loop...")
        self.is_running = True
        
        while self.is_running:
            try:
                # Get current market data
                current_data = await self.api.get_current_data(self.config.SYMBOL)
                if current_data is None:
                    print("⚠️  No current data available, retrying...")
                    await asyncio.sleep(self.config.DATA_RETRY_DELAY)
                    continue
                
                # Generate signals
                signals = self.signal_generator.generate_live_signals(current_data)
                
                # Execute trades based on signals
                if signals:
                    await self.execute_signals(signals, current_data)
                
                # Update position manager
                await self.position_manager.update_positions(self.api)
                
                # Check for margin calls
                await self.check_margin_calls()
                
                # Log performance metrics
                self.log_performance_metrics()
                
                # Wait for next iteration
                await asyncio.sleep(self.config.TRADING_INTERVAL)
                
            except Exception as e:
                print(f"❌ Error in live trading loop: {e}")
                await asyncio.sleep(self.config.ERROR_RETRY_DELAY)
    
    async def execute_signals(self, signals, current_data):
        """Execute trading signals"""
        for signal in signals:
            try:
                if signal['action'] == 'BUY':
                    await self.executor.place_buy_order(
                        symbol=self.config.SYMBOL,
                        size=signal['size'],
                        reason=signal['reason']
                    )
                elif signal['action'] == 'SELL':
                    await self.executor.place_sell_order(
                        symbol=self.config.SYMBOL,
                        size=signal['size'],
                        reason=signal['reason']
                    )
                    
            except Exception as e:
                print(f"❌ Error executing signal {signal}: {e}")
    
    async def check_margin_calls(self):
        """Check for margin calls and handle them"""
        try:
            margin_status = await self.position_manager.check_margin_status(self.api)
            
            if margin_status['is_margin_call']:
                print(f"⚠️  MARGIN CALL: Level {margin_status['margin_level_pct']:.1f}%")
                
                if margin_status['is_forced_liquidation']:
                    print("🚨 FORCED LIQUIDATION REQUIRED")
                    await self.executor.force_liquidate_all(self.config.SYMBOL)
                    
        except Exception as e:
            print(f"❌ Error checking margin calls: {e}")
    
    def log_performance_metrics(self):
        """Log current performance metrics"""
        try:
            metrics = self.performance_analyzer.calculate_current_metrics(
                self.position_manager.get_positions(),
                self.api.get_account_balance()
            )
            
            # Log to file
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'equity': metrics['total_equity'],
                'positions': len(metrics['open_positions']),
                'daily_pnl': metrics['daily_pnl'],
                'margin_level': metrics['margin_level_pct']
            }
            
            self.performance_analyzer.log_metrics(log_entry)
            
        except Exception as e:
            print(f"❌ Error logging performance metrics: {e}")
    
    async def run_backtest(self, start_date=None, end_date=None):
        """Run backtest using historical data"""
        print("📊 Running backtest...")
        
        try:
            # Load historical data
            if start_date is None:
                start_date = datetime.now() - timedelta(days=365)
            if end_date is None:
                end_date = datetime.now()
            
            data = await self.api.get_historical_data(
                self.config.SYMBOL, 
                start_date, 
                end_date
            )
            
            if data is None or len(data) == 0:
                print("❌ No historical data available for backtest")
                return None
            
            # Run backtest
            results = self.signal_generator.run_backtest(
                data,
                initial_capital=self.config.INITIAL_CAPITAL,
                commission_pct=self.config.COMMISSION_PCT,
                slippage_ticks=self.config.SLIPPAGE_TICKS
            )
            
            # Print results
            print("\n" + "="*50)
            print("BACKTEST RESULTS")
            print("="*50)
            for metric, value in results['metrics'].items():
                print(f"{metric:<25}: {value}")
            
            return results
            
        except Exception as e:
            print(f"❌ Backtest failed: {e}")
            return None
    
    async def shutdown(self):
        """Gracefully shutdown the bot"""
        print("🛑 Shutting down Gaussian Channel Bot...")
        self.is_running = False
        
        try:
            # Close API connection
            await self.api.disconnect()
            print("✅ API connection closed")
            
            # Save final performance metrics
            self.performance_analyzer.save_final_metrics()
            print("✅ Performance metrics saved")
            
        except Exception as e:
            print(f"❌ Error during shutdown: {e}")


async def main():
    """Main function to run the bot"""
    bot = GaussianChannelBot()
    
    try:
        # Initialize bot
        if not await bot.initialize():
            print("❌ Failed to initialize bot")
            return
        
        # Check command line arguments
        if len(sys.argv) > 1:
            if sys.argv[1] == "backtest":
                # Run backtest
                await bot.run_backtest()
            elif sys.argv[1] == "live":
                # Run live trading
                await bot.run_live_trading()
            else:
                print("Usage: python main.py [backtest|live]")
                return
        else:
            # Default to live trading
            await bot.run_live_trading()
            
    except KeyboardInterrupt:
        print("\n🛑 Received interrupt signal")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        await bot.shutdown()


if __name__ == "__main__":
    # Run the bot
    asyncio.run(main()) 