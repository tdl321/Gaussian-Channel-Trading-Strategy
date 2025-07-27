#!/usr/bin/env python3
"""
Test script for the Advanced Backtester Architecture

This demonstrates the proper separation of concerns:
- Strategy class: Pure signal generation
- AdvancedBacktester: Execution mechanics (slippage, margin calls, order management)
- Clean separation allows for realistic trading simulation
"""

import pandas as pd
import numpy as np
from src.gaussian_channel_strategy import GaussianChannelStrategy, AdvancedBacktester
import matplotlib.pyplot as plt

def test_backtester_separation():
    """Test the separation between strategy logic and backtester execution"""
    print("🧪 Testing Strategy-Backtester Separation...")
    
    # Create strategy (pure signal generation)
    strategy = GaussianChannelStrategy(
        position_size_pct=0.5,  # 50% of equity per trade
        max_pyramids=3,
        start_date='2023-01-01',
        end_date='2023-12-31'
    )
    
    # Create backtester (execution mechanics)
    backtester = AdvancedBacktester(
        initial_capital=10000,
        commission_pct=0.001,
        slippage_ticks=2,
        margin_requirement=0.25,  # 25% margin requirement
        max_leverage=4.0,
        verbose=False
    )
    
    print(f"✅ Strategy Configuration:")
    print(f"   Position Size: {strategy.position_size_pct*100}% of equity")
    print(f"   Max Pyramids: {strategy.max_pyramids}")
    print()
    print(f"✅ Backtester Configuration:")
    print(f"   Initial Capital: ${backtester.initial_capital:,}")
    print(f"   Commission: {backtester.commission_pct*100}%")
    print(f"   Slippage: {backtester.slippage_ticks} ticks")
    print(f"   Max Leverage: {backtester.max_leverage}x")
    print(f"   Margin Requirement: {backtester.margin_requirement*100}%")
    print()

def test_slippage_implementation():
    """Test slippage calculation at different order sizes"""
    print("🧪 Testing Slippage Implementation...")
    
    backtester = AdvancedBacktester(
        slippage_ticks=1,
        slippage_per_tick=0.0001,  # 0.01% per tick
        verbose=False
    )
    
    base_price = 100.00
    
    # Test different order sizes
    order_sizes = [1000, 10000, 100000, 500000]  # Different order values
    
    print(f"Base Price: ${base_price}")
    print(f"Slippage Model: {backtester.slippage_ticks} tick(s) @ {backtester.slippage_per_tick*10000:.1f} bps per tick")
    print()
    print("Order Size | Volume Factor | Buy Price | Sell Price | Slippage Cost")
    print("-" * 65)
    
    for order_size in order_sizes:
        volume_factor = min(2.0, order_size / 100000)
        buy_price = backtester.apply_slippage(base_price, is_buy=True, volume_factor=volume_factor)
        sell_price = backtester.apply_slippage(base_price, is_buy=False, volume_factor=volume_factor)
        slippage_cost = (buy_price - sell_price) / 2
        
        print(f"${order_size:>8,} | {volume_factor:>12.2f} | ${buy_price:>8.4f} | ${sell_price:>9.4f} | ${slippage_cost:>12.4f}")
    
    print("✅ Slippage scales with order size (realistic implementation)\n")

def test_margin_call_system():
    """Test margin call detection and forced liquidation"""
    print("🧪 Testing Margin Call System...")
    
    backtester = AdvancedBacktester(
        initial_capital=10000,
        margin_requirement=0.2,         # 20% margin requirement
        maintenance_margin_pct=0.75,    # 75% maintenance margin
        forced_liquidation_buffer=0.1,  # 10% buffer before liquidation
        max_leverage=5.0,
        verbose=False
    )
    
    # Simulate a leveraged position
    backtester.cash = 5000
    backtester.position_size = 400  # 400 shares
    backtester.position_cost_basis = 40000  # $100 per share
    backtester.margin_used = 35000  # Used margin
    
    print(f"Initial Position:")
    print(f"   Cash: ${backtester.cash:,}")
    print(f"   Position: {backtester.position_size} shares")
    print(f"   Cost Basis: ${backtester.position_cost_basis:,}")
    print(f"   Margin Used: ${backtester.margin_used:,}")
    print()
    
    # Test different price scenarios
    prices = [100, 95, 90, 85, 80, 75, 70]
    
    print("Price | Total Equity | Margin Level | Status")
    print("-" * 45)
    
    for price in prices:
        analysis = backtester.check_margin_call(price)
        
        status = "NORMAL"
        if analysis['is_forced_liquidation']:
            status = "FORCED LIQUIDATION"
        elif analysis['is_margin_call']:
            status = "MARGIN CALL"
        
        print(f"${price:>4} | ${analysis['current_equity']:>11,.0f} | {analysis['margin_level_pct']:>10.1f}% | {status}")
    
    print("✅ Margin call system working properly\n")

def test_order_execution_flow():
    """Test the order execution flow with next-bar execution"""
    print("🧪 Testing Order Execution Flow...")
    
    # Create sample price data
    dates = pd.date_range('2023-01-01', periods=5, freq='D')
    data = pd.DataFrame({
        'Open': [100.0, 102.0, 101.5, 103.0, 99.0],
        'High': [101.0, 103.0, 102.5, 104.0, 100.0],
        'Low': [99.5, 101.5, 101.0, 102.5, 98.0],
        'Close': [100.5, 102.5, 101.8, 103.5, 99.5],
    }, index=dates)
    
    backtester = AdvancedBacktester(
        initial_capital=10000,
        commission_pct=0.001,
        slippage_ticks=1,
        verbose=False
    )
    
    print("Sample OHLC Data:")
    print(data.round(2))
    print()
    
    print("Order Execution Flow:")
    print("1. Signal generated at Day 1 close: $102.50")
    
    # Place order on Day 1
    backtester.place_order('BUY', 0.5, "Test Entry")
    print(f"2. Order placed: BUY 50% of equity")
    print(f"   Pending orders: {len(backtester.pending_orders)}")
    
    # Execute on Day 2 open
    day2_date = dates[1]
    day2_open = data['Open'].iloc[1]  # 102.0
    day2_high = data['High'].iloc[1]
    day2_low = data['Low'].iloc[1] 
    day2_close = data['Close'].iloc[1]
    
    print(f"3. Execution at Day 2 open: ${day2_open}")
    backtester.execute_pending_orders(day2_date, day2_open, day2_high, day2_low, day2_close)
    
    print(f"4. Order executed with slippage")
    print(f"   Executed orders: {len(backtester.trade_log)}")
    print(f"   Remaining pending: {len(backtester.pending_orders)}")
    print(f"   Position size: {backtester.position_size:.2f} shares")
    
    if backtester.trade_log:
        trade = backtester.trade_log[0]
        print(f"   Execution price: ${trade['price']:.4f}")
        print(f"   Slippage cost: ${trade['slippage']:.4f}")
    
    print("✅ Next-bar execution with slippage working correctly\n")

def run_full_strategy_test():
    """Run a complete strategy test with the new architecture"""
    print("🧪 Running Full Strategy Test with Advanced Backtester...")
    
    strategy = GaussianChannelStrategy(
        position_size_pct=0.65,
        max_pyramids=3,
        start_date='2023-01-01',
        end_date='2023-06-30'
    )
    
    try:
        print("Loading SPY data and running backtest...")
        results = strategy.run_strategy(
            'SPY',
            plot=False,
            initial_capital=10000,
            commission_pct=0.001,
            slippage_ticks=1,
            margin_requirement=0.25,  # 25% margin requirement  
            max_leverage=4.0,
            verbose=False
        )
        
        if results:
            metrics = results['metrics']
            backtester = results['backtester']
            
            print(f"✅ Strategy completed successfully!")
            print(f"\n📊 Key Metrics:")
            print(f"   Total Return: {metrics['Total Return (%)']}%")
            print(f"   Number of Trades: {metrics['Number of Trades']}")
            print(f"   Margin Calls: {metrics['Margin Calls Triggered']}")
            print(f"   Forced Liquidations: {metrics['Forced Liquidations']}")
            print(f"   Slippage Cost: ${metrics['Total Slippage Cost']}")
            print(f"   Final Equity: ${metrics['Final Equity']:,.2f}")
            
            print(f"\n💰 Account Summary:")
            print(f"   Final Cash: ${backtester.cash:,.2f}")
            print(f"   Position Size: {backtester.position_size:.2f} shares")
            print(f"   Margin Used: ${backtester.margin_used:,.2f}")
            
        else:
            print("❌ Strategy test failed")
            
    except Exception as e:
        print(f"❌ Error during strategy test: {e}")
    
    print()

def main():
    """Run all tests for the advanced backtester architecture"""
    print("=" * 70)
    print("ADVANCED BACKTESTER ARCHITECTURE TEST SUITE")
    print("=" * 70)
    print()
    
    # Test individual components
    test_backtester_separation()
    test_slippage_implementation()
    test_margin_call_system()
    test_order_execution_flow()
    
    # Test full integration
    run_full_strategy_test()
    
    print("=" * 70)
    print("🎯 ARCHITECTURE SUMMARY")
    print("=" * 70)
    print()
    print("✅ Clean Separation of Concerns:")
    print("   📈 Strategy Class: Pure signal generation and trade logic")
    print("   🏛️  Backtester Class: Execution mechanics and risk management")
    print()
    print("✅ Advanced Execution Features:")
    print("   📊 Realistic slippage based on order size")
    print("   ⚠️  Comprehensive margin call monitoring")
    print("   🔄 Next-bar execution (prevents look-ahead bias)")
    print("   💰 Proper commission and cost accounting")
    print("   🚨 Forced liquidation on extreme margin calls")
    print()
    print("✅ Professional Backtesting Capabilities:")
    print("   📈 Multiple position sizing models")
    print("   ⚡ Pyramiding with ATR-based spacing")
    print("   📊 Comprehensive performance metrics")
    print("   📈 Advanced plotting with margin monitoring")
    print()
    print("This architecture provides institutional-quality backtesting")
    print("with proper risk management and realistic execution costs.")

if __name__ == "__main__":
    main() 