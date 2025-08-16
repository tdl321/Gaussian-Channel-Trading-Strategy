#!/usr/bin/env python3
"""
Test script to verify backtest.py integration
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.backtest import run_btc_backtest, run_eth_backtest


def test_backtest_integration():
    """Test that backtest functions work from tests directory"""
    
    print("🧪 Testing Backtest Integration")
    print("=" * 50)
    
    try:
        print("\n1. Testing BTC backtest function...")
        # Test with minimal data to avoid long execution
        result = run_btc_backtest(initial_cash=1000, debug_mode=True)
        if result['success']:
            print("   ✅ BTC backtest function works")
        else:
            print(f"   ❌ BTC backtest failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"   ❌ BTC backtest error: {e}")
        return False
    
    try:
        print("\n2. Testing ETH backtest function...")
        # Test with minimal data to avoid long execution
        result = run_eth_backtest(initial_cash=1000, debug_mode=True)
        if result['success']:
            print("   ✅ ETH backtest function works")
        else:
            print(f"   ❌ ETH backtest failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"   ❌ ETH backtest error: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 Backtest integration test passed!")
    return True


if __name__ == "__main__":
    success = test_backtest_integration()
    sys.exit(0 if success else 1)
