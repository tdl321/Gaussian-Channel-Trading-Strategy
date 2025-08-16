#!/usr/bin/env python3
"""
Test runner for Gaussian Channel Trading Bot
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_gaussian_filter import test_gaussian_filter
from test_signal_generation import test_signal_generation
from test_backtest_integration import test_backtest_integration


def run_all_tests():
    """Run all tests for the Gaussian Channel Trading Bot"""
    
    print("🧪 Running Gaussian Channel Trading Bot Tests")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    try:
        print("\n1. Testing Gaussian Filter...")
        test_gaussian_filter()
        tests_passed += 1
        print("   ✅ PASSED")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
    
    try:
        print("\n2. Testing Signal Generation...")
        test_signal_generation()
        tests_passed += 1
        print("   ✅ PASSED")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
    
    try:
        print("\n3. Testing Backtest Integration...")
        test_backtest_integration()
        tests_passed += 1
        print("   ✅ PASSED")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Strategy components are working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 