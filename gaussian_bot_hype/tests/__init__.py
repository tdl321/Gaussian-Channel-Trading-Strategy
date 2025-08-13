"""
Test module for Gaussian Channel Trading Bot.

Contains tests for core strategy components:
- Gaussian filter functionality
- Signal generation logic
- Live trading simulation
"""

from .test_gaussian_filter import test_gaussian_filter
from .test_signal_generation import test_signal_generation

__all__ = ['test_gaussian_filter', 'test_signal_generation'] 