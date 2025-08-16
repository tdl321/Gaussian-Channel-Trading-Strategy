#!/usr/bin/env python3
"""
Gaussian Channel Filter implementation converted from Pine Script
Reproduces the f_filt9x and f_pole functions EXACTLY
"""

import numpy as np
import pandas as pd


def calculate_rma(series, period):
    """
    Calculate Recursive Moving Average (RMA) / Wilder's smoothing
    This matches PineScript's ta.atr() calculation exactly
    
    RMA formula: RMA = (previous_RMA * (period-1) + current_value) / period
    This is equivalent to EMA with alpha = 1/period
    """
    alpha = 1.0 / period
    rma = series.ewm(alpha=alpha, adjust=False).mean()
    return rma


class GaussianChannelFilter:
    """
    Gaussian Channel Filter implementation converted from Pine Script
    """
    
    def __init__(self, poles=5, period=135, multiplier=2.859):
        self.poles = poles
        self.period = period
        self.multiplier = multiplier
        
        # Pre-calculate filter coefficients
        beta = (1 - np.cos(4 * np.arcsin(1) / period)) / (np.power(1.414, 2/poles) - 1)
        self.alpha = -beta + np.sqrt(np.power(beta, 2) + 2*beta)
        
        # Initialize filter state arrays
        self.filter_states_src = {}
        self.filter_states_tr = {}
        for i in range(1, 10):  # Support up to 9 poles
            self.filter_states_src[i] = np.zeros(10)
            self.filter_states_tr[i] = np.zeros(10)
    
    def _get_binomial_weights(self, pole_num):
        """Get exact binomial coefficient weights as used in Pine Script"""
        weights = {
            1: [0, 0, 0, 0, 0, 0, 0, 0],
            2: [1, 0, 0, 0, 0, 0, 0, 0],
            3: [3, 1, 0, 0, 0, 0, 0, 0],
            4: [6, 4, 1, 0, 0, 0, 0, 0],
            5: [10, 10, 5, 1, 0, 0, 0, 0],
            6: [15, 20, 15, 6, 1, 0, 0, 0],
            7: [21, 35, 35, 21, 7, 1, 0, 0],
            8: [28, 56, 70, 56, 28, 8, 1, 0],
            9: [36, 84, 126, 126, 84, 36, 9, 1]
        }
        return weights.get(pole_num, [0] * 8)
    
    def _f_filt9x(self, alpha, source_val, pole_num, filter_state):
        """
        Exact implementation of Pine Script f_filt9x function
        """
        x = 1 - alpha
        weights = self._get_binomial_weights(pole_num)
        
        # Shift previous filter values
        filter_state[1:] = filter_state[:-1]
        
        # Calculate new filter value using exact Pine Script formula
        filt_val = np.power(alpha, pole_num) * source_val
        filt_val += pole_num * x * filter_state[1]
        
        # Apply binomial weights with alternating signs
        m2, m3, m4, m5, m6, m7, m8, m9 = weights
        
        if pole_num >= 2:
            filt_val -= m2 * np.power(x, 2) * filter_state[2]
        if pole_num >= 3:
            filt_val += m3 * np.power(x, 3) * filter_state[3]
        if pole_num >= 4:
            filt_val -= m4 * np.power(x, 4) * filter_state[4]
        if pole_num >= 5:
            filt_val += m5 * np.power(x, 5) * filter_state[5]
        if pole_num >= 6:
            filt_val -= m6 * np.power(x, 6) * filter_state[6]
        if pole_num >= 7:
            filt_val += m7 * np.power(x, 7) * filter_state[7]
        if pole_num >= 8:
            filt_val -= m8 * np.power(x, 8) * filter_state[8]
        if pole_num == 9:
            filt_val += m9 * np.power(x, 9) * filter_state[9]
        
        filter_state[0] = filt_val
        return filt_val
    
    def _f_pole(self, alpha, source_series, tr_series, pole_num):
        """
        Exact implementation of Pine Script f_pole function
        """
        n = len(source_series)
        filt_src = np.full(n, np.nan)
        filt_tr = np.full(n, np.nan)
        
        # Process each data point
        for i in range(n):
            if not (np.isnan(source_series.iloc[i]) or np.isnan(tr_series.iloc[i])):
                # Calculate filter for the specified number of poles
                filt_src[i] = self._f_filt9x(alpha, source_series.iloc[i], pole_num, self.filter_states_src[pole_num])
                filt_tr[i] = self._f_filt9x(alpha, tr_series.iloc[i], pole_num, self.filter_states_tr[pole_num])
        
        return filt_src, filt_tr
    
    def apply_filter(self, source_data, true_range_data):
        """Apply exact Pine Script Gaussian filter to source and true range data"""
        # Drop NaN values but preserve index for later reindexing
        valid_mask = source_data.notna() & true_range_data.notna()
        
        if valid_mask.sum() < self.period:
            # Not enough data, return NaN series
            return (pd.Series(np.nan, index=source_data.index),
                    pd.Series(np.nan, index=source_data.index),
                    pd.Series(np.nan, index=source_data.index))
        
        # Get valid data
        valid_src = source_data[valid_mask]
        valid_tr = true_range_data[valid_mask]
        
        srcdata = valid_src.copy()
        trdata = valid_tr.copy()
        
        # Apply the exact Pine Script filter
        filt_src, filt_tr = self._f_pole(self.alpha, srcdata, trdata, self.poles)
        
        # Calculate bands
        hband = filt_src + filt_tr * self.multiplier
        lband = filt_src - filt_tr * self.multiplier
        
        # Create result series with original index
        filt_result = pd.Series(np.nan, index=source_data.index)
        hband_result = pd.Series(np.nan, index=source_data.index)
        lband_result = pd.Series(np.nan, index=source_data.index)
        
        # Fill in the calculated values
        filt_result.loc[valid_src.index] = filt_src
        hband_result.loc[valid_src.index] = hband
        lband_result.loc[valid_src.index] = lband
        
        return filt_result, hband_result, lband_result
    
    def reset_states(self):
        """Reset filter states for new data series"""
        for i in range(1, 10):
            self.filter_states_src[i] = np.zeros(10)
            self.filter_states_tr[i] = np.zeros(10) 