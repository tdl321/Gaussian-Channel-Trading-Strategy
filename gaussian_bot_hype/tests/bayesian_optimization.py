#!/usr/bin/env python3
"""
Gaussian Channel Strategy - Bayesian Optimization
Optimize strategy parameters using Bayesian optimization for BTC data
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

# Bayesian optimization
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt.plots import plot_convergence, plot_objective

# Import our backtest runner
from run_backtest import run_backtest, load_data_file
from gaussian_bot_hype.strategy.backtest import run_backtrader_backtest, GaussianChannelStrategy, analyze_backtrader_results


class BayesianOptimizer:
    """
    Bayesian optimization for Gaussian Channel Strategy parameters
    """
    
    def __init__(self, data, initial_cash=10000, commission=0.001, slippage_perc=0.01):
        """
        Initialize the Bayesian optimizer
        
        Args:
            data: Historical price data
            initial_cash: Starting capital
            commission: Commission rate
            slippage_perc: Slippage percentage
        """
        self.data = data
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage_perc = slippage_perc
        
        # Optimization history
        self.optimization_history = []
        self.best_params = None
        self.best_score = float('-inf')
        
        # Define the parameter space
        self.space = [
            Integer(2, 8, name='poles'),           # 2-8 poles
            Integer(48, 200, name='period'),       # 48-200 days
            Real(0.8, 3.0, name='multiplier'),     # 0.8-3.0 multiplier
            Integer(5, 21, name='atr_period')      # 5-21 ATR period
        ]
        
        print("🎯 Bayesian Optimization Initialized")
        print(f"📊 Data: {len(data)} bars from {data.index[0]} to {data.index[-1]}")
        print(f"💰 Initial Capital: ${initial_cash:,.2f}")
        print(f"📈 Parameter Space:")
        print(f"   Poles: 2-8")
        print(f"   Period: 48-200 days")
        print(f"   Multiplier: 0.8-3.0")
        print(f"   ATR Period: 5-21")
        print("=" * 80)
    
    def objective_function(self, params):
        """
        Objective function for optimization
        Maximize Sharpe ratio while minimizing drawdown
        
        Args:
            params: [poles, period, multiplier, atr_period]
            
        Returns:
            float: Negative score (minimization problem)
        """
        poles, period, multiplier, atr_period = params
        
        try:
            # Run backtest with these parameters
            strategy_params = {
                'poles': int(poles),
                'period': int(period),
                'multiplier': float(multiplier),
                'position_size_pct': 1.0,
                'atr_period': int(atr_period)
            }
            
            # Run backtest
            cerebro = run_backtrader_backtest(
                data=self.data,
                strategy_class=GaussianChannelStrategy,
                strategy_params=strategy_params,
                initial_cash=self.initial_cash,
                commission=self.commission,
                slippage_perc=self.slippage_perc
            )
            
            # Analyze results
            results = analyze_backtrader_results(cerebro)
            
            # Extract metrics
            total_return = results.get('total_return_pct', 0)
            final_value = results.get('final_value', self.initial_cash)
            initial_cash = results.get('initial_cash', self.initial_cash)
            
            # Calculate additional metrics if available
            strategy = results.get('strategy')
            if strategy:
                # Get trade statistics
                trades = getattr(strategy, 'trades', [])
                if trades:
                    # Calculate Sharpe ratio approximation
                    returns = []
                    for trade in trades:
                        if hasattr(trade, 'pnl') and hasattr(trade, 'duration'):
                            if trade.duration > 0:
                                daily_return = trade.pnl / trade.duration
                                returns.append(daily_return)
                    
                    if returns:
                        avg_return = np.mean(returns)
                        std_return = np.std(returns)
                        sharpe_ratio = avg_return / std_return if std_return > 0 else 0
                    else:
                        sharpe_ratio = 0
                else:
                    sharpe_ratio = 0
            else:
                sharpe_ratio = 0
            
            # Calculate drawdown approximation
            if final_value > initial_cash:
                # Simple drawdown calculation
                max_drawdown = max(0, (final_value - initial_cash) / initial_cash * 100)
            else:
                max_drawdown = 100
            
            # Composite score (maximize Sharpe, minimize drawdown, maximize return)
            # Higher Sharpe ratio is better, lower drawdown is better, higher return is better
            score = (sharpe_ratio * 0.4) + (total_return * 0.4) - (max_drawdown * 0.2)
            
            # Store optimization history
            optimization_result = {
                'params': strategy_params,
                'total_return': total_return,
                'final_value': final_value,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'score': score,
                'timestamp': datetime.now()
            }
            self.optimization_history.append(optimization_result)
            
            # Update best parameters
            if score > self.best_score:
                self.best_score = score
                self.best_params = strategy_params.copy()
                print(f"🏆 NEW BEST: Score={score:.4f}, Return={total_return:.2f}%, Sharpe={sharpe_ratio:.4f}, Drawdown={max_drawdown:.2f}%")
                print(f"   Params: Poles={poles}, Period={period}, Multiplier={multiplier:.2f}, ATR={atr_period}")
            
            # Return negative score (minimization problem)
            return -score
            
        except Exception as e:
            print(f"❌ Error in objective function: {e}")
            return 0  # Return neutral score on error
    
    def optimize(self, n_calls=50, n_random_starts=10):
        """
        Run Bayesian optimization
        
        Args:
            n_calls: Number of optimization iterations
            n_random_starts: Number of random initial points
            
        Returns:
            dict: Best parameters and results
        """
        print(f"🚀 Starting Bayesian Optimization")
        print(f"📊 Iterations: {n_calls}")
        print(f"🎲 Random starts: {n_random_starts}")
        print("=" * 80)
        
        start_time = time.time()
        
        # Run Bayesian optimization
        result = gp_minimize(
            func=self.objective_function,
            dimensions=self.space,
            n_calls=n_calls,
            n_random_starts=n_random_starts,
            noise=0.1,
            verbose=True
        )
        
        optimization_time = time.time() - start_time
        
        # Extract best parameters
        best_poles = int(result.x[0])
        best_period = int(result.x[1])
        best_multiplier = float(result.x[2])
        best_atr_period = int(result.x[3])
        
        print("\n" + "=" * 80)
        print("🎯 OPTIMIZATION COMPLETED")
        print("=" * 80)
        print(f"⏱️  Time: {optimization_time:.2f} seconds")
        print(f"🏆 Best Score: {-result.fun:.4f}")
        print(f"📊 Best Parameters:")
        print(f"   Poles: {best_poles}")
        print(f"   Period: {best_period}")
        print(f"   Multiplier: {best_multiplier:.3f}")
        print(f"   ATR Period: {best_atr_period}")
        
        # Run final backtest with best parameters
        print(f"\n🔄 Running final backtest with optimized parameters...")
        final_params = {
            'poles': best_poles,
            'period': best_period,
            'multiplier': best_multiplier,
            'position_size_pct': 1.0,
            'atr_period': best_atr_period
        }
        
        final_cerebro = run_backtrader_backtest(
            data=self.data,
            strategy_class=GaussianChannelStrategy,
            strategy_params=final_params,
            initial_cash=self.initial_cash,
            commission=self.commission,
            slippage_perc=self.slippage_perc
        )
        
        final_results = analyze_backtrader_results(final_cerebro)
        
        # Create optimization summary
        optimization_summary = {
            'best_params': final_params,
            'best_score': -result.fun,
            'optimization_time': optimization_time,
            'n_iterations': n_calls,
            'final_results': final_results,
            'optimization_history': self.optimization_history
        }
        
        return optimization_summary
    
    def print_optimization_history(self, top_n=10):
        """
        Print top optimization results
        
        Args:
            top_n: Number of top results to show
        """
        if not self.optimization_history:
            print("No optimization history available")
            return
        
        # Sort by score
        sorted_history = sorted(self.optimization_history, key=lambda x: x['score'], reverse=True)
        
        print(f"\n📊 TOP {min(top_n, len(sorted_history))} OPTIMIZATION RESULTS")
        print("=" * 100)
        print(f"{'Rank':<4} {'Score':<8} {'Return%':<8} {'Sharpe':<8} {'Drawdown%':<10} {'Poles':<6} {'Period':<7} {'Mult':<6} {'ATR':<4}")
        print("-" * 100)
        
        for i, result in enumerate(sorted_history[:top_n]):
            params = result['params']
            print(f"{i+1:<4} {result['score']:<8.4f} {result['total_return']:<8.2f} {result['sharpe_ratio']:<8.4f} "
                  f"{result['max_drawdown']:<10.2f} {params['poles']:<6} {params['period']:<7} "
                  f"{params['multiplier']:<6.2f} {params['atr_period']:<4}")
    
    def save_results(self, filename=None):
        """
        Save optimization results to file
        
        Args:
            filename: Output filename
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"bayesian_optimization_results_{timestamp}.json"
        
        import json
        
        # Prepare data for JSON serialization
        serializable_history = []
        for result in self.optimization_history:
            serializable_result = {
                'params': result['params'],
                'total_return': result['total_return'],
                'final_value': result['final_value'],
                'sharpe_ratio': result['sharpe_ratio'],
                'max_drawdown': result['max_drawdown'],
                'score': result['score'],
                'timestamp': result['timestamp'].isoformat()
            }
            serializable_history.append(serializable_result)
        
        results_data = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'optimization_history': serializable_history,
            'data_info': {
                'bars': len(self.data),
                'start_date': self.data.index[0].isoformat(),
                'end_date': self.data.index[-1].isoformat()
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"💾 Results saved to: {filename}")


def main():
    """Main function"""
    print("🎯 Gaussian Channel Strategy - Bayesian Optimization")
    print("📈 Optimizing parameters for BTC data")
    print("=" * 80)
    
    # Load BTC data
    print("📊 Loading BTC data...")
    data = load_data_file("gaussian_bot_hype/data/btc_1d_data_2018_to_2025.csv", "BTC")
    
    # Initialize optimizer
    optimizer = BayesianOptimizer(
        data=data,
        initial_cash=10000,
        commission=0.001,
        slippage_perc=0.01
    )
    
    # Run optimization
    print("\n🚀 Starting optimization...")
    results = optimizer.optimize(n_calls=30, n_random_starts=5)
    
    # Print results
    print("\n" + "=" * 80)
    print("📊 FINAL OPTIMIZATION RESULTS")
    print("=" * 80)
    
    final_results = results['final_results']
    best_params = results['best_params']
    
    print(f"🏆 Best Parameters Found:")
    print(f"   Poles: {best_params['poles']}")
    print(f"   Period: {best_params['period']}")
    print(f"   Multiplier: {best_params['multiplier']:.3f}")
    print(f"   ATR Period: {best_params['atr_period']}")
    
    print(f"\n📈 Performance Results:")
    print(f"   Initial Capital: ${final_results['initial_cash']:,.2f}")
    print(f"   Final Value: ${final_results['final_value']:,.2f}")
    print(f"   Total Return: {final_results['total_return_pct']:.2f}%")
    print(f"   Optimization Score: {results['best_score']:.4f}")
    
    # Print optimization history
    optimizer.print_optimization_history(top_n=10)
    
    # Save results
    optimizer.save_results()
    
    print("\n✅ Bayesian optimization completed!")
    return results


if __name__ == "__main__":
    main()
