"""
Backtesting Module

This module provides a comprehensive framework for backtesting trading strategies
and optimizing strategy parameters.
"""

from backtesting.engine import BacktestEngine, BacktestResult, Portfolio
from backtesting.optimization import (
    ParameterGrid, grid_search, random_search, 
    walk_forward_optimization, cross_validate
)
from backtesting.visualization import (
    plot_equity_curve, plot_drawdown, plot_returns_distribution,
    plot_monthly_returns, plot_trade_analysis, compare_strategies,
    plot_optimization_results
)
from backtesting.config import BacktestConfig, load_config, save_config, validate_parameters

__all__ = [
    # Engine
    'BacktestEngine',
    'BacktestResult',
    'Portfolio',
    
    # Optimization
    'ParameterGrid',
    'grid_search',
    'random_search',
    'walk_forward_optimization',
    'cross_validate',
    
    # Visualization
    'plot_equity_curve',
    'plot_drawdown',
    'plot_returns_distribution',
    'plot_monthly_returns',
    'plot_trade_analysis',
    'compare_strategies',
    'plot_optimization_results',
    
    # Configuration
    'BacktestConfig',
    'load_config',
    'save_config',
    'validate_parameters'
]