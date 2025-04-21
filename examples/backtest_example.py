#!/usr/bin/env python
"""
Backtesting Example

This script demonstrates how to use the backtesting framework.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting import (
    BacktestEngine, BacktestResult, BacktestConfig,
    load_config, save_config, grid_search,
    plot_equity_curve, plot_drawdown, plot_returns_distribution,
    plot_monthly_returns, plot_trade_analysis
)
from models.random_forest import RandomForestModel
from models.gradient_boosting import GradientBoostingModel
from models.svm import SVMModel
from strategies.ml_trading_strategy import MLTradingStrategy

def generate_sample_data(ticker: str, start_date: str, end_date: str, seed: int = 42) -> pd.DataFrame:
    """
    Generate sample price data for demonstration
    
    Parameters:
    -----------
    ticker : str
        Ticker symbol
    start_date : str
        Start date
    end_date : str
        End date
    seed : int
        Random seed
        
    Returns:
    --------
    pd.DataFrame
        Sample price data
    """
    # Set random seed
    np.random.seed(seed)
    
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # Generate random price data
    initial_price = 100.0
    returns = np.random.normal(0.0005, 0.015, size=len(date_range))
    prices = initial_price * (1 + returns).cumprod()
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': date_range,
        'open': prices * (1 + np.random.normal(0, 0.002, size=len(date_range))),
        'high': prices * (1 + np.random.normal(0.005, 0.003, size=len(date_range))),
        'low': prices * (1 + np.random.normal(-0.005, 0.003, size=len(date_range))),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, size=len(date_range))
    })
    
    # Ensure high is always highest and low is always lowest
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)
    
    # Add adjusted close
    df['adj_close'] = df['close']
    
    return df

def simple_backtest_example():
    """Run a simple backtest example"""
    print("\n=== Simple Backtest Example ===\n")
    
    # Generate sample data
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    start_date = '2020-01-01'
    end_date = '2023-12-31'
    
    historical_data = {}
    for ticker in tickers:
        historical_data[ticker] = generate_sample_data(ticker, start_date, end_date, seed=ord(ticker[0]))
        print(f"Generated sample data for {ticker}: {len(historical_data[ticker])} rows")
    
    # Initialize models
    print("\nInitializing models...")
    models = [
        RandomForestModel(),
        GradientBoostingModel(),
        SVMModel()
    ]
    
    # Initialize strategy
    print("Initializing strategy...")
    strategy = MLTradingStrategy(
        models=models,
        initial_capital=10000.0,
        risk_percentage=0.02,
        ensemble_method='dynamic_ensemble'
    )
    
    # Create backtest engine
    print("Creating backtest engine...")
    engine = BacktestEngine(
        initial_capital=10000.0,
        commission=0.001,
        slippage=0.001
    )
    
    # Run backtest
    print("Running backtest...")
    result = engine.run_backtest(
        strategy=strategy,
        historical_data=historical_data,
        start_date=start_date,
        end_date=end_date,
        parameters={
            'risk_percentage': 0.02,
            'ensemble_method': 'dynamic_ensemble',
            'signal_threshold': 0.7
        }
    )
    
    # Print results
    print("\nBacktest Results:")
    print(f"Strategy: {result.strategy_name}")
    print(f"Initial Capital: ${result.equity_curve['total_value'].iloc[0]:.2f}")
    print(f"Final Capital: ${result.equity_curve['total_value'].iloc[-1]:.2f}")
    print(f"Total Return: {result.metrics.get('total_return', 0):.2%}")
    print(f"CAGR: {result.metrics.get('cagr', 0):.2%}")
    print(f"Sharpe Ratio: {result.metrics.get('sharpe_ratio', 0):.2f}")
    print(f"Max Drawdown: {result.metrics.get('max_drawdown', 0):.2%}")
    print(f"Number of Trades: {len(result.trades)}")
    
    # Plot results
    print("\nGenerating plots...")
    plot_equity_curve(result)
    plot_drawdown(result)
    plot_returns_distribution(result)
    plot_monthly_returns(result)
    plot_trade_analysis(result)
    
    plt.show()

def config_backtest_example():
    """Run a backtest example using configuration file"""
    print("\n=== Configuration-based Backtest Example ===\n")
    
    # Create configuration
    config = BacktestConfig(
        strategy_name="MLTradingStrategy",
        tickers=['AAPL', 'MSFT', 'GOOGL'],
        start_date='2020-01-01',
        end_date='2023-12-31',
        initial_capital=10000.0,
        commission=0.001,
        slippage=0.001,
        parameters={
            'risk_percentage': 0.02,
            'ensemble_method': 'dynamic_ensemble',
            'signal_threshold': 0.7
        },
        data_source={
            'type': 'sample',  # Use sample data for demonstration
            'data_dir': 'data'
        },
        output={
            'save_results': True,
            'output_dir': 'backtest_results',
            'plot_results': True
        }
    )
    
    # Save configuration
    config_file = 'config/example_backtest_config.json'
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    save_config(config, config_file)
    print(f"Saved configuration to {config_file}")
    
    # Load configuration
    loaded_config = load_config(config_file)
    print(f"Loaded configuration from {config_file}")
    
    # Generate sample data
    historical_data = {}
    for ticker in loaded_config.tickers:
        historical_data[ticker] = generate_sample_data(
            ticker, 
            loaded_config.start_date, 
            loaded_config.end_date, 
            seed=ord(ticker[0])
        )
        print(f"Generated sample data for {ticker}: {len(historical_data[ticker])} rows")
    
    # Initialize models
    print("\nInitializing models...")
    models = [
        RandomForestModel(),
        GradientBoostingModel(),
        SVMModel()
    ]
    
    # Initialize strategy
    print("Initializing strategy...")
    strategy = MLTradingStrategy(
        models=models,
        initial_capital=loaded_config.initial_capital,
        risk_percentage=loaded_config.parameters.get('risk_percentage', 0.02),
        ensemble_method=loaded_config.parameters.get('ensemble_method', 'dynamic_ensemble')
    )
    
    # Create backtest engine
    print("Creating backtest engine...")
    engine = BacktestEngine(
        initial_capital=loaded_config.initial_capital,
        commission=loaded_config.commission,
        slippage=loaded_config.slippage
    )
    
    # Run backtest
    print("Running backtest...")
    result = engine.run_backtest(
        strategy=strategy,
        historical_data=historical_data,
        start_date=loaded_config.start_date,
        end_date=loaded_config.end_date,
        parameters=loaded_config.parameters
    )
    
    # Print results
    print("\nBacktest Results:")
    print(f"Strategy: {result.strategy_name}")
    print(f"Initial Capital: ${result.equity_curve['total_value'].iloc[0]:.2f}")
    print(f"Final Capital: ${result.equity_curve['total_value'].iloc[-1]:.2f}")
    print(f"Total Return: {result.metrics.get('total_return', 0):.2%}")
    print(f"CAGR: {result.metrics.get('cagr', 0):.2%}")
    print(f"Sharpe Ratio: {result.metrics.get('sharpe_ratio', 0):.2f}")
    print(f"Max Drawdown: {result.metrics.get('max_drawdown', 0):.2%}")
    print(f"Number of Trades: {len(result.trades)}")
    
    # Create output directory
    output_dir = loaded_config.output.get('output_dir', 'backtest_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    if loaded_config.output.get('save_results', True):
        # Generate timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save equity curve
        equity_file = os.path.join(output_dir, f"{loaded_config.strategy_name}_{timestamp}_equity.csv")
        result.equity_curve.to_csv(equity_file, index=False)
        print(f"Saved equity curve to {equity_file}")
    
    # Plot results
    if loaded_config.output.get('plot_results', True):
        print("\nGenerating plots...")
        plot_equity_curve(result)
        plot_drawdown(result)
        plot_returns_distribution(result)
        plot_monthly_returns(result)
        plot_trade_analysis(result)
        
        plt.show()

def optimization_example():
    """Run a parameter optimization example"""
    print("\n=== Parameter Optimization Example ===\n")
    
    # Generate sample data
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    start_date = '2020-01-01'
    end_date = '2023-12-31'
    
    historical_data = {}
    for ticker in tickers:
        historical_data[ticker] = generate_sample_data(ticker, start_date, end_date, seed=ord(ticker[0]))
        print(f"Generated sample data for {ticker}: {len(historical_data[ticker])} rows")
    
    # Initialize models
    print("\nInitializing models...")
    models = [
        RandomForestModel(),
        GradientBoostingModel(),
        SVMModel()
    ]
    
    # Initialize strategy
    print("Initializing strategy...")
    strategy = MLTradingStrategy(
        models=models,
        initial_capital=10000.0,
        risk_percentage=0.02,
        ensemble_method='dynamic_ensemble'
    )
    
    # Define parameter grid
    param_grid = {
        'risk_percentage': [0.01, 0.02, 0.03, 0.05],
        'ensemble_method': ['simple_average', 'weighted_average', 'majority_vote', 'weighted_vote', 'confidence_weighted', 'dynamic_ensemble'],
        'signal_threshold': [0.5, 0.6, 0.7, 0.8]
    }
    
    # Run grid search
    print("Running grid search optimization...")
    best_params, best_result = grid_search(
        strategy=strategy,
        historical_data=historical_data,
        param_grid=param_grid,
        metric='sharpe_ratio',
        start_date=start_date,
        end_date=end_date,
        initial_capital=10000.0,
        commission=0.001,
        slippage=0.001,
        n_jobs=1,
        verbose=True
    )
    
    # Print results
    print("\nOptimization Results:")
    print(f"Best Parameters: {best_params}")
    print(f"Best Sharpe Ratio: {best_result.metrics.get('sharpe_ratio', 0):.2f}")
    print(f"Total Return: {best_result.metrics.get('total_return', 0):.2%}")
    print(f"CAGR: {best_result.metrics.get('cagr', 0):.2%}")
    print(f"Max Drawdown: {best_result.metrics.get('max_drawdown', 0):.2%}")
    print(f"Number of Trades: {len(best_result.trades)}")
    
    # Plot results
    print("\nGenerating plots...")
    plot_equity_curve(best_result, title=f"Equity Curve (Optimized Parameters)")
    plot_drawdown(best_result, title=f"Drawdown (Optimized Parameters)")
    plot_returns_distribution(best_result, title=f"Returns Distribution (Optimized Parameters)")
    plot_monthly_returns(best_result, title=f"Monthly Returns (Optimized Parameters)")
    plot_trade_analysis(best_result, title=f"Trade Analysis (Optimized Parameters)")
    
    plt.show()

def main():
    """Main function"""
    print("Backtesting Framework Example")
    print("============================")
    
    # Run examples
    simple_backtest_example()
    config_backtest_example()
    optimization_example()

if __name__ == "__main__":
    main()