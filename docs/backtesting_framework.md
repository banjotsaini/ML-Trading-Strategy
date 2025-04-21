# Backtesting Framework

This document provides detailed information about the backtesting framework in the ML Trading Strategy system.

## Overview

The backtesting framework allows for historical performance evaluation of different trading strategies and parameter optimization. It provides a comprehensive set of tools for simulating trading strategies on historical data, calculating performance metrics, and optimizing strategy parameters.

## Components

The backtesting framework consists of several key components:

1. **Backtest Engine**: Core engine for simulating trading strategies on historical data
2. **Portfolio**: Manages portfolio state during backtesting
3. **Backtest Result**: Container for backtest results
4. **Parameter Optimization**: Tools for optimizing strategy parameters
5. **Visualization**: Tools for visualizing backtest results
6. **Configuration**: Tools for managing backtest configuration

## Getting Started

### Basic Backtest

Here's a simple example of how to run a backtest:

```python
from backtesting import BacktestEngine
from strategies.ml_trading_strategy import MLTradingStrategy
from models.random_forest import RandomForestModel

# Initialize models
models = [RandomForestModel()]

# Initialize strategy
strategy = MLTradingStrategy(
    models=models,
    initial_capital=10000.0,
    risk_percentage=0.02,
    ensemble_method='dynamic_ensemble'
)

# Create backtest engine
engine = BacktestEngine(
    initial_capital=10000.0,
    commission=0.001,
    slippage=0.001
)

# Run backtest
result = engine.run_backtest(
    strategy=strategy,
    historical_data=historical_data,
    start_date='2020-01-01',
    end_date='2023-12-31',
    parameters={
        'risk_percentage': 0.02,
        'ensemble_method': 'dynamic_ensemble',
        'signal_threshold': 0.7
    }
)

# Print results
print(f"Total Return: {result.metrics.get('total_return', 0):.2%}")
print(f"Sharpe Ratio: {result.metrics.get('sharpe_ratio', 0):.2f}")
print(f"Max Drawdown: {result.metrics.get('max_drawdown', 0):.2%}")
```

### Configuration-Based Backtest

You can also run a backtest using a configuration file:

```python
from backtesting import load_config, BacktestEngine
from strategies.ml_trading_strategy import MLTradingStrategy
from models.random_forest import RandomForestModel

# Load configuration
config = load_config('config/backtest_config.json')

# Initialize models
models = [RandomForestModel()]

# Initialize strategy
strategy = MLTradingStrategy(
    models=models,
    initial_capital=config.initial_capital,
    risk_percentage=config.parameters.get('risk_percentage', 0.02),
    ensemble_method=config.parameters.get('ensemble_method', 'dynamic_ensemble')
)

# Create backtest engine
engine = BacktestEngine(
    initial_capital=config.initial_capital,
    commission=config.commission,
    slippage=config.slippage
)

# Run backtest
result = engine.run_backtest(
    strategy=strategy,
    historical_data=historical_data,
    start_date=config.start_date,
    end_date=config.end_date,
    parameters=config.parameters
)
```

### Parameter Optimization

The framework supports parameter optimization using grid search or random search:

```python
from backtesting import grid_search
from strategies.ml_trading_strategy import MLTradingStrategy
from models.random_forest import RandomForestModel

# Initialize models
models = [RandomForestModel()]

# Initialize strategy
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
best_params, best_result = grid_search(
    strategy=strategy,
    historical_data=historical_data,
    param_grid=param_grid,
    metric='sharpe_ratio',
    start_date='2020-01-01',
    end_date='2023-12-31',
    initial_capital=10000.0,
    commission=0.001,
    slippage=0.001,
    n_jobs=1,
    verbose=True
)

print(f"Best Parameters: {best_params}")
print(f"Best Sharpe Ratio: {best_result.metrics.get('sharpe_ratio', 0):.2f}")
```

### Walk-Forward Optimization

The framework also supports walk-forward optimization:

```python
from backtesting import walk_forward_optimization
from strategies.ml_trading_strategy import MLTradingStrategy
from models.random_forest import RandomForestModel

# Initialize models
models = [RandomForestModel()]

# Initialize strategy
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

# Run walk-forward optimization
results = walk_forward_optimization(
    strategy=strategy,
    historical_data=historical_data,
    param_grid=param_grid,
    train_size=252,  # 1 year
    test_size=63,    # 3 months
    step_size=63,    # 3 months
    metric='sharpe_ratio',
    optimization_method='grid',
    initial_capital=10000.0,
    commission=0.001,
    slippage=0.001,
    n_jobs=1,
    verbose=True
)
```

### Visualization

The framework provides various visualization tools:

```python
from backtesting import (
    plot_equity_curve, plot_drawdown, plot_returns_distribution,
    plot_monthly_returns, plot_trade_analysis
)

# Plot equity curve
plot_equity_curve(result)

# Plot drawdown
plot_drawdown(result)

# Plot returns distribution
plot_returns_distribution(result)

# Plot monthly returns
plot_monthly_returns(result)

# Plot trade analysis
plot_trade_analysis(result)
```

## Backtest Engine

The `BacktestEngine` class is the core of the backtesting framework. It simulates trading strategies on historical data and calculates performance metrics.

### Parameters

- `initial_capital`: Initial capital for the backtest
- `commission`: Commission per trade (percentage)
- `slippage`: Slippage per trade (percentage)

### Methods

- `run_backtest(strategy, historical_data, start_date, end_date, parameters)`: Run a backtest

## Portfolio

The `Portfolio` class manages the portfolio state during backtesting.

### Parameters

- `initial_capital`: Initial capital for the portfolio

### Methods

- `buy(ticker, quantity, price, date)`: Execute a buy order
- `sell(ticker, quantity, price, date)`: Execute a sell order
- `update_value(market_data, date)`: Update portfolio value
- `get_equity_curve()`: Get equity curve
- `get_current_value(market_data)`: Get current portfolio value
- `get_positions()`: Get current positions
- `get_cash()`: Get current cash

## Backtest Result

The `BacktestResult` class is a container for backtest results.

### Parameters

- `strategy_name`: Name of the strategy
- `equity_curve`: Equity curve with columns: date, cash, positions_value, total_value
- `trades`: List of trades
- `parameters`: Strategy parameters

### Methods

- `get_summary()`: Get summary of backtest results

## Parameter Optimization

The framework provides several tools for parameter optimization:

### ParameterGrid

The `ParameterGrid` class defines a parameter search space for optimization.

### Methods

- `grid_search(strategy, historical_data, param_grid, metric, ...)`: Perform grid search optimization
- `random_search(strategy, historical_data, param_grid, n_iter, metric, ...)`: Perform random search optimization
- `walk_forward_optimization(strategy, historical_data, param_grid, train_size, test_size, ...)`: Perform walk-forward optimization
- `cross_validate(strategy, historical_data, parameters, n_splits, metric, ...)`: Perform cross-validation

## Visualization

The framework provides several visualization tools:

### Methods

- `plot_equity_curve(result, benchmark, figsize, title, save_path)`: Plot equity curve
- `plot_drawdown(result, figsize, title, save_path)`: Plot drawdown
- `plot_returns_distribution(result, figsize, title, save_path)`: Plot returns distribution
- `plot_monthly_returns(result, figsize, title, save_path)`: Plot monthly returns heatmap
- `plot_trade_analysis(result, figsize, title, save_path)`: Plot trade analysis
- `compare_strategies(results, figsize, title, save_path)`: Compare multiple strategies
- `plot_optimization_results(param_values, metric_values, param_names, metric_name, ...)`: Plot optimization results

## Configuration

The framework provides tools for managing backtest configuration:

### BacktestConfig

The `BacktestConfig` class is a container for backtest configuration.

### Parameters

- `strategy_name`: Name of the strategy
- `tickers`: List of tickers to backtest
- `start_date`: Start date for backtest
- `end_date`: End date for backtest
- `initial_capital`: Initial capital
- `commission`: Commission per trade (percentage)
- `slippage`: Slippage per trade (percentage)
- `parameters`: Strategy parameters
- `data_source`: Data source configuration
- `optimization`: Optimization configuration
- `output`: Output configuration

### Methods

- `load_config(config_path)`: Load configuration from file
- `save_config(config, config_path)`: Save configuration to file
- `validate_parameters(parameters, parameter_schema)`: Validate parameters against schema

## Performance Metrics

The framework calculates various performance metrics:

- **Total Return**: Total return over the backtest period
- **CAGR**: Compound Annual Growth Rate
- **Sharpe Ratio**: Risk-adjusted return
- **Sortino Ratio**: Downside risk-adjusted return
- **Max Drawdown**: Maximum drawdown
- **Volatility**: Annualized volatility
- **Calmar Ratio**: Return / Max Drawdown
- **Win Rate**: Percentage of winning trades
- **Profit Factor**: Gross profit / Gross loss
- **Average Trade**: Average profit/loss per trade

## Command-Line Interface

The framework provides a command-line interface for running backtests:

```bash
python scripts/backtest.py --config config/backtest_config.json
```

Command-line options:
- `--config`: Path to configuration file
- `--strategy`: Strategy name
- `--tickers`: Comma-separated list of tickers
- `--start-date`: Start date (YYYY-MM-DD)
- `--end-date`: End date (YYYY-MM-DD)
- `--initial-capital`: Initial capital
- `--commission`: Commission per trade (percentage)
- `--slippage`: Slippage per trade (percentage)
- `--optimize`: Run parameter optimization
- `--walk-forward`: Run walk-forward optimization
- `--log-level`: Logging level

## Examples

The framework includes several example scripts:

- `examples/backtest_example.py`: Basic backtest example
- `examples/optimization_example.py`: Parameter optimization example
- `examples/walk_forward_example.py`: Walk-forward optimization example

## Testing

The framework includes unit tests:

- `tests/test_backtesting.py`: Tests for the backtesting framework

## Best Practices

1. **Start Simple**: Begin with a simple backtest before moving to more complex optimizations
2. **Avoid Overfitting**: Be cautious of overfitting when optimizing parameters
3. **Use Walk-Forward Testing**: Use walk-forward testing to validate strategy robustness
4. **Consider Transaction Costs**: Always include realistic commission and slippage
5. **Validate Results**: Cross-validate results to ensure robustness
6. **Use Realistic Data**: Use realistic historical data with proper adjustments
7. **Avoid Look-Ahead Bias**: Ensure the strategy doesn't use future data
8. **Test Multiple Markets**: Test the strategy on multiple markets and time periods
9. **Consider Risk**: Always consider risk metrics, not just returns
10. **Document Everything**: Document all assumptions and parameters