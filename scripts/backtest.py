#!/usr/bin/env python
"""
Backtesting Script

This script runs backtests for trading strategies.
"""

import os
import sys
import logging
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting import (
    BacktestEngine, BacktestResult, BacktestConfig,
    load_config, save_config, grid_search, random_search,
    walk_forward_optimization, cross_validate,
    plot_equity_curve, plot_drawdown, plot_returns_distribution,
    plot_monthly_returns, plot_trade_analysis, compare_strategies,
    plot_optimization_results
)
from data.data_loader import DataLoader
from strategies.ml_trading_strategy import MLTradingStrategy
from models.random_forest import RandomForestModel
from models.gradient_boosting import GradientBoostingModel
from models.svm import SVMModel

def setup_logging(log_level: str = 'INFO'):
    """
    Setup logging configuration
    
    Parameters:
    -----------
    log_level : str
        Logging level
    """
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Setup logging
    log_file = f"logs/backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_historical_data(config: BacktestConfig) -> Dict[str, pd.DataFrame]:
    """
    Load historical data
    
    Parameters:
    -----------
    config : BacktestConfig
        Backtest configuration
        
    Returns:
    --------
    Dict[str, pd.DataFrame]
        Historical market data (ticker -> DataFrame)
    """
    logger = logging.getLogger(__name__)
    logger.info("Loading historical data...")
    
    # Get data source configuration
    data_source = config.data_source
    data_source_type = data_source.get('type', 'csv')
    
    # Load data based on source type
    if data_source_type == 'csv':
        # Load from CSV files
        data_dir = data_source.get('data_dir', 'data')
        
        historical_data = {}
        for ticker in config.tickers:
            # Construct file path
            file_path = os.path.join(data_dir, f"{ticker}.csv")
            
            if not os.path.exists(file_path):
                logger.warning(f"Data file not found for {ticker}: {file_path}")
                continue
            
            # Load data
            try:
                df = pd.read_csv(file_path)
                
                # Convert date column to datetime
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                
                historical_data[ticker] = df
                logger.info(f"Loaded data for {ticker}: {len(df)} rows")
            except Exception as e:
                logger.error(f"Error loading data for {ticker}: {str(e)}")
    
    elif data_source_type == 'api':
        # Load from API
        api_key = data_source.get('api_key', '')
        
        if not api_key:
            logger.warning("API key not provided")
        
        # Use DataLoader to fetch data
        historical_data = DataLoader.fetch_historical_data(
            config.tickers,
            start_date=config.start_date,
            end_date=config.end_date,
            api_key=api_key
        )
    
    else:
        raise ValueError(f"Unsupported data source type: {data_source_type}")
    
    # Filter by date range if specified
    if config.start_date or config.end_date:
        for ticker, data in historical_data.items():
            # Ensure date is datetime
            if 'date' in data.columns and not pd.api.types.is_datetime64_any_dtype(data['date']):
                data['date'] = pd.to_datetime(data['date'])
            
            # Filter by date
            if 'date' in data.columns:
                mask = pd.Series(True, index=data.index)
                
                if config.start_date:
                    start_date = pd.to_datetime(config.start_date)
                    mask &= (data['date'] >= start_date)
                
                if config.end_date:
                    end_date = pd.to_datetime(config.end_date)
                    mask &= (data['date'] <= end_date)
                
                historical_data[ticker] = data[mask].reset_index(drop=True)
            else:
                logger.warning(f"Date column not found in data for {ticker}")
    
    return historical_data

def initialize_strategy(config: BacktestConfig) -> MLTradingStrategy:
    """
    Initialize trading strategy
    
    Parameters:
    -----------
    config : BacktestConfig
        Backtest configuration
        
    Returns:
    --------
    MLTradingStrategy
        Trading strategy
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Initializing strategy: {config.strategy_name}")
    
    # Initialize models
    models = []
    
    # Random Forest model
    logger.info("Initializing Random Forest model")
    rf_model = RandomForestModel()
    models.append(rf_model)
    
    # Gradient Boosting model
    logger.info("Initializing Gradient Boosting model")
    gb_model = GradientBoostingModel()
    models.append(gb_model)
    
    # SVM model
    logger.info("Initializing SVM model")
    svm_model = SVMModel()
    models.append(svm_model)
    
    # Initialize strategy
    strategy = MLTradingStrategy(
        models=models,
        initial_capital=config.initial_capital,
        risk_percentage=config.parameters.get('risk_percentage', 0.02),
        ensemble_method=config.parameters.get('ensemble_method', 'dynamic_ensemble')
    )
    
    return strategy

def run_backtest(config: BacktestConfig) -> BacktestResult:
    """
    Run backtest
    
    Parameters:
    -----------
    config : BacktestConfig
        Backtest configuration
        
    Returns:
    --------
    BacktestResult
        Backtest result
    """
    logger = logging.getLogger(__name__)
    logger.info("Running backtest...")
    
    # Load historical data
    historical_data = load_historical_data(config)
    
    # Initialize strategy
    strategy = initialize_strategy(config)
    
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
    
    logger.info("Backtest completed")
    
    return result

def run_optimization(config: BacktestConfig) -> Tuple[Dict[str, Any], BacktestResult]:
    """
    Run parameter optimization
    
    Parameters:
    -----------
    config : BacktestConfig
        Backtest configuration
        
    Returns:
    --------
    Tuple[Dict[str, Any], BacktestResult]
        Best parameters and corresponding backtest result
    """
    logger = logging.getLogger(__name__)
    logger.info("Running parameter optimization...")
    
    # Load historical data
    historical_data = load_historical_data(config)
    
    # Initialize strategy
    strategy = initialize_strategy(config)
    
    # Get optimization configuration
    optimization_config = config.optimization
    method = optimization_config.get('method', 'grid')
    param_grid = optimization_config.get('param_grid', {})
    metric = optimization_config.get('metric', 'sharpe_ratio')
    n_iter = optimization_config.get('n_iter', 10)
    n_jobs = optimization_config.get('n_jobs', 1)
    
    # Run optimization
    if method == 'grid':
        logger.info(f"Running grid search optimization for {metric}...")
        best_params, best_result = grid_search(
            strategy=strategy,
            historical_data=historical_data,
            param_grid=param_grid,
            metric=metric,
            start_date=config.start_date,
            end_date=config.end_date,
            initial_capital=config.initial_capital,
            commission=config.commission,
            slippage=config.slippage,
            n_jobs=n_jobs,
            verbose=True
        )
    elif method == 'random':
        logger.info(f"Running random search optimization for {metric}...")
        best_params, best_result = random_search(
            strategy=strategy,
            historical_data=historical_data,
            param_grid=param_grid,
            n_iter=n_iter,
            metric=metric,
            start_date=config.start_date,
            end_date=config.end_date,
            initial_capital=config.initial_capital,
            commission=config.commission,
            slippage=config.slippage,
            n_jobs=n_jobs,
            verbose=True
        )
    else:
        raise ValueError(f"Unsupported optimization method: {method}")
    
    logger.info(f"Optimization completed. Best {metric}: {best_result.metrics.get(metric, 0):.4f}")
    logger.info(f"Best parameters: {best_params}")
    
    return best_params, best_result

def run_walk_forward(config: BacktestConfig) -> List[Dict[str, Any]]:
    """
    Run walk-forward optimization
    
    Parameters:
    -----------
    config : BacktestConfig
        Backtest configuration
        
    Returns:
    --------
    List[Dict[str, Any]]
        Walk-forward results
    """
    logger = logging.getLogger(__name__)
    logger.info("Running walk-forward optimization...")
    
    # Load historical data
    historical_data = load_historical_data(config)
    
    # Initialize strategy
    strategy = initialize_strategy(config)
    
    # Get optimization configuration
    optimization_config = config.optimization
    method = optimization_config.get('method', 'grid')
    param_grid = optimization_config.get('param_grid', {})
    metric = optimization_config.get('metric', 'sharpe_ratio')
    n_iter = optimization_config.get('n_iter', 10)
    n_jobs = optimization_config.get('n_jobs', 1)
    
    # Get walk-forward configuration
    wf_config = optimization_config.get('walk_forward', {})
    train_size = wf_config.get('train_size', 252)  # 1 year
    test_size = wf_config.get('test_size', 63)     # 3 months
    step_size = wf_config.get('step_size', test_size)
    
    # Run walk-forward optimization
    results = walk_forward_optimization(
        strategy=strategy,
        historical_data=historical_data,
        param_grid=param_grid,
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
        metric=metric,
        optimization_method=method,
        n_iter=n_iter,
        initial_capital=config.initial_capital,
        commission=config.commission,
        slippage=config.slippage,
        n_jobs=n_jobs,
        verbose=True
    )
    
    logger.info(f"Walk-forward optimization completed with {len(results)} windows")
    
    return results

def save_results(result: BacktestResult, config: BacktestConfig):
    """
    Save backtest results
    
    Parameters:
    -----------
    result : BacktestResult
        Backtest result
    config : BacktestConfig
        Backtest configuration
    """
    logger = logging.getLogger(__name__)
    
    # Get output configuration
    output_config = config.output
    output_dir = output_config.get('output_dir', 'backtest_results')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save results
    if output_config.get('save_results', True):
        # Save summary
        summary_file = os.path.join(output_dir, f"{config.strategy_name}_{timestamp}_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(result.get_summary(), f, indent=4, default=str)
        
        logger.info(f"Saved summary to {summary_file}")
        
        # Save equity curve
        equity_file = os.path.join(output_dir, f"{config.strategy_name}_{timestamp}_equity.csv")
        result.equity_curve.to_csv(equity_file, index=False)
        
        logger.info(f"Saved equity curve to {equity_file}")
        
        # Save trades
        trades_file = os.path.join(output_dir, f"{config.strategy_name}_{timestamp}_trades.json")
        with open(trades_file, 'w') as f:
            json.dump(result.trades, f, indent=4, default=str)
        
        logger.info(f"Saved trades to {trades_file}")
    
    # Generate plots
    if output_config.get('plot_results', True):
        # Create plots directory
        plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot equity curve
        equity_plot_file = os.path.join(plots_dir, f"{config.strategy_name}_{timestamp}_equity.png")
        plot_equity_curve(result, save_path=equity_plot_file)
        
        logger.info(f"Saved equity curve plot to {equity_plot_file}")
        
        # Plot drawdown
        drawdown_plot_file = os.path.join(plots_dir, f"{config.strategy_name}_{timestamp}_drawdown.png")
        plot_drawdown(result, save_path=drawdown_plot_file)
        
        logger.info(f"Saved drawdown plot to {drawdown_plot_file}")
        
        # Plot returns distribution
        returns_plot_file = os.path.join(plots_dir, f"{config.strategy_name}_{timestamp}_returns.png")
        plot_returns_distribution(result, save_path=returns_plot_file)
        
        logger.info(f"Saved returns distribution plot to {returns_plot_file}")
        
        # Plot monthly returns
        monthly_plot_file = os.path.join(plots_dir, f"{config.strategy_name}_{timestamp}_monthly.png")
        plot_monthly_returns(result, save_path=monthly_plot_file)
        
        logger.info(f"Saved monthly returns plot to {monthly_plot_file}")
        
        # Plot trade analysis
        trades_plot_file = os.path.join(plots_dir, f"{config.strategy_name}_{timestamp}_trades.png")
        plot_trade_analysis(result, save_path=trades_plot_file)
        
        logger.info(f"Saved trade analysis plot to {trades_plot_file}")

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Backtesting Script')
    
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--strategy', type=str, help='Strategy name')
    parser.add_argument('--tickers', type=str, help='Comma-separated list of tickers')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--initial-capital', type=float, help='Initial capital')
    parser.add_argument('--commission', type=float, help='Commission per trade (percentage)')
    parser.add_argument('--slippage', type=float, help='Slippage per trade (percentage)')
    parser.add_argument('--optimize', action='store_true', help='Run parameter optimization')
    parser.add_argument('--walk-forward', action='store_true', help='Run walk-forward optimization')
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level')
    
    return parser.parse_args()

def main():
    """Main entry point"""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        if args.config:
            config = load_config(args.config)
            
            # Override configuration with command-line arguments
            if args.strategy:
                config.strategy_name = args.strategy
            
            if args.tickers:
                config.tickers = [ticker.strip() for ticker in args.tickers.split(',')]
            
            if args.start_date:
                config.start_date = args.start_date
            
            if args.end_date:
                config.end_date = args.end_date
            
            if args.initial_capital:
                config.initial_capital = args.initial_capital
            
            if args.commission:
                config.commission = args.commission
            
            if args.slippage:
                config.slippage = args.slippage
            
            if args.optimize:
                config.optimization['enabled'] = True
            
            if args.walk_forward:
                config.optimization['enabled'] = True
                config.optimization['walk_forward'] = {'enabled': True}
        else:
            # Create configuration from command-line arguments
            if not args.strategy:
                args.strategy = 'MLTradingStrategy'
            
            if not args.tickers:
                logger.error("Tickers must be specified")
                return
            
            tickers = [ticker.strip() for ticker in args.tickers.split(',')]
            
            config = BacktestConfig(
                strategy_name=args.strategy,
                tickers=tickers,
                start_date=args.start_date,
                end_date=args.end_date,
                initial_capital=args.initial_capital or 10000.0,
                commission=args.commission or 0.0,
                slippage=args.slippage or 0.0,
                optimization={'enabled': args.optimize or args.walk_forward}
            )
            
            if args.walk_forward:
                config.optimization['walk_forward'] = {'enabled': True}
        
        # Log configuration
        logger.info(f"Strategy: {config.strategy_name}")
        logger.info(f"Tickers: {config.tickers}")
        logger.info(f"Period: {config.start_date or 'start'} to {config.end_date or 'end'}")
        logger.info(f"Initial Capital: ${config.initial_capital:.2f}")
        logger.info(f"Commission: {config.commission:.4%}")
        logger.info(f"Slippage: {config.slippage:.4%}")
        
        # Run backtest or optimization
        if config.optimization.get('enabled', False):
            if config.optimization.get('walk_forward', {}).get('enabled', False):
                # Run walk-forward optimization
                results = run_walk_forward(config)
                
                # TODO: Save walk-forward results
                logger.info("Walk-forward optimization completed")
            else:
                # Run parameter optimization
                best_params, best_result = run_optimization(config)
                
                # Save results
                save_results(best_result, config)
                
                logger.info("Optimization completed")
        else:
            # Run backtest
            result = run_backtest(config)
            
            # Save results
            save_results(result, config)
            
            logger.info("Backtest completed")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())