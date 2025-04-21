import numpy as np
import pandas as pd
import itertools
import logging
import random
import time
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from concurrent.futures import ProcessPoolExecutor, as_completed

from backtesting.engine import BacktestEngine, BacktestResult

class ParameterGrid:
    """Define parameter search space for optimization"""
    
    def __init__(self, param_grid: Dict[str, List[Any]]):
        """
        Initialize parameter grid
        
        Parameters:
        -----------
        param_grid : Dict[str, List[Any]]
            Dictionary of parameter names and possible values
        """
        self.param_grid = param_grid
        self.param_names = list(param_grid.keys())
        self.param_values = list(param_grid.values())
    
    def __iter__(self):
        """
        Iterate through parameter combinations
        
        Yields:
        -------
        Dict[str, Any]
            Parameter combination
        """
        for values in itertools.product(*self.param_values):
            yield dict(zip(self.param_names, values))
    
    def __len__(self):
        """
        Get number of parameter combinations
        
        Returns:
        --------
        int
            Number of parameter combinations
        """
        return np.prod([len(values) for values in self.param_values])
    
    def sample(self, n_samples: int) -> List[Dict[str, Any]]:
        """
        Sample random parameter combinations
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
            
        Returns:
        --------
        List[Dict[str, Any]]
            List of parameter combinations
        """
        # Convert to list to avoid repeated sampling
        all_params = list(self)
        
        # Sample parameters
        if n_samples >= len(all_params):
            return all_params
        
        return random.sample(all_params, n_samples)

def grid_search(
    strategy: Any,
    historical_data: Dict[str, pd.DataFrame],
    param_grid: Dict[str, List[Any]],
    metric: str = 'sharpe_ratio',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    initial_capital: float = 10000.0,
    commission: float = 0.0,
    slippage: float = 0.0,
    n_jobs: int = 1,
    verbose: bool = True
) -> Tuple[Dict[str, Any], BacktestResult]:
    """
    Perform grid search for parameter optimization
    
    Parameters:
    -----------
    strategy : Any
        Trading strategy class to optimize
    historical_data : Dict[str, pd.DataFrame]
        Historical market data (ticker -> DataFrame)
    param_grid : Dict[str, List[Any]]
        Dictionary of parameter names and possible values
    metric : str
        Performance metric to optimize
    start_date : str, optional
        Start date for backtest
    end_date : str, optional
        End date for backtest
    initial_capital : float
        Initial capital
    commission : float
        Commission per trade (percentage)
    slippage : float
        Slippage per trade (percentage)
    n_jobs : int
        Number of parallel jobs
    verbose : bool
        Whether to print progress
        
    Returns:
    --------
    Tuple[Dict[str, Any], BacktestResult]
        Best parameters and corresponding backtest result
    """
    logger = logging.getLogger(__name__)
    
    # Create parameter grid
    grid = ParameterGrid(param_grid)
    
    if verbose:
        logger.info(f"Grid search with {len(grid)} parameter combinations")
    
    # Initialize best result
    best_score = -float('inf')
    best_params = None
    best_result = None
    
    # Create backtest engine
    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission=commission,
        slippage=slippage
    )
    
    # Define evaluation function
    def evaluate_params(params):
        # Run backtest with parameters
        result = engine.run_backtest(
            strategy=strategy,
            historical_data=historical_data,
            start_date=start_date,
            end_date=end_date,
            parameters=params
        )
        
        # Get score
        score = result.metrics.get(metric, -float('inf'))
        
        return params, result, score
    
    # Run grid search
    if n_jobs > 1:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Submit jobs
            futures = [executor.submit(evaluate_params, params) for params in grid]
            
            # Process results as they complete
            for i, future in enumerate(as_completed(futures)):
                params, result, score = future.result()
                
                if verbose and (i + 1) % 10 == 0:
                    logger.info(f"Processed {i+1}/{len(grid)} parameter combinations")
                
                # Update best result
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_result = result
                    
                    if verbose:
                        logger.info(f"New best {metric}: {best_score:.4f} with params: {best_params}")
    else:
        # Sequential execution
        for i, params in enumerate(grid):
            params, result, score = evaluate_params(params)
            
            if verbose and (i + 1) % 10 == 0:
                logger.info(f"Processed {i+1}/{len(grid)} parameter combinations")
            
            # Update best result
            if score > best_score:
                best_score = score
                best_params = params
                best_result = result
                
                if verbose:
                    logger.info(f"New best {metric}: {best_score:.4f} with params: {best_params}")
    
    if verbose:
        logger.info(f"Best {metric}: {best_score:.4f} with params: {best_params}")
    
    return best_params, best_result

def random_search(
    strategy: Any,
    historical_data: Dict[str, pd.DataFrame],
    param_grid: Dict[str, List[Any]],
    n_iter: int = 10,
    metric: str = 'sharpe_ratio',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    initial_capital: float = 10000.0,
    commission: float = 0.0,
    slippage: float = 0.0,
    n_jobs: int = 1,
    verbose: bool = True
) -> Tuple[Dict[str, Any], BacktestResult]:
    """
    Perform random search for parameter optimization
    
    Parameters:
    -----------
    strategy : Any
        Trading strategy class to optimize
    historical_data : Dict[str, pd.DataFrame]
        Historical market data (ticker -> DataFrame)
    param_grid : Dict[str, List[Any]]
        Dictionary of parameter names and possible values
    n_iter : int
        Number of iterations (parameter combinations to try)
    metric : str
        Performance metric to optimize
    start_date : str, optional
        Start date for backtest
    end_date : str, optional
        End date for backtest
    initial_capital : float
        Initial capital
    commission : float
        Commission per trade (percentage)
    slippage : float
        Slippage per trade (percentage)
    n_jobs : int
        Number of parallel jobs
    verbose : bool
        Whether to print progress
        
    Returns:
    --------
    Tuple[Dict[str, Any], BacktestResult]
        Best parameters and corresponding backtest result
    """
    logger = logging.getLogger(__name__)
    
    # Create parameter grid
    grid = ParameterGrid(param_grid)
    
    # Sample parameters
    sampled_params = grid.sample(n_iter)
    
    if verbose:
        logger.info(f"Random search with {len(sampled_params)} parameter combinations")
    
    # Create reduced param grid with sampled parameters
    reduced_param_grid = {}
    for param_name in param_grid.keys():
        reduced_param_grid[param_name] = [params[param_name] for params in sampled_params]
    
    # Run grid search with sampled parameters
    return grid_search(
        strategy=strategy,
        historical_data=historical_data,
        param_grid=reduced_param_grid,
        metric=metric,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        commission=commission,
        slippage=slippage,
        n_jobs=n_jobs,
        verbose=verbose
    )

def walk_forward_optimization(
    strategy: Any,
    historical_data: Dict[str, pd.DataFrame],
    param_grid: Dict[str, List[Any]],
    train_size: int,
    test_size: int,
    metric: str = 'sharpe_ratio',
    step_size: Optional[int] = None,
    optimization_method: str = 'grid',
    n_iter: int = 10,
    initial_capital: float = 10000.0,
    commission: float = 0.0,
    slippage: float = 0.0,
    n_jobs: int = 1,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Perform walk-forward optimization
    
    Parameters:
    -----------
    strategy : Any
        Trading strategy class to optimize
    historical_data : Dict[str, pd.DataFrame]
        Historical market data (ticker -> DataFrame)
    param_grid : Dict[str, List[Any]]
        Dictionary of parameter names and possible values
    train_size : int
        Number of periods for training
    test_size : int
        Number of periods for testing
    metric : str
        Performance metric to optimize
    step_size : int, optional
        Number of periods to step forward (default: test_size)
    optimization_method : str
        Optimization method ('grid' or 'random')
    n_iter : int
        Number of iterations for random search
    initial_capital : float
        Initial capital
    commission : float
        Commission per trade (percentage)
    slippage : float
        Slippage per trade (percentage)
    n_jobs : int
        Number of parallel jobs
    verbose : bool
        Whether to print progress
        
    Returns:
    --------
    List[Dict[str, Any]]
        List of results for each walk-forward window
    """
    logger = logging.getLogger(__name__)
    
    # Set default step size
    if step_size is None:
        step_size = test_size
    
    # Get common dates
    dates = []
    for ticker, data in historical_data.items():
        # Ensure date is the index
        if 'date' in data.columns:
            data = data.set_index('date')
        
        # Get dates
        ticker_dates = list(data.index.astype(str))
        
        if not dates:
            dates = ticker_dates
        else:
            dates = [d for d in dates if d in ticker_dates]
    
    # Sort dates
    dates = sorted(dates)
    
    if len(dates) < train_size + test_size:
        logger.error(f"Not enough data for walk-forward optimization. Need at least {train_size + test_size} periods, but only have {len(dates)}.")
        return []
    
    # Initialize results
    results = []
    
    # Create backtest engine
    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission=commission,
        slippage=slippage
    )
    
    # Walk forward
    for i in range(0, len(dates) - train_size - test_size + 1, step_size):
        # Get train and test dates
        train_start_idx = i
        train_end_idx = i + train_size - 1
        test_start_idx = train_end_idx + 1
        test_end_idx = test_start_idx + test_size - 1
        
        train_start = dates[train_start_idx]
        train_end = dates[train_end_idx]
        test_start = dates[test_start_idx]
        test_end = dates[test_end_idx]
        
        if verbose:
            logger.info(f"Walk-forward window {i//step_size + 1}: Train {train_start} to {train_end}, Test {test_start} to {test_end}")
        
        # Optimize parameters on training data
        if optimization_method == 'grid':
            best_params, _ = grid_search(
                strategy=strategy,
                historical_data=historical_data,
                param_grid=param_grid,
                metric=metric,
                start_date=train_start,
                end_date=train_end,
                initial_capital=initial_capital,
                commission=commission,
                slippage=slippage,
                n_jobs=n_jobs,
                verbose=verbose
            )
        elif optimization_method == 'random':
            best_params, _ = random_search(
                strategy=strategy,
                historical_data=historical_data,
                param_grid=param_grid,
                n_iter=n_iter,
                metric=metric,
                start_date=train_start,
                end_date=train_end,
                initial_capital=initial_capital,
                commission=commission,
                slippage=slippage,
                n_jobs=n_jobs,
                verbose=verbose
            )
        else:
            raise ValueError(f"Unknown optimization method: {optimization_method}")
        
        # Test parameters on test data
        test_result = engine.run_backtest(
            strategy=strategy,
            historical_data=historical_data,
            start_date=test_start,
            end_date=test_end,
            parameters=best_params
        )
        
        # Store results
        window_result = {
            'window': i//step_size + 1,
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'best_params': best_params,
            'train_metric': test_result.metrics.get(metric, 0.0),
            'test_result': test_result
        }
        
        results.append(window_result)
        
        if verbose:
            logger.info(f"Window {i//step_size + 1} - Test {metric}: {test_result.metrics.get(metric, 0.0):.4f}")
    
    return results

def cross_validate(
    strategy: Any,
    historical_data: Dict[str, pd.DataFrame],
    parameters: Dict[str, Any],
    n_splits: int = 5,
    metric: str = 'sharpe_ratio',
    initial_capital: float = 10000.0,
    commission: float = 0.0,
    slippage: float = 0.0,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Perform cross-validation
    
    Parameters:
    -----------
    strategy : Any
        Trading strategy class to evaluate
    historical_data : Dict[str, pd.DataFrame]
        Historical market data (ticker -> DataFrame)
    parameters : Dict[str, Any]
        Strategy parameters
    n_splits : int
        Number of cross-validation splits
    metric : str
        Performance metric to evaluate
    initial_capital : float
        Initial capital
    commission : float
        Commission per trade (percentage)
    slippage : float
        Slippage per trade (percentage)
    verbose : bool
        Whether to print progress
        
    Returns:
    --------
    Dict[str, Any]
        Cross-validation results
    """
    logger = logging.getLogger(__name__)
    
    # Get common dates
    dates = []
    for ticker, data in historical_data.items():
        # Ensure date is the index
        if 'date' in data.columns:
            data = data.set_index('date')
        
        # Get dates
        ticker_dates = list(data.index.astype(str))
        
        if not dates:
            dates = ticker_dates
        else:
            dates = [d for d in dates if d in ticker_dates]
    
    # Sort dates
    dates = sorted(dates)
    
    if len(dates) < n_splits:
        logger.error(f"Not enough data for cross-validation. Need at least {n_splits} periods, but only have {len(dates)}.")
        return {}
    
    # Calculate split size
    split_size = len(dates) // n_splits
    
    # Initialize results
    results = []
    
    # Create backtest engine
    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission=commission,
        slippage=slippage
    )
    
    # Perform cross-validation
    for i in range(n_splits):
        # Get test indices
        test_start_idx = i * split_size
        test_end_idx = (i + 1) * split_size - 1 if i < n_splits - 1 else len(dates) - 1
        
        # Get test dates
        test_start = dates[test_start_idx]
        test_end = dates[test_end_idx]
        
        if verbose:
            logger.info(f"Cross-validation fold {i+1}/{n_splits}: Test {test_start} to {test_end}")
        
        # Run backtest on test data
        test_result = engine.run_backtest(
            strategy=strategy,
            historical_data=historical_data,
            start_date=test_start,
            end_date=test_end,
            parameters=parameters
        )
        
        # Store results
        fold_result = {
            'fold': i + 1,
            'test_start': test_start,
            'test_end': test_end,
            'test_result': test_result,
            'test_metric': test_result.metrics.get(metric, 0.0)
        }
        
        results.append(fold_result)
        
        if verbose:
            logger.info(f"Fold {i+1}/{n_splits} - Test {metric}: {test_result.metrics.get(metric, 0.0):.4f}")
    
    # Calculate aggregate metrics
    metrics = [r['test_metric'] for r in results]
    
    cv_results = {
        'mean': np.mean(metrics),
        'std': np.std(metrics),
        'min': np.min(metrics),
        'max': np.max(metrics),
        'folds': results
    }
    
    if verbose:
        logger.info(f"Cross-validation results - Mean {metric}: {cv_results['mean']:.4f} ± {cv_results['std']:.4f}")
    
    return cv_results