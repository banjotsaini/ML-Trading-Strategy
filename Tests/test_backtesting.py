import unittest
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting import (
    BacktestEngine, BacktestResult, BacktestConfig,
    load_config, save_config, validate_parameters,
    ParameterGrid, grid_search, random_search
)
from models.random_forest import RandomForestModel
from strategies.ml_trading_strategy import MLTradingStrategy

class TestBacktestEngine(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        # Generate sample data
        self.tickers = ['AAPL', 'MSFT']
        self.start_date = '2020-01-01'
        self.end_date = '2020-12-31'
        
        self.historical_data = {}
        for ticker in self.tickers:
            self.historical_data[ticker] = self.generate_sample_data(ticker)
        
        # Initialize models
        self.models = [RandomForestModel()]
        
        # Initialize strategy
        self.strategy = MLTradingStrategy(
            models=self.models,
            initial_capital=10000.0,
            risk_percentage=0.02,
            ensemble_method='dynamic_ensemble'
        )
        
        # Create backtest engine
        self.engine = BacktestEngine(
            initial_capital=10000.0,
            commission=0.001,
            slippage=0.001
        )
    
    def generate_sample_data(self, ticker: str) -> pd.DataFrame:
        """Generate sample price data for testing"""
        # Set random seed
        np.random.seed(42)
        
        # Create date range
        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='B')
        
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
    
    def test_run_backtest(self):
        """Test running a backtest"""
        # Run backtest
        result = self.engine.run_backtest(
            strategy=self.strategy,
            historical_data=self.historical_data,
            start_date=self.start_date,
            end_date=self.end_date,
            parameters={
                'risk_percentage': 0.02,
                'ensemble_method': 'dynamic_ensemble',
                'signal_threshold': 0.7
            }
        )
        
        # Check result
        self.assertIsInstance(result, BacktestResult)
        self.assertEqual(result.strategy_name, self.strategy.__class__.__name__)
        self.assertIsNotNone(result.equity_curve)
        self.assertGreater(len(result.equity_curve), 0)
        self.assertIsNotNone(result.trades)
        self.assertIsNotNone(result.metrics)
        
        # Check metrics
        self.assertIn('total_return', result.metrics)
        self.assertIn('sharpe_ratio', result.metrics)
        self.assertIn('max_drawdown', result.metrics)
    
    def test_portfolio(self):
        """Test portfolio functionality"""
        # Run backtest
        result = self.engine.run_backtest(
            strategy=self.strategy,
            historical_data=self.historical_data,
            start_date=self.start_date,
            end_date=self.end_date,
            parameters={
                'risk_percentage': 0.02,
                'ensemble_method': 'dynamic_ensemble',
                'signal_threshold': 0.7
            }
        )
        
        # Check equity curve
        self.assertIn('total_value', result.equity_curve.columns)
        self.assertIn('cash', result.equity_curve.columns)
        self.assertIn('positions_value', result.equity_curve.columns)
        
        # Check initial and final values
        initial_value = result.equity_curve['total_value'].iloc[0]
        final_value = result.equity_curve['total_value'].iloc[-1]
        
        self.assertEqual(initial_value, self.engine.initial_capital)
        self.assertIsNotNone(final_value)

class TestBacktestConfig(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = BacktestConfig(
            strategy_name="MLTradingStrategy",
            tickers=['AAPL', 'MSFT'],
            start_date='2020-01-01',
            end_date='2020-12-31',
            initial_capital=10000.0,
            commission=0.001,
            slippage=0.001,
            parameters={
                'risk_percentage': 0.02,
                'ensemble_method': 'dynamic_ensemble',
                'signal_threshold': 0.7
            },
            data_source={
                'type': 'csv',
                'data_dir': 'data'
            },
            output={
                'save_results': True,
                'output_dir': 'backtest_results',
                'plot_results': True
            }
        )
    
    def test_config_to_dict(self):
        """Test converting configuration to dictionary"""
        config_dict = self.config.to_dict()
        
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict['strategy_name'], self.config.strategy_name)
        self.assertEqual(config_dict['tickers'], self.config.tickers)
        self.assertEqual(config_dict['start_date'], self.config.start_date)
        self.assertEqual(config_dict['end_date'], self.config.end_date)
        self.assertEqual(config_dict['initial_capital'], self.config.initial_capital)
        self.assertEqual(config_dict['commission'], self.config.commission)
        self.assertEqual(config_dict['slippage'], self.config.slippage)
        self.assertEqual(config_dict['parameters'], self.config.parameters)
        self.assertEqual(config_dict['data_source'], self.config.data_source)
        self.assertEqual(config_dict['output'], self.config.output)
    
    def test_config_from_dict(self):
        """Test creating configuration from dictionary"""
        config_dict = self.config.to_dict()
        new_config = BacktestConfig.from_dict(config_dict)
        
        self.assertEqual(new_config.strategy_name, self.config.strategy_name)
        self.assertEqual(new_config.tickers, self.config.tickers)
        self.assertEqual(new_config.start_date, self.config.start_date)
        self.assertEqual(new_config.end_date, self.config.end_date)
        self.assertEqual(new_config.initial_capital, self.config.initial_capital)
        self.assertEqual(new_config.commission, self.config.commission)
        self.assertEqual(new_config.slippage, self.config.slippage)
        self.assertEqual(new_config.parameters, self.config.parameters)
        self.assertEqual(new_config.data_source, self.config.data_source)
        self.assertEqual(new_config.output, self.config.output)
    
    def test_save_load_config(self):
        """Test saving and loading configuration"""
        # Create temporary file
        config_file = 'test_config.json'
        
        try:
            # Save configuration
            save_config(self.config, config_file)
            
            # Load configuration
            loaded_config = load_config(config_file)
            
            # Check loaded configuration
            self.assertEqual(loaded_config.strategy_name, self.config.strategy_name)
            self.assertEqual(loaded_config.tickers, self.config.tickers)
            self.assertEqual(loaded_config.start_date, self.config.start_date)
            self.assertEqual(loaded_config.end_date, self.config.end_date)
            self.assertEqual(loaded_config.initial_capital, self.config.initial_capital)
            self.assertEqual(loaded_config.commission, self.config.commission)
            self.assertEqual(loaded_config.slippage, self.config.slippage)
            self.assertEqual(loaded_config.parameters, self.config.parameters)
            self.assertEqual(loaded_config.data_source, self.config.data_source)
            self.assertEqual(loaded_config.output, self.config.output)
        
        finally:
            # Clean up
            if os.path.exists(config_file):
                os.remove(config_file)
    
    def test_validate_parameters(self):
        """Test parameter validation"""
        # Define parameter schema
        parameter_schema = {
            'risk_percentage': {
                'type': 'float',
                'min': 0.0,
                'max': 0.1,
                'required': True
            },
            'ensemble_method': {
                'type': 'str',
                'choices': ['simple_average', 'weighted_average', 'majority_vote', 'weighted_vote', 'confidence_weighted', 'dynamic_ensemble'],
                'required': True
            },
            'signal_threshold': {
                'type': 'float',
                'min': 0.0,
                'max': 1.0,
                'default': 0.5
            }
        }
        
        # Valid parameters
        valid_params = {
            'risk_percentage': 0.02,
            'ensemble_method': 'dynamic_ensemble',
            'signal_threshold': 0.7
        }
        
        validated_params = validate_parameters(valid_params, parameter_schema)
        self.assertEqual(validated_params, valid_params)
        
        # Missing required parameter
        invalid_params = {
            'ensemble_method': 'dynamic_ensemble',
            'signal_threshold': 0.7
        }
        
        with self.assertRaises(ValueError):
            validate_parameters(invalid_params, parameter_schema)
        
        # Invalid parameter value
        invalid_params = {
            'risk_percentage': 0.2,  # Exceeds max
            'ensemble_method': 'dynamic_ensemble',
            'signal_threshold': 0.7
        }
        
        with self.assertRaises(ValueError):
            validate_parameters(invalid_params, parameter_schema)
        
        # Invalid parameter type
        invalid_params = {
            'risk_percentage': '0.02',  # String instead of float
            'ensemble_method': 'dynamic_ensemble',
            'signal_threshold': 0.7
        }
        
        with self.assertRaises(ValueError):
            validate_parameters(invalid_params, parameter_schema)
        
        # Invalid choice
        invalid_params = {
            'risk_percentage': 0.02,
            'ensemble_method': 'invalid_method',  # Not in choices
            'signal_threshold': 0.7
        }
        
        with self.assertRaises(ValueError):
            validate_parameters(invalid_params, parameter_schema)
        
        # Default value
        params_with_default = {
            'risk_percentage': 0.02,
            'ensemble_method': 'dynamic_ensemble'
        }
        
        validated_params = validate_parameters(params_with_default, parameter_schema)
        self.assertEqual(validated_params['signal_threshold'], 0.5)  # Default value

class TestParameterOptimization(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        # Generate sample data
        self.tickers = ['AAPL']
        self.start_date = '2020-01-01'
        self.end_date = '2020-12-31'
        
        self.historical_data = {}
        for ticker in self.tickers:
            # Create date range
            date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='B')
            
            # Generate random price data
            np.random.seed(42)
            initial_price = 100.0
            returns = np.random.normal(0.0005, 0.015, size=len(date_range))
            prices = initial_price * (1 + returns).cumprod()
            
            # Create DataFrame
            df = pd.DataFrame({
                'date': date_range,
                'close': prices,
                'adj_close': prices
            })
            
            self.historical_data[ticker] = df
        
        # Initialize models
        self.models = [RandomForestModel()]
        
        # Initialize strategy
        self.strategy = MLTradingStrategy(
            models=self.models,
            initial_capital=10000.0,
            risk_percentage=0.02,
            ensemble_method='dynamic_ensemble'
        )
        
        # Define parameter grid
        self.param_grid = {
            'risk_percentage': [0.01, 0.02],
            'ensemble_method': ['simple_average', 'dynamic_ensemble'],
            'signal_threshold': [0.5, 0.7]
        }
    
    def test_parameter_grid(self):
        """Test parameter grid"""
        grid = ParameterGrid(self.param_grid)
        
        # Check length
        self.assertEqual(len(grid), 8)  # 2 x 2 x 2 = 8 combinations
        
        # Check combinations
        combinations = list(grid)
        self.assertEqual(len(combinations), 8)
        
        # Check first combination
        self.assertIn('risk_percentage', combinations[0])
        self.assertIn('ensemble_method', combinations[0])
        self.assertIn('signal_threshold', combinations[0])
    
    def test_grid_search(self):
        """Test grid search optimization"""
        # Run grid search
        best_params, best_result = grid_search(
            strategy=self.strategy,
            historical_data=self.historical_data,
            param_grid=self.param_grid,
            metric='sharpe_ratio',
            start_date=self.start_date,
            end_date=self.end_date,
            initial_capital=10000.0,
            commission=0.001,
            slippage=0.001,
            n_jobs=1,
            verbose=False
        )
        
        # Check results
        self.assertIsInstance(best_params, dict)
        self.assertIn('risk_percentage', best_params)
        self.assertIn('ensemble_method', best_params)
        self.assertIn('signal_threshold', best_params)
        
        self.assertIsInstance(best_result, BacktestResult)
        self.assertIn('sharpe_ratio', best_result.metrics)
    
    def test_random_search(self):
        """Test random search optimization"""
        # Run random search
        best_params, best_result = random_search(
            strategy=self.strategy,
            historical_data=self.historical_data,
            param_grid=self.param_grid,
            n_iter=4,  # Half of the total combinations
            metric='sharpe_ratio',
            start_date=self.start_date,
            end_date=self.end_date,
            initial_capital=10000.0,
            commission=0.001,
            slippage=0.001,
            n_jobs=1,
            verbose=False
        )
        
        # Check results
        self.assertIsInstance(best_params, dict)
        self.assertIn('risk_percentage', best_params)
        self.assertIn('ensemble_method', best_params)
        self.assertIn('signal_threshold', best_params)
        
        self.assertIsInstance(best_result, BacktestResult)
        self.assertIn('sharpe_ratio', best_result.metrics)

if __name__ == '__main__':
    unittest.main()