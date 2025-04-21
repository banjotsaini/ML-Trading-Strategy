import os
import json
import yaml
import logging
from typing import Dict, List, Any, Optional, Union

class BacktestConfig:
    """Container for backtest configuration"""
    
    def __init__(
        self,
        strategy_name: str,
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        initial_capital: float = 10000.0,
        commission: float = 0.0,
        slippage: float = 0.0,
        parameters: Optional[Dict[str, Any]] = None,
        data_source: Optional[Dict[str, Any]] = None,
        optimization: Optional[Dict[str, Any]] = None,
        output: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize backtest configuration
        
        Parameters:
        -----------
        strategy_name : str
            Name of the strategy
        tickers : List[str]
            List of tickers to backtest
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
        parameters : Dict[str, Any], optional
            Strategy parameters
        data_source : Dict[str, Any], optional
            Data source configuration
        optimization : Dict[str, Any], optional
            Optimization configuration
        output : Dict[str, Any], optional
            Output configuration
        """
        self.strategy_name = strategy_name
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.parameters = parameters or {}
        self.data_source = data_source or {}
        self.optimization = optimization or {}
        self.output = output or {}
        
        # Set default values
        self._set_defaults()
        
        # Validate configuration
        self._validate()
    
    def _set_defaults(self):
        """Set default values for configuration"""
        # Data source defaults
        if 'type' not in self.data_source:
            self.data_source['type'] = 'csv'
        
        # Optimization defaults
        if 'enabled' not in self.optimization:
            self.optimization['enabled'] = False
        
        # Output defaults
        if 'save_results' not in self.output:
            self.output['save_results'] = True
        if 'output_dir' not in self.output:
            self.output['output_dir'] = 'backtest_results'
        if 'plot_results' not in self.output:
            self.output['plot_results'] = True
    
    def _validate(self):
        """Validate configuration"""
        # Check required fields
        if not self.strategy_name:
            raise ValueError("Strategy name is required")
        
        if not self.tickers:
            raise ValueError("Tickers list is required")
        
        # Check numeric values
        if self.initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
        
        if self.commission < 0:
            raise ValueError("Commission cannot be negative")
        
        if self.slippage < 0:
            raise ValueError("Slippage cannot be negative")
        
        # Check dates
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("Start date cannot be after end date")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary
        
        Returns:
        --------
        Dict[str, Any]
            Configuration dictionary
        """
        return {
            'strategy_name': self.strategy_name,
            'tickers': self.tickers,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'initial_capital': self.initial_capital,
            'commission': self.commission,
            'slippage': self.slippage,
            'parameters': self.parameters,
            'data_source': self.data_source,
            'optimization': self.optimization,
            'output': self.output
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BacktestConfig':
        """
        Create configuration from dictionary
        
        Parameters:
        -----------
        config_dict : Dict[str, Any]
            Configuration dictionary
            
        Returns:
        --------
        BacktestConfig
            Backtest configuration
        """
        return cls(
            strategy_name=config_dict.get('strategy_name', ''),
            tickers=config_dict.get('tickers', []),
            start_date=config_dict.get('start_date'),
            end_date=config_dict.get('end_date'),
            initial_capital=config_dict.get('initial_capital', 10000.0),
            commission=config_dict.get('commission', 0.0),
            slippage=config_dict.get('slippage', 0.0),
            parameters=config_dict.get('parameters', {}),
            data_source=config_dict.get('data_source', {}),
            optimization=config_dict.get('optimization', {}),
            output=config_dict.get('output', {})
        )

def load_config(config_path: str) -> BacktestConfig:
    """
    Load configuration from file
    
    Parameters:
    -----------
    config_path : str
        Path to configuration file
        
    Returns:
    --------
    BacktestConfig
        Backtest configuration
    """
    logger = logging.getLogger(__name__)
    
    # Check if file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Determine file format
    file_ext = os.path.splitext(config_path)[1].lower()
    
    try:
        # Load configuration
        if file_ext == '.json':
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        elif file_ext in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {file_ext}")
        
        # Create configuration
        config = BacktestConfig.from_dict(config_dict)
        
        logger.info(f"Loaded configuration from {config_path}")
        
        return config
    
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {str(e)}")
        raise

def save_config(config: BacktestConfig, config_path: str):
    """
    Save configuration to file
    
    Parameters:
    -----------
    config : BacktestConfig
        Backtest configuration
    config_path : str
        Path to configuration file
    """
    logger = logging.getLogger(__name__)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
    
    # Determine file format
    file_ext = os.path.splitext(config_path)[1].lower()
    
    try:
        # Convert configuration to dictionary
        config_dict = config.to_dict()
        
        # Save configuration
        if file_ext == '.json':
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=4)
        elif file_ext in ['.yaml', '.yml']:
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported configuration file format: {file_ext}")
        
        logger.info(f"Saved configuration to {config_path}")
    
    except Exception as e:
        logger.error(f"Error saving configuration to {config_path}: {str(e)}")
        raise

def validate_parameters(parameters: Dict[str, Any], parameter_schema: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate parameters against schema
    
    Parameters:
    -----------
    parameters : Dict[str, Any]
        Parameters to validate
    parameter_schema : Dict[str, Dict[str, Any]]
        Parameter schema
        
    Returns:
    --------
    Dict[str, Any]
        Validated parameters
    """
    validated_params = {}
    
    for param_name, param_schema in parameter_schema.items():
        # Check if parameter is required
        if param_schema.get('required', False) and param_name not in parameters:
            raise ValueError(f"Required parameter missing: {param_name}")
        
        # Get parameter value
        if param_name in parameters:
            param_value = parameters[param_name]
            
            # Validate parameter type
            param_type = param_schema.get('type')
            if param_type:
                if param_type == 'int' and not isinstance(param_value, int):
                    raise ValueError(f"Parameter {param_name} must be an integer")
                elif param_type == 'float' and not isinstance(param_value, (int, float)):
                    raise ValueError(f"Parameter {param_name} must be a number")
                elif param_type == 'bool' and not isinstance(param_value, bool):
                    raise ValueError(f"Parameter {param_name} must be a boolean")
                elif param_type == 'str' and not isinstance(param_value, str):
                    raise ValueError(f"Parameter {param_name} must be a string")
                elif param_type == 'list' and not isinstance(param_value, list):
                    raise ValueError(f"Parameter {param_name} must be a list")
                elif param_type == 'dict' and not isinstance(param_value, dict):
                    raise ValueError(f"Parameter {param_name} must be a dictionary")
            
            # Validate parameter range
            if 'min' in param_schema and param_value < param_schema['min']:
                raise ValueError(f"Parameter {param_name} must be at least {param_schema['min']}")
            
            if 'max' in param_schema and param_value > param_schema['max']:
                raise ValueError(f"Parameter {param_name} must be at most {param_schema['max']}")
            
            # Validate parameter choices
            if 'choices' in param_schema and param_value not in param_schema['choices']:
                raise ValueError(f"Parameter {param_name} must be one of: {param_schema['choices']}")
            
            # Add parameter to validated parameters
            validated_params[param_name] = param_value
        elif 'default' in param_schema:
            # Use default value
            validated_params[param_name] = param_schema['default']
    
    return validated_params