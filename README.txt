### Project Setup Instructions

1. Create a new project directory
2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install requirements:
```bash
pip install -r requirements.txt
```

4. Run the main script:
```bash
python main.py
```

### Key Features
- Modular design
- Machine learning-based trading strategy
- Sophisticated ensemble methods
- Real-time market data integration
- Comprehensive backtesting framework
- Risk management
- Logging and error handling
- Configurable settings
- Notification system

### Ensemble Methods
The project now includes advanced ensemble methods for combining predictions from multiple machine learning models:

1. **Simple Average**: Basic averaging of model predictions
2. **Weighted Average**: Weighted averaging based on model weights
3. **Majority Vote**: Decision based on majority voting
4. **Weighted Vote**: Decision based on weighted voting
5. **Confidence Weighted**: Weighting based on model confidence
6. **Dynamic Ensemble**: Adaptive weighting based on model performance

The system automatically evaluates and selects the best ensemble method based on historical performance.

### Real-Time Market Data Integration
The system now includes real-time market data integration that connects to live data feed APIs and updates trading signals continuously during market hours:

1. **Multiple Data Providers**: Support for Alpha Vantage and WebSocket-based providers
2. **Streaming Data Processing**: Efficient processing of real-time market data
3. **Continuous Signal Generation**: Trading signals updated in real-time
4. **Market Hours Awareness**: Automatically operates during market hours
5. **Notification System**: Email and SMS alerts for trading signals and system status
6. **Resilient Operation**: Error handling and automatic reconnection

To use the real-time trading feature:

1. Configure your API keys in `config/live_trading_config.json`
2. Run the live trading script:
```bash
python scripts/live_trading.py --config config/live_trading_config.json
```

### Backtesting Framework
The system now includes a comprehensive backtesting framework for historical performance evaluation of different trading strategies and parameter optimization:

1. **Strategy Simulation**: Simulate trading strategies on historical data
2. **Performance Metrics**: Calculate comprehensive performance metrics (Sharpe ratio, drawdown, etc.)
3. **Parameter Optimization**: Grid search and random search for optimal parameters
4. **Walk-Forward Testing**: Time-series cross-validation with walk-forward optimization
5. **Visualization**: Visualize equity curves, drawdowns, returns distribution, and more
6. **Configuration Management**: Manage backtest configurations with JSON/YAML files

To use the backtesting framework:

1. Configure your backtest settings in `config/backtest_config.json`
2. Run the backtest script:
```bash
python scripts/backtest.py --config config/backtest_config.json
```

Or use the convenience scripts:
```bash
./run_backtest.sh  # Unix/Linux/Mac
run_backtest.bat   # Windows
```


ml-trading-strategy/
│
├── backtesting/
│   ├── __init__.py
│   ├── engine.py             # Core backtesting engine
│   ├── optimization.py       # Parameter optimization utilities
│   ├── visualization.py      # Visualization utilities
│   └── config.py             # Configuration utilities
│
├── config/
│   ├── __init__.py
│   ├── settings.py           # Global configuration settings
│   ├── credentials.py        # Secure storage of API keys (gitignored)
│   ├── live_trading_config.json # Configuration for live trading
│   └── backtest_config.json  # Configuration for backtesting
│
├── data/
│   ├── __init__.py
│   ├── data_loader.py        # Functions for downloading and storing market data
│   ├── feature_store.py      # Feature engineering and management
│   ├── real_time_feed.py     # Real-time market data feed connector
│   └── stream_processor.py   # Streaming data processor
│
├── docs/
│   ├── real_time_integration.md  # Documentation for real-time integration
│   └── backtesting_framework.md  # Documentation for backtesting framework
│
├── examples/
│   ├── __init__.py
│   ├── real_time_example.py  # Example of real-time data integration
│   └── backtest_example.py   # Example of backtesting framework
│
├── models/
│   ├── __init__.py
│   ├── base_model.py         # Abstract base class for ML models
│   ├── random_forest.py      # Random Forest model implementation
│   ├── gradient_boosting.py  # Gradient Boosting model implementation
│   ├── svm.py                # Support Vector Machine model implementation
│   └── ensemble_methods.py   # Advanced ensemble methods implementation
│
├── strategies/
│   ├── __init__.py
│   ├── base_strategy.py      # Base trading strategy class
│   ├── ml_trading_strategy.py # Machine learning-based trading strategy
│   └── risk_management.py    # Risk management and position sizing
│
├── trading/
│   ├── __init__.py
│   ├── broker_interface.py   # Abstract broker API interface
│   ├── order_execution.py    # Order placement and tracking
│   ├── portfolio_manager.py  # Portfolio allocation and management
│   └── signal_service.py     # Continuous signal generation service
│
├── utils/
│   ├── __init__.py
│   ├── logger.py             # Logging utility
│   ├── performance_metrics.py # Calculate trading performance metrics
│   ├── notifications.py      # Send alerts and notifications
│   ├── market_scheduler.py   # Market hours scheduling utilities
│   └── resilience.py         # Error handling and reconnection logic
│
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_models.py
│   ├── test_strategy.py
│   ├── test_trading.py
│   ├── test_real_time_feed.py # Tests for real-time data feed
│   └── test_backtesting.py   # Tests for backtesting framework
│
├── scripts/
│   ├── __init__.py
│   ├── train_models.py       # Script to train ML models
│   ├── backtest.py           # Backtesting script
│   └── live_trading.py       # Live trading execution script
│
├── requirements.txt
├── README.md
├── main.py                   # Main entry point for the application
├── run_backtest.sh           # Shell script to run backtest
├── run_backtest.bat          # Batch script to run backtest
├── run_live_trading.sh       # Shell script to run live trading
└── run_live_trading.bat      # Batch script to run live trading
