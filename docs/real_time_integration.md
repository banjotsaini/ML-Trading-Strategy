# Real-Time Market Data Integration

This document provides detailed information about the real-time market data integration feature in the ML Trading Strategy system.

## Overview

The real-time market data integration feature connects to live data feed APIs and updates trading signals continuously during market hours. This enables the system to react to market changes in real-time and generate timely trading signals.

## Components

The real-time market data integration consists of several key components:

1. **Real-Time Data Feed**: Connects to market data providers and retrieves real-time data
2. **Stream Processor**: Processes incoming market data and calculates technical indicators
3. **Signal Generator Service**: Continuously generates trading signals based on real-time data
4. **Market Scheduler**: Manages operations during market hours
5. **Notification System**: Sends alerts for trading signals and system status
6. **Resilience Utilities**: Handles errors and reconnection logic

## Supported Data Providers

The system currently supports the following data providers:

1. **Alpha Vantage**: REST API-based provider with comprehensive market data
2. **WebSocket Providers**: Generic support for WebSocket-based providers (e.g., Polygon.io, IEX Cloud)

## Configuration

To use the real-time market data integration, you need to configure it properly:

1. Create a configuration file (e.g., `config/live_trading_config.json`) with the following structure:

```json
{
    "provider": "alphavantage",
    "api_key": "YOUR_API_KEY_HERE",
    "tickers": ["AAPL", "GOOGL", "MSFT", "AMZN", "META"],
    "update_interval": 60,
    "signal_threshold": 0.7,
    "window_size": 100,
    "ensemble_method": "dynamic_ensemble",
    "notifications": {
        "enabled": true,
        "throttle_period": 300,
        "email": {
            "enabled": true,
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "username": "your-email@gmail.com",
            "password": "your-app-password",
            "from_email": "your-email@gmail.com"
        },
        "sms": {
            "enabled": false
        }
    },
    "notification_recipients": [
        "recipient-email@example.com"
    ],
    "market_hours": {
        "timezone": "America/New_York",
        "include_extended_hours": false
    }
}
```

2. Obtain an API key from your chosen data provider:
   - Alpha Vantage: [https://www.alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key)
   - Polygon.io: [https://polygon.io/dashboard/signup](https://polygon.io/dashboard/signup)

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Live Trading Service

To start the live trading service:

```bash
python scripts/live_trading.py --config config/live_trading_config.json
```

Command-line options:
- `--config`: Path to configuration file
- `--api-key`: API key for data provider (overrides config file)
- `--provider`: Data provider name (overrides config file)
- `--tickers`: Comma-separated list of tickers (overrides config file)
- `--log-level`: Logging level (default: INFO)

Example:
```bash
python scripts/live_trading.py --config config/live_trading_config.json --log-level DEBUG
```

### Using the API in Custom Scripts

You can also use the real-time market data integration in your custom scripts:

```python
from data.real_time_feed import RealTimeDataFeed
from data.stream_processor import StreamProcessor
from strategies.ml_trading_strategy import MLTradingStrategy
from trading.signal_service import SignalGeneratorService

# Initialize data feed
data_feed = RealTimeDataFeed(
    provider_name='alphavantage',
    api_key='YOUR_API_KEY_HERE'
)

# Connect to data feed
data_feed.connect()

# Subscribe to tickers
data_feed.subscribe(['AAPL', 'MSFT', 'GOOGL'])

# Initialize stream processor
stream_processor = StreamProcessor(window_size=100)

# Initialize trading strategy
strategy = MLTradingStrategy(models)

# Initialize signal generator service
signal_service = SignalGeneratorService(
    data_feed=data_feed,
    stream_processor=stream_processor,
    strategy=strategy,
    tickers=['AAPL', 'MSFT', 'GOOGL'],
    update_interval=60,
    signal_threshold=0.7
)

# Register callback for signals
def handle_signal(signal):
    print(f"Received signal: {signal['ticker']} - {signal['signal']}")
    print(f"Confidence: {signal['confidence']:.4f}")
    print(f"Price: ${signal['price']:.2f}")

signal_service.register_signal_callback(handle_signal)

# Start signal generator service
signal_service.start()

# ... later when done
signal_service.stop()
```

## Notifications

The system can send notifications for trading signals and system status:

1. **Signal Alerts**: Sent when a new trading signal is generated
2. **Error Alerts**: Sent when an error occurs in the system
3. **Status Updates**: Sent periodically to provide system status

To enable notifications, configure the `notifications` section in the configuration file.

## Market Hours

The system is aware of market hours and will only generate signals during market hours. You can configure the timezone and whether to include extended hours in the configuration file.

## Error Handling and Resilience

The system includes robust error handling and reconnection logic:

1. **Retry Mechanism**: Automatically retries failed operations with exponential backoff
2. **Circuit Breaker**: Prevents cascading failures by stopping operations when a service is failing repeatedly
3. **Connection Management**: Manages API connections with resilience

## Extending the System

### Adding a New Data Provider

To add a new data provider:

1. Create a new provider class that inherits from `DataFeedProvider`
2. Implement the required methods: `connect()`, `subscribe()`, `get_latest_data()`, and `disconnect()`
3. Add the provider to the `RealTimeDataFeed` class

Example:

```python
class MyCustomProvider(DataFeedProvider):
    def __init__(self, api_key):
        self.api_key = api_key
        self.is_connected = False
        
    def connect(self):
        # Implementation
        self.is_connected = True
        return True
        
    def subscribe(self, tickers):
        # Implementation
        return True
        
    def get_latest_data(self, ticker):
        # Implementation
        return data
        
    def disconnect(self):
        # Implementation
        self.is_connected = False
        return True
```

### Customizing Signal Generation

You can customize how signals are generated by modifying the `SignalGeneratorService` class or by creating a custom signal generator that uses the real-time data feed.

## Troubleshooting

### Common Issues

1. **API Key Issues**: Ensure your API key is valid and has not exceeded rate limits
2. **Connection Failures**: Check your internet connection and firewall settings
3. **Data Provider Outages**: Check the status of your data provider's service

### Logging

The system logs detailed information about its operation. Check the log files in the `logs` directory for troubleshooting.

## Performance Considerations

1. **API Rate Limits**: Be aware of your data provider's rate limits
2. **Resource Usage**: Monitor CPU and memory usage, especially with many tickers
3. **Network Bandwidth**: Real-time data can consume significant bandwidth

## Security Considerations

1. **API Keys**: Keep your API keys secure and never commit them to version control
2. **Email Credentials**: Use app-specific passwords for email notifications
3. **Data Privacy**: Be aware of data privacy regulations when storing market data