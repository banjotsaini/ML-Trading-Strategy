#!/usr/bin/env python
"""
Real-Time Market Data Integration Example

This script demonstrates how to use the real-time market data integration feature.
"""

import os
import sys
import time
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.real_time_feed import RealTimeDataFeed
from data.stream_processor import StreamProcessor
from utils.market_scheduler import MarketScheduler

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def print_market_data(data):
    """Print market data in a formatted way"""
    print("\n" + "="*50)
    print(f"Ticker: {data['ticker']}")
    print(f"Price: ${data['price']:.2f}")
    print(f"Volume: {data['volume']}")
    print(f"Change: {data['change']}")
    print(f"Change %: {data['change_percent']}%")
    print(f"Timestamp: {data['timestamp']}")
    print("="*50)

def main():
    """Main function"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Check if API key is provided
    api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        print("Error: API key not found. Please set the ALPHA_VANTAGE_API_KEY environment variable.")
        print("Example: export ALPHA_VANTAGE_API_KEY=your_api_key_here")
        return
    
    # Initialize market scheduler
    market_scheduler = MarketScheduler()
    
    # Check if market is open
    if not market_scheduler.is_market_open(include_extended=True):
        next_open = market_scheduler.get_next_market_open()
        print(f"Market is currently closed. Next market open: {next_open}")
        print("This example will still run, but you may not get real-time data.")
    
    # Initialize data feed
    print("Initializing Alpha Vantage data feed...")
    data_feed = RealTimeDataFeed(
        provider_name='alphavantage',
        api_key=api_key
    )
    
    # Initialize stream processor
    print("Initializing stream processor...")
    stream_processor = StreamProcessor(window_size=20)
    
    # Connect to data feed
    print("Connecting to data feed...")
    data_feed.connect()
    
    # Define tickers to monitor
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    # Subscribe to tickers
    print(f"Subscribing to tickers: {', '.join(tickers)}")
    data_feed.subscribe(tickers)
    
    # Process data for each ticker
    print("\nFetching and processing real-time data...")
    print("Press Ctrl+C to exit")
    
    try:
        # Main loop
        while True:
            for ticker in tickers:
                try:
                    # Get latest data
                    latest_data = data_feed.get_latest_data(ticker)
                    
                    # Print data
                    print_market_data(latest_data)
                    
                    # Process data
                    stream_processor.process_tick_data(latest_data)
                    
                    # Get features
                    features = stream_processor.get_latest_features(ticker)
                    if features:
                        print("\nCalculated Features:")
                        for name, value in features.items():
                            print(f"  {name}: {value:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error processing data for {ticker}: {str(e)}")
            
            # Wait before next update
            print("\nWaiting for next update...")
            time.sleep(60)  # Alpha Vantage has rate limits
            
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Exiting...")
    finally:
        # Disconnect from data feed
        print("Disconnecting from data feed...")
        data_feed.disconnect()
        print("Done!")

if __name__ == "__main__":
    main()