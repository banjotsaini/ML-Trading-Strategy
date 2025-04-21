import unittest
import os
import sys
import time
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.real_time_feed import RealTimeDataFeed, AlphaVantageProvider, WebSocketProvider
from data.stream_processor import StreamProcessor

class MockResponse:
    def __init__(self, json_data, status_code=200, text=""):
        self.json_data = json_data
        self.status_code = status_code
        self.text = text
    
    def json(self):
        return self.json_data
    
    def raise_for_status(self):
        if self.status_code != 200:
            raise Exception(f"HTTP Error: {self.status_code}")

class TestRealTimeFeed(unittest.TestCase):
    
    def setUp(self):
        self.api_key = "test_api_key"
        
        # Sample data for testing
        self.sample_quote = {
            "Global Quote": {
                "01. symbol": "AAPL",
                "02. open": "150.0",
                "03. high": "152.0",
                "04. low": "149.0",
                "05. price": "151.0",
                "06. volume": "10000000",
                "07. latest trading day": "2023-01-01",
                "08. previous close": "149.5",
                "09. change": "1.5",
                "10. change percent": "1.0%"
            }
        }
        
        self.sample_websocket_data = {
            "ticker": "AAPL",
            "price": 151.0,
            "volume": 10000000,
            "timestamp": "2023-01-01T12:00:00Z",
            "change": 1.5,
            "changePercent": "1.0"
        }
    
    @patch('requests.get')
    def test_alpha_vantage_provider(self, mock_get):
        # Mock the API response
        mock_get.return_value = MockResponse(self.sample_quote)
        
        # Create provider
        provider = AlphaVantageProvider(self.api_key)
        
        # Test connect
        result = provider.connect()
        self.assertTrue(result)
        self.assertTrue(provider.is_connected)
        
        # Test subscribe
        tickers = ["AAPL", "MSFT", "GOOGL"]
        result = provider.subscribe(tickers)
        self.assertTrue(result)
        self.assertEqual(provider.subscribed_tickers, tickers)
        
        # Test get_latest_data
        data = provider.get_latest_data("AAPL")
        self.assertEqual(data["ticker"], "AAPL")
        self.assertEqual(data["price"], 151.0)
        self.assertEqual(data["volume"], 10000000)
        
        # Test disconnect
        result = provider.disconnect()
        self.assertTrue(result)
        self.assertFalse(provider.is_connected)
    
    @patch('websocket.WebSocketApp')
    def test_websocket_provider(self, mock_websocket):
        # Mock WebSocket
        mock_ws = MagicMock()
        mock_websocket.return_value = mock_ws
        
        # Create provider
        provider = WebSocketProvider(self.api_key, "wss://test.websocket.url")
        
        # Mock the on_open callback to set is_connected
        def mock_on_open(ws):
            provider.is_connected = True
        
        # Replace the run_forever method to call on_open immediately
        def mock_run_forever():
            mock_on_open(mock_ws)
        
        mock_ws.run_forever = mock_run_forever
        
        # Test connect
        result = provider.connect()
        self.assertTrue(result)
        self.assertTrue(provider.is_connected)
        
        # Test subscribe
        tickers = ["AAPL", "MSFT", "GOOGL"]
        result = provider.subscribe(tickers)
        self.assertTrue(result)
        self.assertEqual(provider.subscribed_tickers, tickers)
        
        # Test _process_message
        provider._process_message(self.sample_websocket_data)
        self.assertIn("AAPL", provider.latest_data)
        self.assertEqual(provider.latest_data["AAPL"]["price"], 151.0)
        
        # Test get_latest_data
        data = provider.get_latest_data("AAPL")
        self.assertEqual(data["ticker"], "AAPL")
        self.assertEqual(data["price"], 151.0)
        
        # Test callback
        callback = MagicMock()
        provider.register_callback(callback)
        provider._process_message(self.sample_websocket_data)
        callback.assert_called_once()
        
        # Test disconnect
        result = provider.disconnect()
        self.assertTrue(result)
        self.assertFalse(provider.is_connected)
    
    @patch('data.real_time_feed.AlphaVantageProvider')
    def test_real_time_data_feed(self, mock_provider_class):
        # Mock provider
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider
        
        # Create data feed
        data_feed = RealTimeDataFeed("alphavantage", self.api_key)
        
        # Test connect
        data_feed.connect()
        mock_provider.connect.assert_called_once()
        
        # Test subscribe
        tickers = ["AAPL", "MSFT", "GOOGL"]
        data_feed.subscribe(tickers)
        mock_provider.subscribe.assert_called_once_with(tickers)
        
        # Test get_latest_data
        data_feed.get_latest_data("AAPL")
        mock_provider.get_latest_data.assert_called_once_with("AAPL")
        
        # Test disconnect
        data_feed.disconnect()
        mock_provider.disconnect.assert_called_once()

class TestStreamProcessor(unittest.TestCase):
    
    def setUp(self):
        self.stream_processor = StreamProcessor(window_size=20)
        
        # Sample tick data
        self.tick_data = {
            "ticker": "AAPL",
            "price": 150.0,
            "volume": 10000000,
            "timestamp": "2023-01-01T12:00:00Z",
            "change": 1.0,
            "change_percent": "0.67"
        }
    
    def test_process_tick_data(self):
        # Process single tick
        self.stream_processor.process_tick_data(self.tick_data)
        
        # Check data window
        self.assertIn("AAPL", self.stream_processor.data_windows)
        self.assertEqual(len(self.stream_processor.data_windows["AAPL"]), 1)
        self.assertEqual(self.stream_processor.data_windows["AAPL"][0], 150.0)
        
        # Process more ticks
        for i in range(1, 10):
            tick = self.tick_data.copy()
            tick["price"] = 150.0 + i
            self.stream_processor.process_tick_data(tick)
        
        # Check data window again
        self.assertEqual(len(self.stream_processor.data_windows["AAPL"]), 10)
        self.assertEqual(self.stream_processor.data_windows["AAPL"][-1], 159.0)
    
    def test_update_features(self):
        # Add enough data for feature calculation
        for i in range(30):
            tick = self.tick_data.copy()
            tick["price"] = 150.0 + i * 0.5
            self.stream_processor.process_tick_data(tick)
        
        # Check features
        features = self.stream_processor.get_latest_features("AAPL")
        self.assertIsNotNone(features)
        
        # Check specific features
        self.assertIn("sma_5", features)
        self.assertIn("sma_10", features)
        self.assertIn("sma_20", features)
        self.assertIn("price_change_1", features)
        self.assertIn("rsi_14", features)
        
        # Check feature values
        self.assertAlmostEqual(features["price_change_1"], 0.5, places=1)
        
        # Check feature vector
        feature_vector = self.stream_processor.get_feature_vector("AAPL")
        self.assertIsNotNone(feature_vector)
        self.assertEqual(feature_vector.shape, (1, 8))  # 8 features

if __name__ == '__main__':
    unittest.main()