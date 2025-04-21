import requests
import websocket
import json
import threading
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Callable, Any

class DataFeedProvider(ABC):
    """Abstract base class for data feed providers"""
    
    @abstractmethod
    def connect(self):
        """Establish connection to the data feed"""
        pass
    
    @abstractmethod
    def subscribe(self, tickers: List[str]):
        """Subscribe to specific tickers"""
        pass
    
    @abstractmethod
    def get_latest_data(self, ticker: str) -> Dict[str, Any]:
        """Get latest data for a ticker"""
        pass
    
    @abstractmethod
    def disconnect(self):
        """Disconnect from the data feed"""
        pass

class AlphaVantageProvider(DataFeedProvider):
    """Alpha Vantage data feed provider"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.latest_data = {}
        self.is_connected = False
        
    def connect(self):
        """Establish connection (for REST API, just validate API key)"""
        try:
            # Test connection with a simple request
            params = {
                "function": "TIME_SERIES_INTRADAY",
                "symbol": "MSFT",
                "interval": "1min",
                "apikey": self.api_key,
                "outputsize": "compact"
            }
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            if "Error Message" in response.text:
                raise ConnectionError(f"API Error: {response.text}")
                
            self.is_connected = True
            return True
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Alpha Vantage: {str(e)}")
    
    def subscribe(self, tickers: List[str]):
        """Subscribe to tickers (for REST API, just store the list)"""
        self.subscribed_tickers = tickers
        return True
    
    def get_latest_data(self, ticker: str) -> Dict[str, Any]:
        """Get latest data for a ticker using REST API"""
        if not self.is_connected:
            raise ConnectionError("Not connected to data feed")
            
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": ticker,
            "apikey": self.api_key
        }
        
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if "Global Quote" not in data:
            raise ValueError(f"Invalid response for {ticker}: {data}")
            
        quote = data["Global Quote"]
        
        # Format the data
        latest_data = {
            "ticker": ticker,
            "price": float(quote["05. price"]),
            "volume": int(quote["06. volume"]),
            "timestamp": quote["07. latest trading day"],
            "change": float(quote["09. change"]),
            "change_percent": quote["10. change percent"].strip("%")
        }
        
        # Cache the data
        self.latest_data[ticker] = latest_data
        
        return latest_data
    
    def disconnect(self):
        """Disconnect from the data feed"""
        self.is_connected = False
        return True

class WebSocketProvider(DataFeedProvider):
    """WebSocket-based data feed provider (e.g., for IEX, Polygon.io)"""
    
    def __init__(self, api_key: str, ws_url: str):
        self.api_key = api_key
        self.ws_url = ws_url
        self.ws = None
        self.latest_data = {}
        self.is_connected = False
        self.callbacks = []
        
    def connect(self):
        """Establish WebSocket connection"""
        def on_message(ws, message):
            data = json.loads(message)
            self._process_message(data)
            
        def on_error(ws, error):
            print(f"WebSocket error: {error}")
            
        def on_close(ws, close_status_code, close_msg):
            print(f"WebSocket closed: {close_msg}")
            self.is_connected = False
            
        def on_open(ws):
            print("WebSocket connection established")
            self.is_connected = True
            
        # Create WebSocket connection
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        
        # Start WebSocket connection in a separate thread
        self.ws_thread = threading.Thread(target=self.ws.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()
        
        # Wait for connection to establish
        timeout = 10
        start_time = time.time()
        while not self.is_connected and time.time() - start_time < timeout:
            time.sleep(0.1)
            
        if not self.is_connected:
            raise ConnectionError("Failed to connect to WebSocket within timeout")
            
        return True
    
    def subscribe(self, tickers: List[str]):
        """Subscribe to tickers via WebSocket"""
        if not self.is_connected:
            raise ConnectionError("Not connected to data feed")
            
        # Format subscription message (provider-specific)
        subscription_msg = {
            "type": "subscribe",
            "symbols": tickers,
            "apiKey": self.api_key
        }
        
        self.ws.send(json.dumps(subscription_msg))
        self.subscribed_tickers = tickers
        return True
    
    def _process_message(self, data):
        """Process incoming WebSocket message"""
        # Implementation depends on the specific provider's message format
        # This is a generic example
        if "ticker" in data and "price" in data:
            ticker = data["ticker"]
            
            latest_data = {
                "ticker": ticker,
                "price": float(data["price"]),
                "volume": int(data.get("volume", 0)),
                "timestamp": data.get("timestamp", ""),
                "change": float(data.get("change", 0)),
                "change_percent": data.get("changePercent", "0")
            }
            
            # Cache the data
            self.latest_data[ticker] = latest_data
            
            # Call callbacks
            for callback in self.callbacks:
                callback(latest_data)
    
    def get_latest_data(self, ticker: str) -> Dict[str, Any]:
        """Get latest data for a ticker from cache"""
        if not self.is_connected:
            raise ConnectionError("Not connected to data feed")
            
        if ticker not in self.latest_data:
            raise ValueError(f"No data available for {ticker}")
            
        return self.latest_data[ticker]
    
    def register_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Register callback for new data"""
        self.callbacks.append(callback)
    
    def disconnect(self):
        """Disconnect from the data feed"""
        if self.ws:
            self.ws.close()
        self.is_connected = False
        return True

class RealTimeDataFeed:
    """Main interface for real-time market data"""
    
    def __init__(self, provider_name: str, api_key: str, **kwargs):
        """
        Initialize real-time data feed
        
        Parameters:
        -----------
        provider_name : str
            Name of the data provider ('alphavantage', 'iex', 'polygon', etc.)
        api_key : str
            API key for the data provider
        **kwargs : dict
            Additional provider-specific parameters
        """
        self.provider_name = provider_name.lower()
        self.api_key = api_key
        
        # Initialize provider based on name
        if self.provider_name == 'alphavantage':
            self.provider = AlphaVantageProvider(api_key)
        elif self.provider_name == 'websocket':
            ws_url = kwargs.get('ws_url')
            if not ws_url:
                raise ValueError("WebSocket URL required for WebSocket provider")
            self.provider = WebSocketProvider(api_key, ws_url)
        else:
            raise ValueError(f"Unsupported provider: {provider_name}")
    
    def connect(self):
        """Connect to the data feed"""
        return self.provider.connect()
    
    def subscribe(self, tickers: List[str]):
        """Subscribe to specific tickers"""
        return self.provider.subscribe(tickers)
    
    def get_latest_data(self, ticker: str) -> Dict[str, Any]:
        """Get latest data for a ticker"""
        return self.provider.get_latest_data(ticker)
    
    def register_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Register callback for new data (if supported by provider)"""
        if hasattr(self.provider, 'register_callback'):
            self.provider.register_callback(callback)
        else:
            raise NotImplementedError("Callbacks not supported by this provider")
    
    def disconnect(self):
        """Disconnect from the data feed"""
        return self.provider.disconnect()