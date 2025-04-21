import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from collections import deque

class StreamProcessor:
    """Process incoming market data streams"""
    
    def __init__(self, window_size: int = 100):
        """
        Initialize stream processor
        
        Parameters:
        -----------
        window_size : int
            Size of the sliding window for technical indicators
        """
        self.window_size = window_size
        self.data_windows = {}  # Ticker -> deque of price data
        self.latest_features = {}  # Ticker -> latest feature vector
        
    def process_tick_data(self, tick_data: Dict[str, Any]):
        """
        Process individual price tick
        
        Parameters:
        -----------
        tick_data : Dict[str, Any]
            Tick data from the data feed
        """
        ticker = tick_data['ticker']
        price = tick_data['price']
        
        # Initialize data window if needed
        if ticker not in self.data_windows:
            self.data_windows[ticker] = deque(maxlen=self.window_size)
        
        # Add new price to the window
        self.data_windows[ticker].append(price)
        
        # Update features
        self.update_features(ticker)
        
    def update_features(self, ticker: str):
        """
        Update technical indicators and features for a ticker
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol
        """
        if ticker not in self.data_windows or len(self.data_windows[ticker]) < 2:
            return
        
        # Convert deque to numpy array for calculations
        prices = np.array(self.data_windows[ticker])
        
        # Calculate technical indicators
        features = {}
        
        # Simple Moving Averages
        if len(prices) >= 5:
            features['sma_5'] = np.mean(prices[-5:])
        if len(prices) >= 10:
            features['sma_10'] = np.mean(prices[-10:])
        if len(prices) >= 20:
            features['sma_20'] = np.mean(prices[-20:])
        
        # Price changes
        features['price_change_1'] = prices[-1] - prices[-2]
        if len(prices) >= 5:
            features['price_change_5'] = prices[-1] - prices[-5]
        
        # Relative Strength Index (RSI)
        if len(prices) >= 14:
            delta = np.diff(prices[-15:])
            gain = delta.copy()
            loss = delta.copy()
            gain[gain < 0] = 0
            loss[loss > 0] = 0
            avg_gain = np.mean(gain)
            avg_loss = np.abs(np.mean(loss))
            if avg_loss != 0:
                rs = avg_gain / avg_loss
                features['rsi_14'] = 100 - (100 / (1 + rs))
            else:
                features['rsi_14'] = 100
        
        # Momentum
        if len(prices) >= 10:
            features['momentum_10'] = prices[-1] - prices[-10]
        
        # Volatility
        if len(prices) >= 20:
            features['volatility_20'] = np.std(prices[-20:])
        
        # Store latest features
        self.latest_features[ticker] = features
        
    def get_latest_features(self, ticker: str) -> Optional[Dict[str, float]]:
        """
        Get latest feature set for ML models
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol
            
        Returns:
        --------
        Dict[str, float] or None
            Latest features for the ticker, or None if not available
        """
        return self.latest_features.get(ticker)
    
    def get_feature_vector(self, ticker: str) -> Optional[np.ndarray]:
        """
        Get feature vector for ML model input
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol
            
        Returns:
        --------
        numpy.ndarray or None
            Feature vector for ML model, or None if not enough data
        """
        features = self.get_latest_features(ticker)
        if not features:
            return None
        
        # Create feature vector in the expected order for ML models
        # This order must match the order used during model training
        feature_names = [
            'sma_5', 'sma_10', 'sma_20',
            'price_change_1', 'price_change_5',
            'rsi_14', 'momentum_10', 'volatility_20'
        ]
        
        # Check if we have all required features
        if not all(name in features for name in feature_names):
            return None
        
        # Create feature vector
        feature_vector = np.array([features[name] for name in feature_names])
        return feature_vector.reshape(1, -1)  # Reshape for ML model input