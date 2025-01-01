from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd

class BrokerInterface(ABC):
    def __init__(self, api_key: str = None, secret_key: str = None):
        """
        Initialize broker interface
        
        Parameters:
        -----------
        api_key : str, optional
            API key for authentication
        secret_key : str, optional
            Secret key for authentication
        """
        self.api_key = api_key
        self.secret_key = secret_key
    
    @abstractmethod
    def connect(self):
        """
        Establish connection with broker API
        """
        pass
    
    @abstractmethod
    def get_account_info(self) -> Dict[str, Any]:
        """
        Retrieve account information
        
        Returns:
        --------
        Dict[str, Any]
            Account details including balance, positions, etc.
        """
        pass
    
    @abstractmethod
    def place_market_order(
        self, 
        ticker: str, 
        quantity: int, 
        side: str
    ) -> Dict[str, Any]:
        """
        Place a market order
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        quantity : int
            Number of shares to trade
        side : str
            Order side (buy/sell)
        
        Returns:
        --------
        Dict[str, Any]
            Order execution details
        """
        pass
    
    @abstractmethod
    def place_limit_order(
        self, 
        ticker: str, 
        quantity: int, 
        side: str, 
        limit_price: float
    ) -> Dict[str, Any]:
        """
        Place a limit order
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        quantity : int
            Number of shares to trade
        side : str
            Order side (buy/sell)
        limit_price : float
            Price limit for order
        
        Returns:
        --------
        Dict[str, Any]
            Order execution details
        """
        pass
    
    @abstractmethod
    def get_current_price(self, ticker: str) -> float:
        """
        Get current market price for a ticker
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        
        Returns:
        --------
        float
            Current market price
        """
        pass
    
    @abstractmethod
    def get_historical_data(
        self, 
        ticker: str, 
        timeframe: str = '1d'
    ) -> pd.DataFrame:
        """
        Fetch historical market data
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        timeframe : str, optional
            Data timeframe (e.g., '1d', '1h', '1m')
        
        Returns:
        --------
        pandas.DataFrame
            Historical market data
        """
        pass