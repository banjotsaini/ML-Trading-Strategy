from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np
import pandas as pd

class BaseStrategy(ABC):
    def __init__(
        self, 
        initial_capital: float = 10000, 
        risk_percentage: float = 0.02
    ):
        """
        Initialize base trading strategy
        
        Parameters:
        -----------
        initial_capital : float, optional
            Initial trading capital
        risk_percentage : float, optional
            Maximum risk percentage per trade
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_percentage = risk_percentage
        
        # Tracking performance metrics
        self.trades = []
        self.portfolio_history = []
    
    @abstractmethod
    def generate_signals(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """
        Generate trading signals based on market data
        
        Parameters:
        -----------
        market_data : pandas.DataFrame
            Market data for signal generation
        
        Returns:
        --------
        Dict[str, float]
            Trading signals with confidence scores
        """
        pass
    
    def calculate_position_size(
        self, 
        current_price: float, 
        stop_loss_amount: float
    ) -> int:
        """
        Calculate position size based on risk management
        
        Parameters:
        -----------
        current_price : float
            Current stock price
        stop_loss_amount : float
            Amount of potential loss per share
        
        Returns:
        --------
        int
            Number of shares to trade
        """
        risk_amount = self.current_capital * self.risk_percentage
        shares = int(risk_amount / stop_loss_amount)
        return shares
    
    def record_trade(
        self, 
        ticker: str, 
        action: str, 
        price: float, 
        shares: int
    ):
        """
        Record trade details for performance tracking
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        action : str
            Trade action (buy/sell)
        price : float
            Trade price
        shares : int
            Number of shares traded
        """
        trade_details = {
            'ticker': ticker,
            'action': action,
            'price': price,
            'shares': shares,
            'timestamp': pd.Timestamp.now()
        }
        self.trades.append(trade_details)
    
    def update_portfolio_value(self):
        """
        Update portfolio value and history
        """
        current_value = self.calculate_portfolio_value()
        self.portfolio_history.append({
            'timestamp': pd.Timestamp.now(),
            'portfolio_value': current_value
        })
    
    def calculate_portfolio_value(self) -> float:
        """
        Calculate total portfolio value
        
        Returns:
        --------
        float
            Total portfolio value
        """
        # This method should be overridden in child classes
        # to include current market prices and positions
        return self.current_capital
    
    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate trading performance metrics
        
        Returns:
        --------
        Dict[str, Any]
            Performance metrics
        """
        if not self.portfolio_history:
            return {}
        
        total_return = (
            self.portfolio_history[-1]['portfolio_value'] / 
            self.initial_capital - 1
        ) * 100
        
        return {
            'total_return_percentage': total_return,
            'total_trades': len(self.trades),
            'initial_capital': self.initial_capital,
            'final_capital': self.current_capital
        }