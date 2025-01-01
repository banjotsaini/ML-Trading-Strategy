from typing import Dict, List, Any
import pandas as pd
import numpy as np

class PortfolioManager:
    def __init__(
        self, 
        initial_capital: float = 10000, 
        max_positions: int = 5
    ):
        """
        Initialize portfolio management system
        
        Parameters:
        -----------
        initial_capital : float, optional
            Starting portfolio value
        max_positions : int, optional
            Maximum number of simultaneous positions
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_positions = max_positions
        
        self.positions = {}
        self.transaction_history = []
        self.portfolio_value_history = [
            {
                'timestamp': pd.Timestamp.now(),
                'total_value': initial_capital,
                'positions': {}
            }
        ]
    
    def add_position(
        self, 
        ticker: str, 
        shares: int, 
        entry_price: float
    ) -> Dict[str, Any]:
        """
        Add a new position to the portfolio
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        shares : int
            Number of shares to purchase
        entry_price : float
            Price per share at entry
        
        Returns:
        --------
        Dict[str, Any]
            Position details
        """
        # Check portfolio position limits
        if len(self.positions) >= self.max_positions:
            raise ValueError("Maximum number of positions reached")
        
        # Calculate total position cost
        total_cost = shares * entry_price
        
        # Check sufficient capital
        if total_cost > self.current_capital:
            raise ValueError("Insufficient capital for position")
        
        # Create position record
        position = {
            'ticker': ticker,
            'shares': shares,
            'entry_price': entry_price,
            'entry_timestamp': pd.Timestamp.now(),
            'total_cost': total_cost
        }
        
        # Update portfolio
        self.positions[ticker] = position
        self.current_capital -= total_cost
        
        # Record transaction
        self.transaction_history.append({
            'type': 'BUY',
            'ticker': ticker,
            'shares': shares,
            'price': entry_price,
            'timestamp': pd.Timestamp.now()
        })
        
        return position
    
    def remove_position(
        self, 
        ticker: str, 
        current_price: float
    ) -> Dict[str, Any]:
        """
        Close an existing position
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        current_price : float
            Current market price
        
        Returns:
        --------
        Dict[str, Any]
            Position exit details
        """
        if ticker not in self.positions:
            raise ValueError(f"No position found for {ticker}")
        
        position = self.positions[ticker]
        total_value = position['shares'] * current_price
        
        # Calculate profit/loss
        profit_loss = total_value - position['total_cost']
        
        # Update portfolio
        self.current_capital += total_value
        del self.positions[ticker]
        
        # Record transaction
        self.transaction_history.append({
            'type': 'SELL',
            'ticker': ticker,
            'shares': position['shares'],
            'price': current_price,
            'timestamp': pd.Timestamp.now(),
            'profit_loss': profit_loss
        })
        
        return {
            'ticker': ticker,
            'profit_loss': profit_loss,
            'total_value': total_value
        }
    
    def update_portfolio_value(
        self, 
        current_prices: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Update portfolio value based on current market prices
        
        Parameters:
        -----------
        current_prices : Dict[str, float]
            Current market prices for portfolio tickers
        
        Returns:
        --------
        Dict[str, Any]
            Portfolio valuation details
        """
        position_values = {}
        total_portfolio_value = self.current_capital
        
        for ticker, position in self.positions.items():
            if ticker not in current_prices:
                raise ValueError(f"No current price for {ticker}")
            
            current_price = current_prices[ticker]
            position_value = position['shares'] * current_price
            position_values[ticker] = {
                'current_price': current_price,
                'total_value': position_value,
                'profit_loss': position_value - position['total_cost']
            }
            
            total_portfolio_value += position_value
        
        # Record portfolio value history
        portfolio_snapshot = {
            'timestamp': pd.Timestamp.now(),
            'total_value': total_portfolio_value,
            'positions': position_values,
            'cash': self.current_capital
        }
        
        self.portfolio_value_history.append(portfolio_snapshot)
        
        return portfolio_snapshot
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive portfolio performance metrics
        
        Returns:
        --------
        Dict[str, Any]
            Portfolio performance statistics
        """
        if len(self.portfolio_value_history) < 2:
            return {}
        
        initial_value = self.initial_capital
        final_value = self.portfolio_value_history[-1]['total_value']
        
        # Calculate total return
        total_return = (final_value / initial_value - 1) * 100
        
        # Calculate profit/loss from trades
        total_profit = sum(
            trade.get('profit_loss', 0) 
            for trade in self.transaction_history 
            if trade['type'] == 'SELL'
        )
        
        return {
            'initial_capital': initial_value,
            'final_value': final_value,
            'total_return_percentage': total_return,
            'total_profit': total_profit,
            'total_trades': len(self.transaction_history),
            'current_positions': len(self.positions)
        }