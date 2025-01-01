import numpy as np
import pandas as pd
from typing import Dict, Any, List

class PerformanceMetrics:
    @staticmethod
    def calculate_returns(portfolio_values: List[float]) -> np.ndarray:
        """
        Calculate daily returns from portfolio values
        
        Parameters:
        -----------
        portfolio_values : List[float]
            Sequential portfolio values
        
        Returns:
        --------
        numpy.ndarray
            Daily percentage returns
        """
        return np.array(portfolio_values[1:]) / np.array(portfolio_values[:-1]) - 1
    
    @staticmethod
    def sharpe_ratio(
        returns: np.ndarray, 
        risk_free_rate: float = 0.02
    ) -> float:
        """
        Calculate Sharpe ratio
        
        Parameters:
        -----------
        returns : numpy.ndarray
            Portfolio returns
        risk_free_rate : float, optional
            Annual risk-free rate
        
        Returns:
        --------
        float
            Sharpe ratio
        """
        if len(returns) == 0:
            return 0.0
        
        annualization_factor = np.sqrt(252)  # Trading days in a year
        
        excess_returns = returns - risk_free_rate / 252
        
        return (
            np.mean(excess_returns) / np.std(excess_returns) * 
            annualization_factor
        )
    
    @staticmethod
    def maximum_drawdown(portfolio_values: List[float]) -> float:
        """
        Calculate maximum portfolio drawdown
        
        Parameters:
        -----------
        portfolio_values : List[float]
            Sequential portfolio values
        
        Returns:
        --------
        float
            Maximum drawdown percentage
        """
        cumulative_max = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - cumulative_max) / cumulative_max
        
        return np.min(drawdown) * 100
    
    @staticmethod
    def win_loss_ratio(trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate win/loss trade statistics
        
        Parameters:
        -----------
        trades : List[Dict[str, Any]]
            List of trading transactions
        
        Returns:
        --------
        Dict[str, float]
            Win/loss trade statistics
        """
        winning_trades = [
            trade for trade in trades 
            if trade.get('profit_loss', 0) > 0
        ]
        losing_trades = [
            trade for trade in trades 
            if trade.get('profit_loss', 0) <= 0
        ]
        
        return {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(trades) if trades else 0,
            'avg_win_amount': np.mean([
                trade['profit_loss'] for trade in winning_trades
            ]) if winning_trades else 0,
            'avg_loss_amount': np.mean([
                trade['profit_loss'] for trade in losing_trades
            ]) if losing_trades else 0
        }
    
    @classmethod
    def generate_comprehensive_report(
        cls, 
        portfolio_values: List[float], 
        trades: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive trading performance report
        
        Parameters:
        -----------
        portfolio_values : List[float]
            Sequential portfolio values
        trades : List[Dict[str, Any]]
            List of trading transactions
        
        Returns:
        --------
        Dict[str, Any]
            Comprehensive performance report
        """
        returns = cls.calculate_returns(portfolio_values)
        
        return {
            'total_return': (portfolio_values[-1] / portfolio_values[0] - 1) * 100,
            'sharpe_ratio': cls.sharpe_ratio(returns),
            'maximum_drawdown': cls.maximum_drawdown(portfolio_values),
            'trade_statistics': cls.win_loss_ratio(trades)
        }