import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from datetime import datetime

from config.settings import Settings
from utils.performance_metrics import calculate_sharpe_ratio, calculate_max_drawdown, calculate_cagr

class Portfolio:
    """Portfolio for backtesting"""
    
    def __init__(self, initial_capital: float = 10000.0):
        """
        Initialize portfolio
        
        Parameters:
        -----------
        initial_capital : float
            Initial capital
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}  # ticker -> {'quantity': int, 'cost_basis': float}
        self.history = []  # List of portfolio snapshots
        
    def buy(self, ticker: str, quantity: int, price: float, date: Union[str, datetime]) -> Dict[str, Any]:
        """
        Execute buy order
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol
        quantity : int
            Number of shares to buy
        price : float
            Price per share
        date : str or datetime
            Date of the trade
            
        Returns:
        --------
        Dict[str, Any]
            Trade information
        """
        cost = quantity * price
        
        # Check if we have enough cash
        if cost > self.cash:
            # Adjust quantity based on available cash
            quantity = int(self.cash / price)
            cost = quantity * price
            
            if quantity == 0:
                return {
                    'ticker': ticker,
                    'action': 'BUY',
                    'quantity': 0,
                    'price': price,
                    'cost': 0,
                    'date': date,
                    'status': 'REJECTED',
                    'reason': 'Insufficient funds'
                }
        
        # Execute buy
        self.cash -= cost
        
        # Update position
        if ticker in self.positions:
            # Update existing position
            current_quantity = self.positions[ticker]['quantity']
            current_cost = self.positions[ticker]['cost_basis'] * current_quantity
            
            # Calculate new cost basis
            new_quantity = current_quantity + quantity
            new_cost = current_cost + cost
            new_cost_basis = new_cost / new_quantity if new_quantity > 0 else 0
            
            self.positions[ticker] = {
                'quantity': new_quantity,
                'cost_basis': new_cost_basis
            }
        else:
            # Create new position
            self.positions[ticker] = {
                'quantity': quantity,
                'cost_basis': price
            }
        
        # Record trade
        trade = {
            'ticker': ticker,
            'action': 'BUY',
            'quantity': quantity,
            'price': price,
            'cost': cost,
            'date': date,
            'status': 'EXECUTED'
        }
        
        return trade
    
    def sell(self, ticker: str, quantity: int, price: float, date: Union[str, datetime]) -> Dict[str, Any]:
        """
        Execute sell order
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol
        quantity : int
            Number of shares to sell
        price : float
            Price per share
        date : str or datetime
            Date of the trade
            
        Returns:
        --------
        Dict[str, Any]
            Trade information
        """
        # Check if we have the position
        if ticker not in self.positions or self.positions[ticker]['quantity'] == 0:
            return {
                'ticker': ticker,
                'action': 'SELL',
                'quantity': 0,
                'price': price,
                'proceeds': 0,
                'date': date,
                'status': 'REJECTED',
                'reason': 'Position not found'
            }
        
        # Check if we have enough shares
        available_quantity = self.positions[ticker]['quantity']
        if quantity > available_quantity:
            quantity = available_quantity
        
        # Calculate proceeds
        proceeds = quantity * price
        
        # Execute sell
        self.cash += proceeds
        
        # Update position
        self.positions[ticker]['quantity'] -= quantity
        
        # Calculate profit/loss
        cost_basis = self.positions[ticker]['cost_basis']
        profit_loss = (price - cost_basis) * quantity
        
        # Record trade
        trade = {
            'ticker': ticker,
            'action': 'SELL',
            'quantity': quantity,
            'price': price,
            'proceeds': proceeds,
            'profit_loss': profit_loss,
            'date': date,
            'status': 'EXECUTED'
        }
        
        return trade
    
    def update_value(self, market_data: Dict[str, float], date: Union[str, datetime]) -> float:
        """
        Update portfolio value
        
        Parameters:
        -----------
        market_data : Dict[str, float]
            Current market prices (ticker -> price)
        date : str or datetime
            Date of the update
            
        Returns:
        --------
        float
            Total portfolio value
        """
        # Calculate positions value
        positions_value = 0
        for ticker, position in self.positions.items():
            if position['quantity'] > 0 and ticker in market_data:
                positions_value += position['quantity'] * market_data[ticker]
        
        # Calculate total value
        total_value = self.cash + positions_value
        
        # Record portfolio snapshot
        snapshot = {
            'date': date,
            'cash': self.cash,
            'positions_value': positions_value,
            'total_value': total_value
        }
        self.history.append(snapshot)
        
        return total_value
    
    def get_equity_curve(self) -> pd.DataFrame:
        """
        Get equity curve
        
        Returns:
        --------
        pd.DataFrame
            Equity curve with columns: date, cash, positions_value, total_value
        """
        return pd.DataFrame(self.history)
    
    def get_current_value(self, market_data: Dict[str, float]) -> float:
        """
        Get current portfolio value
        
        Parameters:
        -----------
        market_data : Dict[str, float]
            Current market prices (ticker -> price)
            
        Returns:
        --------
        float
            Total portfolio value
        """
        # Calculate positions value
        positions_value = 0
        for ticker, position in self.positions.items():
            if position['quantity'] > 0 and ticker in market_data:
                positions_value += position['quantity'] * market_data[ticker]
        
        # Calculate total value
        total_value = self.cash + positions_value
        
        return total_value
    
    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get current positions
        
        Returns:
        --------
        Dict[str, Dict[str, Any]]
            Current positions
        """
        return self.positions
    
    def get_cash(self) -> float:
        """
        Get current cash
        
        Returns:
        --------
        float
            Current cash
        """
        return self.cash

class BacktestResult:
    """Container for backtest results"""
    
    def __init__(
        self,
        strategy_name: str,
        equity_curve: pd.DataFrame,
        trades: List[Dict[str, Any]],
        parameters: Dict[str, Any]
    ):
        """
        Initialize backtest result
        
        Parameters:
        -----------
        strategy_name : str
            Name of the strategy
        equity_curve : pd.DataFrame
            Equity curve with columns: date, cash, positions_value, total_value
        trades : List[Dict[str, Any]]
            List of trades
        parameters : Dict[str, Any]
            Strategy parameters
        """
        self.strategy_name = strategy_name
        self.equity_curve = equity_curve
        self.trades = trades
        self.parameters = parameters
        
        # Calculate performance metrics
        self.metrics = self._calculate_metrics()
    
    def _calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate performance metrics
        
        Returns:
        --------
        Dict[str, float]
            Performance metrics
        """
        metrics = {}
        
        # Convert equity curve to returns
        if len(self.equity_curve) > 1:
            # Calculate daily returns
            self.equity_curve['return'] = self.equity_curve['total_value'].pct_change()
            
            # Calculate metrics
            returns = self.equity_curve['return'].dropna()
            
            # Sharpe ratio
            metrics['sharpe_ratio'] = calculate_sharpe_ratio(returns)
            
            # Maximum drawdown
            metrics['max_drawdown'] = calculate_max_drawdown(self.equity_curve['total_value'])
            
            # CAGR
            metrics['cagr'] = calculate_cagr(self.equity_curve['total_value'])
            
            # Total return
            metrics['total_return'] = (
                self.equity_curve['total_value'].iloc[-1] / 
                self.equity_curve['total_value'].iloc[0] - 1
            )
            
            # Volatility
            metrics['volatility'] = returns.std() * np.sqrt(252)  # Annualized
        
        # Trade metrics
        if self.trades:
            # Number of trades
            metrics['num_trades'] = len(self.trades)
            
            # Win rate
            winning_trades = [t for t in self.trades if t.get('profit_loss', 0) > 0]
            metrics['win_rate'] = len(winning_trades) / len(self.trades) if self.trades else 0
            
            # Average profit/loss
            profit_losses = [t.get('profit_loss', 0) for t in self.trades if 'profit_loss' in t]
            metrics['avg_profit_loss'] = np.mean(profit_losses) if profit_losses else 0
            
            # Profit factor
            gross_profit = sum([pl for pl in profit_losses if pl > 0])
            gross_loss = abs(sum([pl for pl in profit_losses if pl < 0]))
            metrics['profit_factor'] = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        return metrics
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of backtest results
        
        Returns:
        --------
        Dict[str, Any]
            Summary of backtest results
        """
        return {
            'strategy_name': self.strategy_name,
            'parameters': self.parameters,
            'metrics': self.metrics,
            'num_trades': len(self.trades),
            'start_date': self.equity_curve['date'].iloc[0] if not self.equity_curve.empty else None,
            'end_date': self.equity_curve['date'].iloc[-1] if not self.equity_curve.empty else None,
            'initial_capital': self.equity_curve['total_value'].iloc[0] if not self.equity_curve.empty else None,
            'final_capital': self.equity_curve['total_value'].iloc[-1] if not self.equity_curve.empty else None
        }

class BacktestEngine:
    """Engine for backtesting trading strategies"""
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission: float = 0.0,
        slippage: float = 0.0
    ):
        """
        Initialize backtest engine
        
        Parameters:
        -----------
        initial_capital : float
            Initial capital
        commission : float
            Commission per trade (percentage)
        slippage : float
            Slippage per trade (percentage)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.logger = logging.getLogger(__name__)
    
    def run_backtest(
        self,
        strategy: Any,
        historical_data: Dict[str, pd.DataFrame],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> BacktestResult:
        """
        Run backtest
        
        Parameters:
        -----------
        strategy : Any
            Trading strategy
        historical_data : Dict[str, pd.DataFrame]
            Historical market data (ticker -> DataFrame)
        start_date : str, optional
            Start date for backtest
        end_date : str, optional
            End date for backtest
        parameters : Dict[str, Any], optional
            Strategy parameters
            
        Returns:
        --------
        BacktestResult
            Backtest results
        """
        self.logger.info(f"Running backtest for {strategy.__class__.__name__}")
        
        # Initialize portfolio
        portfolio = Portfolio(initial_capital=self.initial_capital)
        
        # Initialize trades list
        trades = []
        
        # Get common date range for all tickers
        dates = self._get_common_dates(historical_data, start_date, end_date)
        
        if not dates:
            self.logger.warning("No common dates found for backtest")
            return BacktestResult(
                strategy_name=strategy.__class__.__name__,
                equity_curve=pd.DataFrame(),
                trades=[],
                parameters=parameters or {}
            )
        
        self.logger.info(f"Backtest period: {dates[0]} to {dates[-1]} ({len(dates)} days)")
        
        # Run backtest day by day
        for i, date in enumerate(dates):
            # Get current market data
            current_data = self._get_data_for_date(historical_data, date)
            
            # Skip if no data for this date
            if not current_data:
                continue
            
            # Get historical data up to current date
            historical_data_to_date = self._get_data_to_date(historical_data, date)
            
            # Generate signals
            signals = strategy.generate_signals_for_backtest(
                historical_data_to_date,
                current_data,
                portfolio,
                parameters
            )
            
            # Execute signals
            for signal in signals:
                ticker = signal['ticker']
                action = signal['action']
                quantity = signal['quantity']
                price = current_data[ticker]
                
                # Apply slippage
                if action == 'BUY':
                    price *= (1 + self.slippage)
                elif action == 'SELL':
                    price *= (1 - self.slippage)
                
                # Execute trade
                if action == 'BUY':
                    trade = portfolio.buy(ticker, quantity, price, date)
                elif action == 'SELL':
                    trade = portfolio.sell(ticker, quantity, price, date)
                else:
                    continue
                
                # Apply commission
                if trade['status'] == 'EXECUTED':
                    commission_amount = trade.get('cost', trade.get('proceeds', 0)) * self.commission
                    portfolio.cash -= commission_amount
                    trade['commission'] = commission_amount
                
                # Add trade to list
                trades.append(trade)
            
            # Update portfolio value
            portfolio.update_value(current_data, date)
            
            # Log progress
            if i % 100 == 0:
                self.logger.debug(f"Processed {i+1}/{len(dates)} days")
        
        # Get equity curve
        equity_curve = portfolio.get_equity_curve()
        
        # Create backtest result
        result = BacktestResult(
            strategy_name=strategy.__class__.__name__,
            equity_curve=equity_curve,
            trades=trades,
            parameters=parameters or {}
        )
        
        self.logger.info(f"Backtest completed: {result.get_summary()}")
        
        return result
    
    def _get_common_dates(
        self,
        historical_data: Dict[str, pd.DataFrame],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[str]:
        """
        Get common dates for all tickers
        
        Parameters:
        -----------
        historical_data : Dict[str, pd.DataFrame]
            Historical market data (ticker -> DataFrame)
        start_date : str, optional
            Start date for backtest
        end_date : str, optional
            End date for backtest
            
        Returns:
        --------
        List[str]
            List of common dates
        """
        # Get dates for each ticker
        ticker_dates = {}
        for ticker, data in historical_data.items():
            # Ensure date is the index
            if 'date' in data.columns:
                data = data.set_index('date')
            
            # Get dates
            ticker_dates[ticker] = set(data.index.astype(str))
        
        # Get common dates
        common_dates = set.intersection(*ticker_dates.values()) if ticker_dates else set()
        
        # Filter by start and end dates
        if start_date:
            common_dates = {d for d in common_dates if d >= start_date}
        if end_date:
            common_dates = {d for d in common_dates if d <= end_date}
        
        # Sort dates
        return sorted(list(common_dates))
    
    def _get_data_for_date(
        self,
        historical_data: Dict[str, pd.DataFrame],
        date: str
    ) -> Dict[str, float]:
        """
        Get market data for a specific date
        
        Parameters:
        -----------
        historical_data : Dict[str, pd.DataFrame]
            Historical market data (ticker -> DataFrame)
        date : str
            Date
            
        Returns:
        --------
        Dict[str, float]
            Market data for the date (ticker -> price)
        """
        result = {}
        
        for ticker, data in historical_data.items():
            # Ensure date is the index
            if 'date' in data.columns:
                data = data.set_index('date')
            
            # Get data for date
            if date in data.index:
                # Use adjusted close if available, otherwise close
                if 'adj_close' in data.columns:
                    result[ticker] = data.loc[date, 'adj_close']
                elif 'close' in data.columns:
                    result[ticker] = data.loc[date, 'close']
        
        return result
    
    def _get_data_to_date(
        self,
        historical_data: Dict[str, pd.DataFrame],
        date: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Get historical data up to a specific date
        
        Parameters:
        -----------
        historical_data : Dict[str, pd.DataFrame]
            Historical market data (ticker -> DataFrame)
        date : str
            Date
            
        Returns:
        --------
        Dict[str, pd.DataFrame]
            Historical data up to the date
        """
        result = {}
        
        for ticker, data in historical_data.items():
            # Ensure date is the index
            if 'date' in data.columns:
                date_col = 'date'
                data_copy = data.copy()
            else:
                date_col = data.index.name or 'index'
                data_copy = data.reset_index()
            
            # Filter data up to date
            filtered_data = data_copy[data_copy[date_col] <= date]
            
            # Set index back if needed
            if 'date' not in data.columns:
                filtered_data = filtered_data.set_index(date_col)
            
            result[ticker] = filtered_data
        
        return result