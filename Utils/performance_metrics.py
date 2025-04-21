import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union, Optional

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Calculate Sharpe ratio (risk-adjusted return)
    
    Parameters:
    -----------
    returns : pd.Series
        Series of returns
    risk_free_rate : float
        Risk-free rate (annualized)
    periods_per_year : int
        Number of periods in a year (252 for daily, 12 for monthly, etc.)
        
    Returns:
    --------
    float
        Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0
    
    # Convert risk-free rate to per-period
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    
    # Calculate excess returns
    excess_returns = returns - rf_per_period
    
    # Calculate Sharpe ratio
    sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year)
    
    return sharpe

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Calculate Sortino ratio (downside risk-adjusted return)
    
    Parameters:
    -----------
    returns : pd.Series
        Series of returns
    risk_free_rate : float
        Risk-free rate (annualized)
    periods_per_year : int
        Number of periods in a year (252 for daily, 12 for monthly, etc.)
        
    Returns:
    --------
    float
        Sortino ratio
    """
    if len(returns) < 2:
        return 0.0
    
    # Convert risk-free rate to per-period
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    
    # Calculate excess returns
    excess_returns = returns - rf_per_period
    
    # Calculate downside deviation (only negative returns)
    downside_returns = excess_returns[excess_returns < 0]
    downside_deviation = np.sqrt(np.mean(downside_returns ** 2)) if len(downside_returns) > 0 else 0.0
    
    # Calculate Sortino ratio
    sortino = excess_returns.mean() / downside_deviation * np.sqrt(periods_per_year) if downside_deviation > 0 else 0.0
    
    return sortino

def calculate_max_drawdown(equity_curve: Union[pd.Series, List[float]]) -> float:
    """
    Calculate maximum drawdown
    
    Parameters:
    -----------
    equity_curve : pd.Series or List[float]
        Equity curve (portfolio value over time)
        
    Returns:
    --------
    float
        Maximum drawdown as a positive percentage (0 to 1)
    """
    if isinstance(equity_curve, list):
        equity_curve = pd.Series(equity_curve)
    
    if len(equity_curve) < 2:
        return 0.0
    
    # Calculate running maximum
    running_max = equity_curve.cummax()
    
    # Calculate drawdown
    drawdown = (equity_curve - running_max) / running_max
    
    # Get maximum drawdown
    max_drawdown = drawdown.min()
    
    return abs(max_drawdown)

def calculate_cagr(equity_curve: Union[pd.Series, List[float]], periods_per_year: int = 252) -> float:
    """
    Calculate Compound Annual Growth Rate
    
    Parameters:
    -----------
    equity_curve : pd.Series or List[float]
        Equity curve (portfolio value over time)
    periods_per_year : int
        Number of periods in a year (252 for daily, 12 for monthly, etc.)
        
    Returns:
    --------
    float
        CAGR as a decimal (e.g., 0.1 for 10%)
    """
    if isinstance(equity_curve, list):
        equity_curve = pd.Series(equity_curve)
    
    if len(equity_curve) < 2:
        return 0.0
    
    # Get initial and final values
    initial_value = equity_curve.iloc[0]
    final_value = equity_curve.iloc[-1]
    
    # Calculate number of years
    num_periods = len(equity_curve) - 1
    num_years = num_periods / periods_per_year
    
    # Calculate CAGR
    cagr = (final_value / initial_value) ** (1 / num_years) - 1
    
    return cagr

def calculate_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate annualized volatility
    
    Parameters:
    -----------
    returns : pd.Series
        Series of returns
    periods_per_year : int
        Number of periods in a year (252 for daily, 12 for monthly, etc.)
        
    Returns:
    --------
    float
        Annualized volatility
    """
    if len(returns) < 2:
        return 0.0
    
    # Calculate volatility
    volatility = returns.std() * np.sqrt(periods_per_year)
    
    return volatility

def calculate_calmar_ratio(returns: pd.Series, equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate Calmar ratio (return / max drawdown)
    
    Parameters:
    -----------
    returns : pd.Series
        Series of returns
    equity_curve : pd.Series
        Equity curve (portfolio value over time)
    periods_per_year : int
        Number of periods in a year (252 for daily, 12 for monthly, etc.)
        
    Returns:
    --------
    float
        Calmar ratio
    """
    if len(returns) < 2 or len(equity_curve) < 2:
        return 0.0
    
    # Calculate CAGR
    cagr = calculate_cagr(equity_curve, periods_per_year)
    
    # Calculate max drawdown
    max_dd = calculate_max_drawdown(equity_curve)
    
    # Calculate Calmar ratio
    calmar = cagr / max_dd if max_dd > 0 else 0.0
    
    return calmar

def calculate_win_rate(trades: List[Dict[str, Any]]) -> float:
    """
    Calculate win rate (percentage of winning trades)
    
    Parameters:
    -----------
    trades : List[Dict[str, Any]]
        List of trades with 'profit_loss' key
        
    Returns:
    --------
    float
        Win rate as a decimal (0 to 1)
    """
    if not trades:
        return 0.0
    
    # Count winning trades
    winning_trades = sum(1 for trade in trades if trade.get('profit_loss', 0) > 0)
    
    # Calculate win rate
    win_rate = winning_trades / len(trades)
    
    return win_rate

def calculate_profit_factor(trades: List[Dict[str, Any]]) -> float:
    """
    Calculate profit factor (gross profit / gross loss)
    
    Parameters:
    -----------
    trades : List[Dict[str, Any]]
        List of trades with 'profit_loss' key
        
    Returns:
    --------
    float
        Profit factor
    """
    if not trades:
        return 0.0
    
    # Calculate gross profit and loss
    gross_profit = sum(trade.get('profit_loss', 0) for trade in trades if trade.get('profit_loss', 0) > 0)
    gross_loss = sum(abs(trade.get('profit_loss', 0)) for trade in trades if trade.get('profit_loss', 0) < 0)
    
    # Calculate profit factor
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    return profit_factor

def calculate_average_trade(trades: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate average trade metrics
    
    Parameters:
    -----------
    trades : List[Dict[str, Any]]
        List of trades with 'profit_loss' key
        
    Returns:
    --------
    Dict[str, float]
        Average trade metrics
    """
    if not trades:
        return {
            'avg_profit_loss': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'avg_holding_period': 0.0
        }
    
    # Calculate profit/loss metrics
    profit_losses = [trade.get('profit_loss', 0) for trade in trades if 'profit_loss' in trade]
    wins = [pl for pl in profit_losses if pl > 0]
    losses = [pl for pl in profit_losses if pl < 0]
    
    avg_profit_loss = np.mean(profit_losses) if profit_losses else 0.0
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = np.mean(losses) if losses else 0.0
    
    # Calculate holding period if dates are available
    holding_periods = []
    for trade in trades:
        if 'entry_date' in trade and 'exit_date' in trade:
            entry_date = pd.to_datetime(trade['entry_date'])
            exit_date = pd.to_datetime(trade['exit_date'])
            holding_period = (exit_date - entry_date).days
            holding_periods.append(holding_period)
    
    avg_holding_period = np.mean(holding_periods) if holding_periods else 0.0
    
    return {
        'avg_profit_loss': avg_profit_loss,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'avg_holding_period': avg_holding_period
    }

def calculate_drawdowns(equity_curve: pd.Series) -> pd.DataFrame:
    """
    Calculate all drawdowns
    
    Parameters:
    -----------
    equity_curve : pd.Series
        Equity curve (portfolio value over time)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with drawdown information
    """
    if len(equity_curve) < 2:
        return pd.DataFrame(columns=['start_date', 'end_date', 'recovery_date', 'drawdown', 'duration', 'recovery'])
    
    # Calculate running maximum
    running_max = equity_curve.cummax()
    
    # Calculate drawdown
    drawdown = (equity_curve - running_max) / running_max
    
    # Find drawdown periods
    is_drawdown = drawdown < 0
    
    # Initialize variables
    drawdowns = []
    current_drawdown = None
    
    # Iterate through equity curve
    for date, value in drawdown.items():
        if is_drawdown[date]:
            # In drawdown
            if current_drawdown is None:
                # Start of new drawdown
                current_drawdown = {
                    'start_date': date,
                    'max_drawdown': value,
                    'max_drawdown_date': date
                }
            elif value < current_drawdown['max_drawdown']:
                # New maximum drawdown
                current_drawdown['max_drawdown'] = value
                current_drawdown['max_drawdown_date'] = date
        elif current_drawdown is not None:
            # End of drawdown
            current_drawdown['end_date'] = date
            current_drawdown['duration'] = (pd.to_datetime(date) - pd.to_datetime(current_drawdown['start_date'])).days
            current_drawdown['recovery'] = (pd.to_datetime(date) - pd.to_datetime(current_drawdown['max_drawdown_date'])).days
            
            # Add to list
            drawdowns.append(current_drawdown)
            
            # Reset
            current_drawdown = None
    
    # Handle ongoing drawdown
    if current_drawdown is not None:
        current_drawdown['end_date'] = None
        current_drawdown['duration'] = (pd.to_datetime(drawdown.index[-1]) - pd.to_datetime(current_drawdown['start_date'])).days
        current_drawdown['recovery'] = None
        
        # Add to list
        drawdowns.append(current_drawdown)
    
    # Convert to DataFrame
    df_drawdowns = pd.DataFrame(drawdowns)
    
    # Rename max_drawdown to drawdown
    if 'max_drawdown' in df_drawdowns.columns:
        df_drawdowns = df_drawdowns.rename(columns={'max_drawdown': 'drawdown'})
        df_drawdowns['drawdown'] = df_drawdowns['drawdown'].abs()
    
    return df_drawdowns

def calculate_monthly_returns(equity_curve: pd.Series) -> pd.DataFrame:
    """
    Calculate monthly returns
    
    Parameters:
    -----------
    equity_curve : pd.Series
        Equity curve (portfolio value over time) with datetime index
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with monthly returns
    """
    if len(equity_curve) < 2:
        return pd.DataFrame()
    
    # Ensure datetime index
    if not isinstance(equity_curve.index, pd.DatetimeIndex):
        equity_curve.index = pd.to_datetime(equity_curve.index)
    
    # Resample to month-end
    monthly_equity = equity_curve.resample('M').last()
    
    # Calculate returns
    monthly_returns = monthly_equity.pct_change().dropna()
    
    # Create DataFrame
    df_monthly = pd.DataFrame({
        'year': monthly_returns.index.year,
        'month': monthly_returns.index.month,
        'return': monthly_returns.values
    })
    
    # Pivot to year x month format
    df_pivot = df_monthly.pivot(index='year', columns='month', values='return')
    
    # Add annual return
    df_pivot['Annual'] = df_pivot.apply(lambda row: (1 + row.dropna()).prod() - 1, axis=1)
    
    return df_pivot

def calculate_performance_summary(
    equity_curve: pd.Series,
    trades: List[Dict[str, Any]],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> Dict[str, Any]:
    """
    Calculate comprehensive performance summary
    
    Parameters:
    -----------
    equity_curve : pd.Series
        Equity curve (portfolio value over time)
    trades : List[Dict[str, Any]]
        List of trades
    risk_free_rate : float
        Risk-free rate (annualized)
    periods_per_year : int
        Number of periods in a year (252 for daily, 12 for monthly, etc.)
        
    Returns:
    --------
    Dict[str, Any]
        Performance summary
    """
    # Calculate returns
    returns = equity_curve.pct_change().dropna()
    
    # Calculate metrics
    summary = {
        # Return metrics
        'total_return': equity_curve.iloc[-1] / equity_curve.iloc[0] - 1 if len(equity_curve) > 1 else 0.0,
        'cagr': calculate_cagr(equity_curve, periods_per_year),
        'volatility': calculate_volatility(returns, periods_per_year),
        
        # Risk-adjusted metrics
        'sharpe_ratio': calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year),
        'sortino_ratio': calculate_sortino_ratio(returns, risk_free_rate, periods_per_year),
        'max_drawdown': calculate_max_drawdown(equity_curve),
        'calmar_ratio': calculate_calmar_ratio(returns, equity_curve, periods_per_year),
        
        # Trade metrics
        'num_trades': len(trades),
        'win_rate': calculate_win_rate(trades),
        'profit_factor': calculate_profit_factor(trades),
        'avg_trade': calculate_average_trade(trades),
        
        # Time metrics
        'start_date': equity_curve.index[0] if len(equity_curve) > 0 else None,
        'end_date': equity_curve.index[-1] if len(equity_curve) > 0 else None,
        'duration_days': (pd.to_datetime(equity_curve.index[-1]) - pd.to_datetime(equity_curve.index[0])).days if len(equity_curve) > 1 else 0
    }
    
    return summary