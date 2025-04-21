import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime

from backtesting.engine import BacktestResult

def plot_equity_curve(
    result: BacktestResult,
    benchmark: Optional[pd.Series] = None,
    figsize: Tuple[int, int] = (12, 6),
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot equity curve
    
    Parameters:
    -----------
    result : BacktestResult
        Backtest result
    benchmark : pd.Series, optional
        Benchmark equity curve
    figsize : Tuple[int, int]
        Figure size
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get equity curve
    equity_curve = result.equity_curve
    
    # Convert date to datetime if needed
    if 'date' in equity_curve.columns and not pd.api.types.is_datetime64_any_dtype(equity_curve['date']):
        equity_curve['date'] = pd.to_datetime(equity_curve['date'])
    
    # Plot equity curve
    if 'date' in equity_curve.columns:
        ax.plot(equity_curve['date'], equity_curve['total_value'], label='Strategy')
        
        # Plot benchmark if provided
        if benchmark is not None:
            # Align benchmark to equity curve dates
            benchmark_aligned = benchmark.reindex(equity_curve['date'], method='ffill')
            
            # Normalize benchmark to start at the same value as the strategy
            benchmark_normalized = benchmark_aligned * (equity_curve['total_value'].iloc[0] / benchmark_aligned.iloc[0])
            
            ax.plot(equity_curve['date'], benchmark_normalized, label='Benchmark', alpha=0.7)
    else:
        ax.plot(equity_curve.index, equity_curve['total_value'], label='Strategy')
        
        # Plot benchmark if provided
        if benchmark is not None:
            # Align benchmark to equity curve dates
            benchmark_aligned = benchmark.reindex(equity_curve.index, method='ffill')
            
            # Normalize benchmark to start at the same value as the strategy
            benchmark_normalized = benchmark_aligned * (equity_curve['total_value'].iloc[0] / benchmark_aligned.iloc[0])
            
            ax.plot(equity_curve.index, benchmark_normalized, label='Benchmark', alpha=0.7)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add labels
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value')
    
    # Add title
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Equity Curve - {result.strategy_name}')
    
    # Add legend
    ax.legend()
    
    # Add metrics as text
    metrics_text = (
        f"Total Return: {result.metrics.get('total_return', 0):.2%}\n"
        f"CAGR: {result.metrics.get('cagr', 0):.2%}\n"
        f"Sharpe Ratio: {result.metrics.get('sharpe_ratio', 0):.2f}\n"
        f"Max Drawdown: {result.metrics.get('max_drawdown', 0):.2%}"
    )
    
    # Position text in the upper left
    ax.text(
        0.02, 0.95, metrics_text,
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
    )
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_drawdown(
    result: BacktestResult,
    figsize: Tuple[int, int] = (12, 6),
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot drawdown
    
    Parameters:
    -----------
    result : BacktestResult
        Backtest result
    figsize : Tuple[int, int]
        Figure size
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get equity curve
    equity_curve = result.equity_curve
    
    # Convert date to datetime if needed
    if 'date' in equity_curve.columns and not pd.api.types.is_datetime64_any_dtype(equity_curve['date']):
        equity_curve['date'] = pd.to_datetime(equity_curve['date'])
    
    # Calculate drawdown
    if 'date' in equity_curve.columns:
        equity = equity_curve.set_index('date')['total_value']
    else:
        equity = equity_curve['total_value']
    
    # Calculate running maximum
    running_max = equity.cummax()
    
    # Calculate drawdown
    drawdown = (equity - running_max) / running_max
    
    # Plot drawdown
    if 'date' in equity_curve.columns:
        ax.fill_between(drawdown.index, 0, drawdown, color='red', alpha=0.3)
        ax.plot(drawdown.index, drawdown, color='red', alpha=0.5)
    else:
        ax.fill_between(range(len(drawdown)), 0, drawdown, color='red', alpha=0.3)
        ax.plot(range(len(drawdown)), drawdown, color='red', alpha=0.5)
    
    # Format x-axis
    if 'date' in equity_curve.columns:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add labels
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown')
    
    # Add title
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Drawdown - {result.strategy_name}')
    
    # Add max drawdown as text
    max_dd = result.metrics.get('max_drawdown', 0)
    ax.text(
        0.02, 0.05,
        f"Max Drawdown: {max_dd:.2%}",
        transform=ax.transAxes,
        verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
    )
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_returns_distribution(
    result: BacktestResult,
    figsize: Tuple[int, int] = (12, 6),
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot returns distribution
    
    Parameters:
    -----------
    result : BacktestResult
        Backtest result
    figsize : Tuple[int, int]
        Figure size
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get equity curve
    equity_curve = result.equity_curve
    
    # Convert date to datetime if needed
    if 'date' in equity_curve.columns and not pd.api.types.is_datetime64_any_dtype(equity_curve['date']):
        equity_curve['date'] = pd.to_datetime(equity_curve['date'])
    
    # Calculate returns
    if 'date' in equity_curve.columns:
        equity = equity_curve.set_index('date')['total_value']
    else:
        equity = equity_curve['total_value']
    
    returns = equity.pct_change().dropna()
    
    # Plot returns distribution
    sns.histplot(returns, kde=True, ax=ax)
    
    # Add normal distribution
    x = np.linspace(returns.min(), returns.max(), 100)
    y = np.exp(-(x - returns.mean())**2 / (2 * returns.std()**2)) / (returns.std() * np.sqrt(2 * np.pi))
    y = y * len(returns) * (returns.max() - returns.min()) / 100  # Scale to match histogram
    ax.plot(x, y, 'r--', linewidth=2)
    
    # Add vertical line at 0
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add labels
    ax.set_xlabel('Return')
    ax.set_ylabel('Frequency')
    
    # Add title
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Returns Distribution - {result.strategy_name}')
    
    # Add metrics as text
    metrics_text = (
        f"Mean: {returns.mean():.2%}\n"
        f"Std Dev: {returns.std():.2%}\n"
        f"Skew: {returns.skew():.2f}\n"
        f"Kurtosis: {returns.kurtosis():.2f}"
    )
    
    # Position text in the upper right
    ax.text(
        0.98, 0.95, metrics_text,
        transform=ax.transAxes,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
    )
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_monthly_returns(
    result: BacktestResult,
    figsize: Tuple[int, int] = (12, 8),
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot monthly returns heatmap
    
    Parameters:
    -----------
    result : BacktestResult
        Backtest result
    figsize : Tuple[int, int]
        Figure size
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get equity curve
    equity_curve = result.equity_curve
    
    # Convert date to datetime if needed
    if 'date' in equity_curve.columns and not pd.api.types.is_datetime64_any_dtype(equity_curve['date']):
        equity_curve['date'] = pd.to_datetime(equity_curve['date'])
    
    # Calculate returns
    if 'date' in equity_curve.columns:
        equity = equity_curve.set_index('date')['total_value']
    else:
        equity = equity_curve['total_value']
    
    # Resample to month-end
    monthly_equity = equity.resample('M').last()
    
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
    
    # Add month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    df_pivot.columns = [month_names[i-1] for i in df_pivot.columns]
    
    # Add annual return
    df_pivot['Annual'] = df_pivot.apply(lambda row: (1 + row.dropna()).prod() - 1, axis=1)
    
    # Plot heatmap
    sns.heatmap(
        df_pivot,
        annot=True,
        fmt='.1%',
        cmap='RdYlGn',
        center=0,
        linewidths=1,
        ax=ax,
        cbar_kws={'label': 'Monthly Return'}
    )
    
    # Add title
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Monthly Returns - {result.strategy_name}')
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_trade_analysis(
    result: BacktestResult,
    figsize: Tuple[int, int] = (12, 10),
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot trade analysis
    
    Parameters:
    -----------
    result : BacktestResult
        Backtest result
    figsize : Tuple[int, int]
        Figure size
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    
    # Get trades
    trades = result.trades
    
    # Extract profit/loss
    profit_loss = [trade.get('profit_loss', 0) for trade in trades if 'profit_loss' in trade]
    
    # Plot profit/loss distribution
    axs[0, 0].hist(profit_loss, bins=20, alpha=0.7)
    axs[0, 0].axvline(0, color='black', linestyle='--', alpha=0.5)
    axs[0, 0].set_title('Profit/Loss Distribution')
    axs[0, 0].set_xlabel('Profit/Loss')
    axs[0, 0].set_ylabel('Frequency')
    axs[0, 0].grid(True, alpha=0.3)
    
    # Plot profit/loss over time
    if profit_loss:
        axs[0, 1].plot(range(len(profit_loss)), profit_loss, marker='o', alpha=0.7)
        axs[0, 1].axhline(0, color='black', linestyle='--', alpha=0.5)
        axs[0, 1].set_title('Profit/Loss Over Time')
        axs[0, 1].set_xlabel('Trade Number')
        axs[0, 1].set_ylabel('Profit/Loss')
        axs[0, 1].grid(True, alpha=0.3)
    
    # Plot cumulative profit/loss
    if profit_loss:
        cumulative_pl = np.cumsum(profit_loss)
        axs[1, 0].plot(range(len(cumulative_pl)), cumulative_pl, marker='o', alpha=0.7)
        axs[1, 0].axhline(0, color='black', linestyle='--', alpha=0.5)
        axs[1, 0].set_title('Cumulative Profit/Loss')
        axs[1, 0].set_xlabel('Trade Number')
        axs[1, 0].set_ylabel('Cumulative Profit/Loss')
        axs[1, 0].grid(True, alpha=0.3)
    
    # Plot trade metrics
    metrics = [
        f"Number of Trades: {len(trades)}",
        f"Win Rate: {result.metrics.get('win_rate', 0):.2%}",
        f"Profit Factor: {result.metrics.get('profit_factor', 0):.2f}",
        f"Avg Profit/Loss: {result.metrics.get('avg_profit_loss', 0):.2f}",
        f"Avg Win: {result.metrics.get('avg_win', 0):.2f}",
        f"Avg Loss: {result.metrics.get('avg_loss', 0):.2f}",
        f"Avg Holding Period: {result.metrics.get('avg_holding_period', 0):.1f} days"
    ]
    
    axs[1, 1].axis('off')
    axs[1, 1].text(
        0.5, 0.5, '\n'.join(metrics),
        horizontalalignment='center',
        verticalalignment='center',
        transform=axs[1, 1].transAxes,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
    )
    
    # Add title
    if title:
        fig.suptitle(title, fontsize=16)
    else:
        fig.suptitle(f'Trade Analysis - {result.strategy_name}', fontsize=16)
    
    # Tight layout
    plt.tight_layout()
    if title:
        plt.subplots_adjust(top=0.9)
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def compare_strategies(
    results: List[BacktestResult],
    figsize: Tuple[int, int] = (12, 10),
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare multiple strategies
    
    Parameters:
    -----------
    results : List[BacktestResult]
        List of backtest results
    figsize : Tuple[int, int]
        Figure size
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    
    # Plot equity curves
    for result in results:
        equity_curve = result.equity_curve
        
        # Convert date to datetime if needed
        if 'date' in equity_curve.columns and not pd.api.types.is_datetime64_any_dtype(equity_curve['date']):
            equity_curve['date'] = pd.to_datetime(equity_curve['date'])
        
        # Normalize equity curve
        if 'date' in equity_curve.columns:
            equity = equity_curve.set_index('date')['total_value']
        else:
            equity = equity_curve['total_value']
        
        normalized_equity = equity / equity.iloc[0]
        
        # Plot normalized equity curve
        if 'date' in equity_curve.columns:
            axs[0, 0].plot(normalized_equity.index, normalized_equity, label=result.strategy_name)
        else:
            axs[0, 0].plot(range(len(normalized_equity)), normalized_equity, label=result.strategy_name)
    
    axs[0, 0].set_title('Normalized Equity Curves')
    axs[0, 0].set_xlabel('Date')
    axs[0, 0].set_ylabel('Normalized Value')
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend()
    
    # Plot drawdowns
    for result in results:
        equity_curve = result.equity_curve
        
        # Convert date to datetime if needed
        if 'date' in equity_curve.columns and not pd.api.types.is_datetime64_any_dtype(equity_curve['date']):
            equity_curve['date'] = pd.to_datetime(equity_curve['date'])
        
        # Calculate drawdown
        if 'date' in equity_curve.columns:
            equity = equity_curve.set_index('date')['total_value']
        else:
            equity = equity_curve['total_value']
        
        # Calculate running maximum
        running_max = equity.cummax()
        
        # Calculate drawdown
        drawdown = (equity - running_max) / running_max
        
        # Plot drawdown
        if 'date' in equity_curve.columns:
            axs[0, 1].plot(drawdown.index, drawdown, label=result.strategy_name)
        else:
            axs[0, 1].plot(range(len(drawdown)), drawdown, label=result.strategy_name)
    
    axs[0, 1].set_title('Drawdowns')
    axs[0, 1].set_xlabel('Date')
    axs[0, 1].set_ylabel('Drawdown')
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].legend()
    
    # Plot performance metrics
    metrics = ['total_return', 'cagr', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'profit_factor']
    metric_names = ['Total Return', 'CAGR', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate', 'Profit Factor']
    
    # Create DataFrame for metrics
    metrics_data = []
    for result in results:
        metrics_row = [result.strategy_name]
        for metric in metrics:
            metrics_row.append(result.metrics.get(metric, 0))
        metrics_data.append(metrics_row)
    
    metrics_df = pd.DataFrame(metrics_data, columns=['Strategy'] + metric_names)
    
    # Plot metrics as bar chart
    metrics_df.set_index('Strategy').plot(kind='bar', ax=axs[1, 0])
    axs[1, 0].set_title('Performance Metrics')
    axs[1, 0].set_ylabel('Value')
    axs[1, 0].grid(True, alpha=0.3)
    plt.setp(axs[1, 0].xaxis.get_majorticklabels(), rotation=45)
    
    # Plot metrics as table
    axs[1, 1].axis('off')
    
    # Format metrics for table
    table_data = []
    for _, row in metrics_df.iterrows():
        table_row = [row['Strategy']]
        for metric, value in zip(metric_names, row[1:]):
            if metric in ['Total Return', 'CAGR', 'Max Drawdown', 'Win Rate']:
                table_row.append(f"{value:.2%}")
            else:
                table_row.append(f"{value:.2f}")
        table_data.append(table_row)
    
    table = axs[1, 1].table(
        cellText=table_data,
        colLabels=['Strategy'] + metric_names,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Add title
    if title:
        fig.suptitle(title, fontsize=16)
    else:
        fig.suptitle('Strategy Comparison', fontsize=16)
    
    # Tight layout
    plt.tight_layout()
    if title:
        plt.subplots_adjust(top=0.9)
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_optimization_results(
    param_values: List[Dict[str, Any]],
    metric_values: List[float],
    param_names: List[str],
    metric_name: str,
    figsize: Tuple[int, int] = (12, 10),
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot optimization results
    
    Parameters:
    -----------
    param_values : List[Dict[str, Any]]
        List of parameter combinations
    metric_values : List[float]
        List of metric values
    param_names : List[str]
        List of parameter names to plot
    metric_name : str
        Name of the metric
    figsize : Tuple[int, int]
        Figure size
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    # Create figure
    n_params = len(param_names)
    n_cols = min(n_params, 3)
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Flatten axs if needed
    if n_rows == 1 and n_cols == 1:
        axs = np.array([axs])
    elif n_rows == 1 or n_cols == 1:
        axs = axs.flatten()
    
    # Create DataFrame
    df = pd.DataFrame(param_values)
    df[metric_name] = metric_values
    
    # Plot each parameter
    for i, param in enumerate(param_names):
        row = i // n_cols
        col = i % n_cols
        
        if n_rows > 1 and n_cols > 1:
            ax = axs[row, col]
        else:
            ax = axs[i]
        
        # Group by parameter
        grouped = df.groupby(param)[metric_name].mean().reset_index()
        
        # Plot
        ax.plot(grouped[param], grouped[metric_name], marker='o')
        ax.set_title(f'{param} vs {metric_name}')
        ax.set_xlabel(param)
        ax.set_ylabel(metric_name)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_params, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        if n_rows > 1 and n_cols > 1:
            axs[row, col].axis('off')
        else:
            axs[i].axis('off')
    
    # Add title
    if title:
        fig.suptitle(title, fontsize=16)
    else:
        fig.suptitle(f'Parameter Optimization Results - {metric_name}', fontsize=16)
    
    # Tight layout
    plt.tight_layout()
    if title:
        plt.subplots_adjust(top=0.9)
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig