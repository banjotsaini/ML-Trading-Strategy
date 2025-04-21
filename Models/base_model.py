from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

class BaseMLModel(ABC):
    @abstractmethod
    def train(self, X_train, y_train):
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Make predictions"""
        pass
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Parameters:
        -----------
        X_test : numpy.ndarray
            Test features
        y_test : numpy.ndarray
            Test target variable
        
        Returns:
        --------
        dict
            Performance metrics
        """
        y_pred = self.predict(X_test)
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }
    
    def execute_trading_strategy(self, stock_data):
        """
        Identify stocks to buy when their price drops 20%+ from the 52-week high.
        
        Parameters:
        -----------
        stock_data : pandas.DataFrame
            DataFrame with stock symbols as columns and dates as index,
            containing adjusted closing prices.
        
        Returns:
        --------
        list of tuples
            Each tuple contains (stock_symbol, action, current_price, pct_drop)
        """
        buy_signals = []
        for symbol in stock_data.columns:
            if len(stock_data[symbol]) < 252:  # Ensure enough historical data
                continue
            current_price = stock_data[symbol].iloc[-1]
            high_52w = stock_data[symbol].rolling(window=252).max().iloc[-1]
            if high_52w == 0:
                continue  # Avoid division by zero
            pct_drop = (high_52w - current_price) / high_52w
            if pct_drop >= 0.2:
                buy_signals.append(
                    (symbol, 'BUY', current_price, f"{pct_drop:.1%}")
                )
        return buy_signals