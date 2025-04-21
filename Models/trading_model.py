# trading_model.py
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

class BaseTradingModel(ABC):
    def __init__(self, model_type='gradient_boosting', random_state=42):
        """
        Consolidated trading model with multiple algorithm support
        
        Parameters:
        -----------
        model_type : str, optional (default='gradient_boosting')
            Available options: 'gradient_boosting', 'random_forest', 'svm'
        random_state : int, optional
            Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.is_trained = False
        self._initialize_model()
        self.scaler = StandardScaler() if model_type == 'svm' else None

    def _initialize_model(self):
        """Initialize the selected model with optimized parameters"""
        params = {
            'gradient_boosting': {
                'class': GradientBoostingClassifier,
                'params': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 3,
                    'random_state': self.random_state
                }
            },
            'random_forest': {
                'class': RandomForestClassifier,
                'params': {
                    'n_estimators': 100,
                    'random_state': self.random_state,
                    'n_jobs': -1  # Parallel processing
                }
            },
            'svm': {
                'class': SVC,
                'params': {
                    'kernel': 'rbf',
                    'C': 1.0,
                    'gamma': 'scale',
                    'probability': True,
                    'random_state': self.random_state
                }
            }
        }
        
        config = params[self.model_type]
        self.model = config['class'](**config['params'])

    def train(self, X, y, test_size=0.2):
        """
        Optimized training with automatic feature scaling and validation
        
        Parameters:
        -----------
        X : numpy.ndarray
            Training features
        y : numpy.ndarray
            Training target variable
        test_size : float, optional
            Proportion of data to use for validation
        """
        # Preprocess data
        if self.model_type == 'svm':
            X = self.scaler.fit_transform(X)

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Validate
        val_pred = self.predict(X_val)
        print(f"{self.model_type.upper()} Validation Performance:")
        print(classification_report(y_val, val_pred))

    def predict(self, X):
        """
        High-speed prediction with automatic preprocessing
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input features
        
        Returns:
        --------
        numpy.ndarray
            Predicted class labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before predictions")

        if self.model_type == 'svm':
            X = self.scaler.transform(X)

        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Get prediction probabilities with preprocessing
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input features
        
        Returns:
        --------
        numpy.ndarray
            Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before predictions")

        if self.model_type == 'svm':
            X = self.scaler.transform(X)

        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test):
        """
        Comprehensive performance evaluation
        
        Parameters:
        -----------
        X_test : numpy.ndarray
            Test features
        y_test : numpy.ndarray
            Test target variable
        
        Returns:
        --------
        dict
            Evaluation metrics
        """
        y_pred = self.predict(X_test)
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }

    def execute_trading_strategy(self, stock_data, top_companies, drop_threshold=0.2):
        """
        Enhanced trading strategy execution with real-time analysis
        
        Parameters:
        -----------
        stock_data : pandas.DataFrame
            DataFrame with stock symbols and historical prices
        top_companies : list
            List of top N companies to consider
        drop_threshold : float, optional
            Percentage drop threshold for buy signal
        
        Returns:
        --------
        pd.DataFrame
            DataFrame of buy signals with detailed metrics
        """
        buy_signals = []
        today = stock_data.index[-1]
        
        for symbol in top_companies:
            if symbol not in stock_data.columns:
                continue
                
            series = stock_data[symbol].dropna()
            if len(series) < 252:
                continue
                
            current_price = series.iloc[-1]
            high_52w = series.rolling(252).max().iloc[-1]
            
            if high_52w <= 0:
                continue
                
            pct_drop = (high_52w - current_price) / high_52w
            if pct_drop >= drop_threshold:
                buy_signals.append({
                    'symbol': symbol,
                    'current_price': current_price,
                    '52w_high': high_52w,
                    'pct_drop': pct_drop,
                    'action': 'BUY',
                    'date': today
                })
        
        return pd.DataFrame(buy_signals).sort_values('pct_drop', ascending=False)

# Example usage
if __name__ == "__main__":
    # Initialize model with fastest algorithm (Random Forest)
    model = BaseTradingModel(model_type='random_forest')
    
    # Example training data (replace with real data)
    X = np.random.rand(1000, 10)
    y = np.random.randint(0, 2, 1000)
    
    # Train model
    model.train(X, y)
    
    # Generate predictions
    test_data = np.random.rand(10, 10)
    predictions = model.predict(test_data)
    
    # Execute trading strategy
    top_20_companies = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 
                       'BRK.B', 'NVDA', 'META', 'UNH', 'JNJ',
                       'XOM', 'JPM', 'V', 'PG', 'MA', 
                       'HD', 'CVX', 'LLY', 'PFE', 'BAC']
    
    # Example price data (replace with real historical data)
    dates = pd.date_range(end='2023-08-01', periods=300)
    price_data = pd.DataFrame(
        np.random.lognormal(mean=0, sigma=0.1, size=(300, 20)),
        index=dates,
        columns=top_20_companies
    )
    
    signals = model.execute_trading_strategy(price_data, top_20_companies)
    print("\nTrading Signals:")
    print(signals)