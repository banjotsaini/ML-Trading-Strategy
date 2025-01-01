import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class FeatureStore:
    @staticmethod
    def calculate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for feature engineering
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Historical stock price data
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with added technical indicators
        """
        # Moving Averages
        data['MA5'] = data['Close'].rolling(window=5).mean()
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
        
        # Relative Strength Index (RSI)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        data['BB_Upper'] = data['BB_Middle'] + 2 * data['Close'].rolling(window=20).std()
        data['BB_Lower'] = data['BB_Middle'] - 2 * data['Close'].rolling(window=20).std()
        
        return data.dropna()
    
    @staticmethod
    def prepare_ml_features(
        data: pd.DataFrame, 
        prediction_horizon: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and target variable for machine learning
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame with historical data and technical indicators
        prediction_horizon : int, optional
            Number of days to predict ahead
        
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Features and target variables
        """
        # Create target variable
        data['Target'] = np.where(
            data['Close'].shift(-prediction_horizon) > data['Close'], 
            1, 0
        )
        
        # Select features
        features = [
            'MA5', 'MA20', 'MA50', 'RSI', 'MACD', 
            'Signal_Line', 'BB_Upper', 'BB_Lower', 'Volume'
        ]
        
        X = data[features].values
        y = data['Target'].values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, y