from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from .base_model import BaseMLModel
import numpy as np

class GradientBoostingModel(BaseMLModel):
    def __init__(self, 
                 n_estimators=100, 
                 learning_rate=0.1, 
                 max_depth=3, 
                 random_state=42):
        """
        Initialize Gradient Boosting Model
        
        Parameters:
        -----------
        n_estimators : int, optional
            Number of boosting stages to perform
        learning_rate : float, optional
            Learning rate shrinks the contribution of each tree
        max_depth : int, optional
            Maximum depth of individual regression estimators
        random_state : int, optional
            Random seed for reproducibility
        """
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state
        )
        self.is_trained = False
    
    def train(self, X, y):
        """
        Train the Gradient Boosting model
        
        Parameters:
        -----------
        X : numpy.ndarray
            Training features
        y : numpy.ndarray
            Training target variable
        """
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Print validation performance
        val_pred = self.model.predict(X_val)
        print("Gradient Boosting Validation Performance:")
        print(classification_report(y_val, val_pred))
    
    def predict(self, X):
        """
        Make predictions
        
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
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities
        
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
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict_proba(X)