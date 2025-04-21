from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from .base_model import BaseMLModel
import numpy as np

class RandomForestModel(BaseMLModel):
    def __init__(self, n_estimators=100, random_state=42):
        """
        Initialize Random Forest Model
        
        Parameters:
        -----------
        n_estimators : int, optional
            Number of trees in the forest
        random_state : int, optional
            Random seed for reproducibility
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, 
            random_state=random_state, 
            n_jobs=-1
        )
        self.is_trained = False
    
    def train(self, X, y):
        """
        Train the Random Forest model
        
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
        print("Random Forest Validation Performance:")
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
