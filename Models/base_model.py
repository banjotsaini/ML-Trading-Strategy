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