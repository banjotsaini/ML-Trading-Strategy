from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from .base_model import BaseMLModel
import numpy as np

class SVMModel(BaseMLModel):
    def __init__(self, 
                 kernel='rbf', 
                 C=1.0, 
                 gamma='scale', 
                 random_state=42):
        """
        Initialize Support Vector Machine Model
        
        Parameters:
        -----------
        kernel : str, optional
            Specifies the kernel type to be used
        C : float, optional
            Regularization parameter
        gamma : str or float, optional
            Kernel coefficient
        random_state : int, optional
            Random seed for reproducibility
        """
        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=True,
            random_state=random_state
        )
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def train(self, X, y):
        """
        Train the SVM model
        
        Parameters:
        -----------
        X : numpy.ndarray
            Training features
        y : numpy.ndarray
            Training target variable
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Print validation performance
        val_pred = self.model.predict(X_val)
        print("SVM Validation Performance:")
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
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict(X_scaled)
    
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
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict_proba(X_scaled)