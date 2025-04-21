from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from config.settings import Settings
from models.base_model import BaseMLModel
from models.ensemble_methods import EnsembleModelSelector

class MLTradingStrategy:
    def __init__(
        self, 
        models: List[BaseMLModel], 
        initial_capital: float = Settings.INITIAL_CAPITAL,
        risk_percentage: float = Settings.RISK_PERCENTAGE,
        ensemble_method: str = 'dynamic_ensemble',
        model_evaluation_window: int = 10
    ):
        """
        Initialize ML Trading Strategy
        
        Parameters:
        -----------
        models : List[BaseMLModel]
            List of machine learning models
        initial_capital : float, optional
            Starting trading capital
        risk_percentage : float, optional
            Percentage of capital to risk per trade
        ensemble_method : str, optional
            Ensemble method to use for combining model predictions
        model_evaluation_window : int, optional
            Number of predictions to use for model performance evaluation
        """
        self.models = models
        self.capital = initial_capital
        self.risk_percentage = risk_percentage
        self.portfolio = {}
        
        # Initialize ensemble selector
        self.ensemble_selector = EnsembleModelSelector()
        self.ensemble_method = ensemble_method
        
        # Model performance tracking
        self.model_performances = np.ones(len(models))
        self.model_evaluation_window = model_evaluation_window
        self.prediction_history = [[] for _ in range(len(models))]
        self.actual_outcomes = []
    
    def generate_signals(self, features: np.ndarray) -> Dict[str, float]:
        """
        Generate trading signals using ensemble of models
        
        Parameters:
        -----------
        features : numpy.ndarray
            Input features for prediction
        
        Returns:
        --------
        Dict[str, float]
            Trading signals with confidence scores
        """
        predictions = []
        confidences = []
        
        for model in self.models:
            pred = model.predict(features)
            pred_proba = model.predict_proba(features)
            
            predictions.append(pred[0])
            confidences.append(pred_proba[0].max())
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        confidences = np.array(confidences)
        
        # Apply ensemble method
        ensemble_prediction, ensemble_confidence = self.ensemble_selector.apply_method(
            self.ensemble_method,
            predictions,
            confidences,
            model_performances=self.model_performances
        )
        
        # Record predictions for performance tracking
        self.record_predictions(predictions)
        
        return {
            'signal': 1 if ensemble_prediction > 0.5 else 0,
            'confidence': ensemble_confidence,
            'raw_prediction': ensemble_prediction,
            'model_predictions': predictions.tolist(),
            'model_confidences': confidences.tolist()
        }
    
    def update_model_performances(self, actual_outcome: int):
        """
        Update model performance metrics based on actual outcomes
        
        Parameters:
        -----------
        actual_outcome : int
            Actual outcome (1 for up, 0 for down)
        """
        self.actual_outcomes.append(actual_outcome)
        
        # Only update if we have enough history
        if len(self.actual_outcomes) <= 1:
            return
        
        # Update performance for each model
        for i, model in enumerate(self.models):
            # Get prediction history for this model
            if len(self.prediction_history[i]) > 0:
                # Calculate accuracy over the evaluation window
                window_size = min(self.model_evaluation_window, len(self.prediction_history[i]))
                recent_predictions = self.prediction_history[i][-window_size:]
                recent_outcomes = self.actual_outcomes[-window_size:]
                
                # Convert predictions to binary
                binary_predictions = [1 if p > 0.5 else 0 for p in recent_predictions]
                
                # Calculate accuracy
                correct_predictions = sum(1 for p, a in zip(binary_predictions, recent_outcomes) if p == a)
                accuracy = correct_predictions / window_size
                
                # Update model performance
                self.model_performances[i] = max(0.1, accuracy)  # Ensure minimum weight
    
    def record_predictions(self, predictions: List[float]):
        """
        Record model predictions for performance tracking
        
        Parameters:
        -----------
        predictions : List[float]
            List of predictions from each model
        """
        for i, pred in enumerate(predictions):
            if i < len(self.prediction_history):
                self.prediction_history[i].append(pred)
    
    def calculate_position_size(self, current_price: float, confidence: float = None) -> int:
        """
        Calculate position size based on risk management and prediction confidence
        
        Parameters:
        -----------
        current_price : float
            Current stock price
        confidence : float, optional
            Prediction confidence score
            
        Returns:
        --------
        int
            Number of shares to trade
        """
        # Risk amount per trade
        risk_amount = self.capital * self.risk_percentage
        
        # Adjust risk based on confidence if provided
        if confidence is not None:
            # Scale risk between 50% and 150% based on confidence
            confidence_factor = 0.5 + confidence
            risk_amount *= confidence_factor
        
        # Maximum position size
        max_position = self.capital * Settings.MAX_POSITION_SIZE
        
        # Calculate shares
        shares = int(min(risk_amount / current_price, max_position / current_price))
        
        return shares
    
    def set_ensemble_method(self, method_name: str):
        """
        Set the ensemble method to use
        
        Parameters:
        -----------
        method_name : str
            Name of the ensemble method
        """
        if method_name not in self.ensemble_selector.ensemble_methods:
            raise ValueError(f"Ensemble method '{method_name}' not found")
        
        self.ensemble_method = method_name
    
    def get_available_ensemble_methods(self) -> List[str]:
        """
        Get list of available ensemble methods
        
        Returns:
        --------
        List[str]
            List of available ensemble method names
        """
        return list(self.ensemble_selector.ensemble_methods.keys())
    
    def evaluate_ensemble_methods(self, features: np.ndarray, actual_outcomes: List[int]) -> Dict[str, float]:
        """
        Evaluate performance of different ensemble methods
        
        Parameters:
        -----------
        features : numpy.ndarray
            Input features for prediction
        actual_outcomes : List[int]
            Actual outcomes for evaluation
            
        Returns:
        --------
        Dict[str, float]
            Performance metrics for each ensemble method
        """
        results = {}
        
        for method_name in self.get_available_ensemble_methods():
            correct_predictions = 0
            
            for i, feature in enumerate(features):
                # Get predictions from all models
                predictions = []
                confidences = []
                
                for model in self.models:
                    pred = model.predict(feature.reshape(1, -1))
                    pred_proba = model.predict_proba(feature.reshape(1, -1))
                    
                    predictions.append(pred[0])
                    confidences.append(pred_proba[0].max())
                
                # Convert to numpy arrays
                predictions = np.array(predictions)
                confidences = np.array(confidences)
                
                # Apply ensemble method
                ensemble_prediction, _ = self.ensemble_selector.apply_method(
                    method_name,
                    predictions,
                    confidences,
                    model_performances=self.model_performances
                )
                
                # Convert to binary prediction
                binary_prediction = 1 if ensemble_prediction > 0.5 else 0
                
                # Check if prediction is correct
                if binary_prediction == actual_outcomes[i]:
                    correct_predictions += 1
            
            # Calculate accuracy
            accuracy = correct_predictions / len(actual_outcomes) if len(actual_outcomes) > 0 else 0
            results[method_name] = accuracy
        
        return results