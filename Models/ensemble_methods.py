import numpy as np
from typing import List, Dict, Tuple, Callable, Optional
from models.base_model import BaseMLModel

class EnsembleMethod:
    """Base class for ensemble methods"""
    
    @staticmethod
    def simple_average(predictions: np.ndarray, confidences: np.ndarray) -> Tuple[float, float]:
        """
        Simple averaging ensemble method
        
        Parameters:
        -----------
        predictions : np.ndarray
            Array of predictions from different models
        confidences : np.ndarray
            Array of confidence scores from different models
            
        Returns:
        --------
        Tuple[float, float]
            Ensemble prediction and confidence
        """
        ensemble_prediction = np.mean(predictions)
        average_confidence = np.mean(confidences)
        return ensemble_prediction, average_confidence
    
    @staticmethod
    def weighted_average(
        predictions: np.ndarray, 
        confidences: np.ndarray, 
        weights: np.ndarray
    ) -> Tuple[float, float]:
        """
        Weighted averaging ensemble method
        
        Parameters:
        -----------
        predictions : np.ndarray
            Array of predictions from different models
        confidences : np.ndarray
            Array of confidence scores from different models
        weights : np.ndarray
            Array of weights for each model
            
        Returns:
        --------
        Tuple[float, float]
            Ensemble prediction and confidence
        """
        # Normalize weights
        weights = weights / np.sum(weights)
        
        # Calculate weighted predictions and confidences
        weighted_prediction = np.sum(predictions * weights)
        weighted_confidence = np.sum(confidences * weights)
        
        return weighted_prediction, weighted_confidence
    
    @staticmethod
    def majority_vote(predictions: np.ndarray, confidences: np.ndarray) -> Tuple[float, float]:
        """
        Majority voting ensemble method
        
        Parameters:
        -----------
        predictions : np.ndarray
            Array of predictions from different models
        confidences : np.ndarray
            Array of confidence scores from different models
            
        Returns:
        --------
        Tuple[float, float]
            Ensemble prediction and confidence
        """
        # Convert predictions to binary (0 or 1)
        binary_predictions = (predictions > 0.5).astype(int)
        
        # Count votes
        vote_count = np.sum(binary_predictions)
        total_votes = len(binary_predictions)
        
        # Determine majority
        if vote_count > total_votes / 2:
            ensemble_prediction = 1.0
        else:
            ensemble_prediction = 0.0
        
        # Calculate confidence based on vote ratio
        vote_ratio = max(vote_count, total_votes - vote_count) / total_votes
        
        # Average confidence of winning vote
        winning_indices = np.where(binary_predictions == ensemble_prediction)[0]
        if len(winning_indices) > 0:
            winning_confidence = np.mean(confidences[winning_indices])
        else:
            winning_confidence = 0.5
        
        # Final confidence is a combination of vote ratio and winning models' confidence
        ensemble_confidence = (vote_ratio + winning_confidence) / 2
        
        return ensemble_prediction, ensemble_confidence
    
    @staticmethod
    def weighted_vote(
        predictions: np.ndarray, 
        confidences: np.ndarray, 
        weights: np.ndarray
    ) -> Tuple[float, float]:
        """
        Weighted voting ensemble method
        
        Parameters:
        -----------
        predictions : np.ndarray
            Array of predictions from different models
        confidences : np.ndarray
            Array of confidence scores from different models
        weights : np.ndarray
            Array of weights for each model
            
        Returns:
        --------
        Tuple[float, float]
            Ensemble prediction and confidence
        """
        # Normalize weights
        weights = weights / np.sum(weights)
        
        # Convert predictions to binary (0 or 1)
        binary_predictions = (predictions > 0.5).astype(int)
        
        # Calculate weighted votes
        weighted_votes_for_one = np.sum(weights[binary_predictions == 1])
        weighted_votes_for_zero = np.sum(weights[binary_predictions == 0])
        
        # Determine winner
        if weighted_votes_for_one > weighted_votes_for_zero:
            ensemble_prediction = 1.0
            winning_vote_weight = weighted_votes_for_one
            winning_indices = np.where(binary_predictions == 1)[0]
        else:
            ensemble_prediction = 0.0
            winning_vote_weight = weighted_votes_for_zero
            winning_indices = np.where(binary_predictions == 0)[0]
        
        # Calculate confidence based on vote weight ratio
        vote_weight_ratio = winning_vote_weight / (weighted_votes_for_one + weighted_votes_for_zero)
        
        # Average confidence of winning vote, weighted by model weights
        if len(winning_indices) > 0:
            winning_weights = weights[winning_indices]
            winning_confidences = confidences[winning_indices]
            winning_confidence = np.sum(winning_confidences * winning_weights) / np.sum(winning_weights)
        else:
            winning_confidence = 0.5
        
        # Final confidence is a combination of vote ratio and winning models' confidence
        ensemble_confidence = (vote_weight_ratio + winning_confidence) / 2
        
        return ensemble_prediction, ensemble_confidence
    
    @staticmethod
    def confidence_weighted(predictions: np.ndarray, confidences: np.ndarray) -> Tuple[float, float]:
        """
        Confidence-weighted ensemble method
        
        Parameters:
        -----------
        predictions : np.ndarray
            Array of predictions from different models
        confidences : np.ndarray
            Array of confidence scores from different models
            
        Returns:
        --------
        Tuple[float, float]
            Ensemble prediction and confidence
        """
        # Use confidences as weights
        weights = confidences / np.sum(confidences)
        
        # Calculate weighted prediction
        weighted_prediction = np.sum(predictions * weights)
        
        # Calculate confidence as weighted average of confidences
        ensemble_confidence = np.sum(confidences * weights)
        
        return weighted_prediction, ensemble_confidence
    
    @staticmethod
    def dynamic_ensemble(
        predictions: np.ndarray, 
        confidences: np.ndarray, 
        model_performances: np.ndarray
    ) -> Tuple[float, float]:
        """
        Dynamic ensemble method that adapts based on model performance
        
        Parameters:
        -----------
        predictions : np.ndarray
            Array of predictions from different models
        confidences : np.ndarray
            Array of confidence scores from different models
        model_performances : np.ndarray
            Array of performance metrics for each model
            
        Returns:
        --------
        Tuple[float, float]
            Ensemble prediction and confidence
        """
        # Use model performances as weights
        weights = model_performances / np.sum(model_performances)
        
        # Calculate weighted prediction
        weighted_prediction = np.sum(predictions * weights)
        
        # Calculate confidence as weighted average of confidences
        ensemble_confidence = np.sum(confidences * weights)
        
        return weighted_prediction, ensemble_confidence


class EnsembleModelSelector:
    """Class to select and apply ensemble methods"""
    
    def __init__(self):
        self.ensemble_methods = {
            'simple_average': EnsembleMethod.simple_average,
            'weighted_average': EnsembleMethod.weighted_average,
            'majority_vote': EnsembleMethod.majority_vote,
            'weighted_vote': EnsembleMethod.weighted_vote,
            'confidence_weighted': EnsembleMethod.confidence_weighted,
            'dynamic_ensemble': EnsembleMethod.dynamic_ensemble
        }
    
    def get_method(self, method_name: str) -> Callable:
        """
        Get ensemble method by name
        
        Parameters:
        -----------
        method_name : str
            Name of the ensemble method
            
        Returns:
        --------
        Callable
            Ensemble method function
        """
        if method_name not in self.ensemble_methods:
            raise ValueError(f"Ensemble method '{method_name}' not found")
        
        return self.ensemble_methods[method_name]
    
    def apply_method(
        self, 
        method_name: str, 
        predictions: np.ndarray, 
        confidences: np.ndarray, 
        weights: Optional[np.ndarray] = None,
        model_performances: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        """
        Apply ensemble method to predictions
        
        Parameters:
        -----------
        method_name : str
            Name of the ensemble method
        predictions : np.ndarray
            Array of predictions from different models
        confidences : np.ndarray
            Array of confidence scores from different models
        weights : np.ndarray, optional
            Array of weights for each model
        model_performances : np.ndarray, optional
            Array of performance metrics for each model
            
        Returns:
        --------
        Tuple[float, float]
            Ensemble prediction and confidence
        """
        method = self.get_method(method_name)
        
        if method_name in ['weighted_average', 'weighted_vote']:
            if weights is None:
                weights = np.ones(len(predictions))
            return method(predictions, confidences, weights)
        elif method_name == 'dynamic_ensemble':
            if model_performances is None:
                model_performances = np.ones(len(predictions))
            return method(predictions, confidences, model_performances)
        else:
            return method(predictions, confidences)