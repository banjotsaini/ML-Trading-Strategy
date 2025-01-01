class MLTradingStrategy:
    def __init__(
        self, 
        models: List[BaseMLModel], 
        initial_capital: float = Settings.INITIAL_CAPITAL,
        risk_percentage: float = Settings.RISK_PERCENTAGE
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
        """
        self.models = models
        self.capital = initial_capital
        self.risk_percentage = risk_percentage
        self.portfolio = {}
    
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
        
        # Ensemble voting
        ensemble_prediction = np.mean(predictions)
        average_confidence = np.mean(confidences)
        
        return {
            'signal': 1 if ensemble_prediction > 0.5 else 0,
            'confidence': average_confidence
        }
    
    def calculate_position_size(self, current_price: float) -> int:
        """
        Calculate position size based on risk management
        
        Parameters:
        -----------
        current_price : float
            Current stock price
        
        Returns:
        --------
        int
            Number of shares to trade
        """
        # Risk amount per trade
        risk_amount = self.capital * self.risk_percentage
        
        # Maximum position size
        max_position = self.capital * Settings.MAX_POSITION_SIZE
        
        # Calculate shares
        shares = int(min(risk_amount / current_price, max_position / current_price))
        
        return shares