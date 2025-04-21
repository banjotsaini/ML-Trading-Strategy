import logging
import numpy as np
from config.settings import Settings
from data.data_loader import DataLoader
from data.feature_store import FeatureStore
from models.random_forest import RandomForestModel
from models.gradient_boosting import GradientBoostingModel
from models.svm import SVMModel
from strategies.ml_trading_strategy import MLTradingStrategy

def setup_logging():
    """Configure logging for the application"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # 1. Load Historical Data
        logger.info("Fetching historical market data...")
        historical_data = DataLoader.fetch_historical_data(
            Settings.TICKERS
        )
        
        # 2. Prepare ML Models
        logger.info("Training multiple ML models for ensemble...")
        ticker_models = {}
        
        for ticker in Settings.TICKERS:
            # Prepare features for each ticker
            data = historical_data[ticker]
            technical_data = FeatureStore.calculate_technical_indicators(data)
            X, y = FeatureStore.prepare_ml_features(technical_data)
            
            # Train multiple models for each ticker
            models = []
            
            # Random Forest model
            logger.info(f"Training Random Forest model for {ticker}...")
            rf_model = RandomForestModel()
            rf_model.train(X, y)
            models.append(rf_model)
            
            # Gradient Boosting model
            logger.info(f"Training Gradient Boosting model for {ticker}...")
            gb_model = GradientBoostingModel()
            gb_model.train(X, y)
            models.append(gb_model)
            
            # SVM model
            logger.info(f"Training SVM model for {ticker}...")
            svm_model = SVMModel()
            svm_model.train(X, y)
            models.append(svm_model)
            
            ticker_models[ticker] = models
        
        # 3. Evaluate Ensemble Methods
        logger.info("Evaluating different ensemble methods...")
        # Use a small validation set for evaluation
        validation_features = []
        validation_outcomes = []
        
        for ticker, data in historical_data.items():
            # Use last 30 days for validation
            technical_data = FeatureStore.calculate_technical_indicators(data)
            X, y = FeatureStore.prepare_ml_features(technical_data)
            
            if len(X) > 30:
                validation_features.extend(X[-30:])
                validation_outcomes.extend(y[-30:])
        
        # Initialize strategy with first ticker's models for evaluation
        first_ticker = Settings.TICKERS[0]
        temp_strategy = MLTradingStrategy(ticker_models[first_ticker])
        
        # Evaluate ensemble methods
        if validation_features and validation_outcomes:
            ensemble_results = temp_strategy.evaluate_ensemble_methods(
                np.array(validation_features), 
                validation_outcomes
            )
            
            # Log results
            logger.info("Ensemble Method Evaluation Results:")
            for method, accuracy in ensemble_results.items():
                logger.info(f"  {method}: {accuracy:.4f}")
            
            # Find best method
            best_method = max(ensemble_results.items(), key=lambda x: x[1])[0]
            logger.info(f"Best ensemble method: {best_method}")
        else:
            best_method = 'dynamic_ensemble'
            logger.info(f"Using default ensemble method: {best_method}")
        
        # 4. Generate Trading Signals for each ticker
        logger.info("Generating trading signals...")
        
        for ticker, models in ticker_models.items():
            # Initialize strategy with best ensemble method
            strategy = MLTradingStrategy(models, ensemble_method=best_method)
            
            # Get latest data
            data = historical_data[ticker]
            
            # Use latest features
            latest_features = FeatureStore.prepare_ml_features(
                FeatureStore.calculate_technical_indicators(data)
            )[0][-1].reshape(1, -1)
            
            # Generate trading signal
            signal_data = strategy.generate_signals(latest_features)
            
            # Get current price
            current_price = data['Close'][-1]
            
            # Calculate position size using confidence
            position_size = strategy.calculate_position_size(
                current_price, 
                confidence=signal_data['confidence']
            )
            
            # Log trading information
            logger.info(f"\nTicker: {ticker}")
            logger.info(f"Signal: {'BUY' if signal_data['signal'] == 1 else 'SELL/HOLD'}")
            logger.info(f"Confidence: {signal_data['confidence']:.4f}")
            logger.info(f"Position Size: {position_size} shares")
            logger.info(f"Current Price: ${current_price:.2f}")
            
            # Log individual model predictions
            logger.info("Individual Model Predictions:")
            model_types = ["Random Forest", "Gradient Boosting", "SVM"]
            for i, (pred, conf) in enumerate(zip(signal_data['model_predictions'], signal_data['model_confidences'])):
                if i < len(model_types):
                    logger.info(f"  {model_types[i]}: {pred:.4f} (confidence: {conf:.4f})")
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()