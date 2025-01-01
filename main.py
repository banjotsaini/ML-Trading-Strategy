import logging
from config.settings import Settings
from data.data_loader import DataLoader
from data.feature_store import FeatureStore
from models.random_forest import RandomForestModel
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
        models = []
        for ticker in Settings.TICKERS:
            # Prepare features for each ticker
            data = historical_data[ticker]
            technical_data = FeatureStore.calculate_technical_indicators(data)
            X, y = FeatureStore.prepare_ml_features(technical_data)
            
            # Train Random Forest model
            rf_model = RandomForestModel()
            rf_model.train(X, y)
            models.append(rf_model)
        
        # 3. Initialize Trading Strategy
        strategy = MLTradingStrategy(models)
        
        # 4. Generate Trading Signals
        for ticker, data in historical_data.items():
            # Use latest features
            latest_features = FeatureStore.prepare_ml_features(
                FeatureStore.calculate_technical_indicators(data)
            )[0][-1].reshape(1, -1)
            
            # Generate trading signal
            signal = strategy.generate_signals(latest_features)
            
            # Get current price
            current_price = data['Close'][-1]
            
            # Calculate position size
            position_size = strategy.calculate_position_size(current_price)
            
            # Log trading information
            logger.info(f"Ticker: {ticker}")
            logger.info(f"Signal: {signal}")
            logger.info(f"Position Size: {position_size} shares")
            logger.info(f"Current Price: ${current_price:.2f}")
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()