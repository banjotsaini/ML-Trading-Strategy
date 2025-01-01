import os
from pathlib import Path

class Settings:
    # Project Paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / 'data'
    MODELS_DIR = BASE_DIR / 'models'

    # Trading Parameters
    INITIAL_CAPITAL = 2500
    RISK_PERCENTAGE = 0.02
    MAX_POSITION_SIZE = 0.15  # 15% of portfolio per position

    # Tickers to Trade
    TICKERS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']

    # Model Training Parameters
    TRAIN_TEST_SPLIT = 0.2
    RANDOM_SEED = 42
    