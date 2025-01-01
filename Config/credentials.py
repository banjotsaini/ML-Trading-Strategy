class Credentials:
    # Secure storage of API keys
    BROKER_API_KEY = os.environ.get('BROKER_API_KEY')
    MARKET_DATA_API_KEY = os.environ.get('MARKET_DATA_API_KEY')
    