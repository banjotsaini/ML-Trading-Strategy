#!/bin/bash
# Script to run the live trading service

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check if API key is set
if [ -z "$ALPHA_VANTAGE_API_KEY" ]; then
    echo "Warning: ALPHA_VANTAGE_API_KEY environment variable is not set."
    echo "You can set it with: export ALPHA_VANTAGE_API_KEY=your_api_key_here"
    echo "Alternatively, you can specify it in the config file."
fi

# Check if config file exists
CONFIG_FILE="config/live_trading_config.json"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    echo "Please create the config file before running the service."
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the live trading service
echo "Starting live trading service..."
python scripts/live_trading.py --config "$CONFIG_FILE" "$@"

# Check exit status
if [ $? -ne 0 ]; then
    echo "Error: Live trading service exited with an error."
    exit 1
fi

echo "Live trading service completed successfully."