#!/bin/bash
# Script to run the backtesting framework

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check if config file exists
CONFIG_FILE="config/backtest_config.json"
if [ "$1" != "" ]; then
    CONFIG_FILE="$1"
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    echo "Please create the config file before running the backtest."
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the backtest
echo "Starting backtest..."
python scripts/backtest.py --config "$CONFIG_FILE" "${@:2}"

# Check exit status
if [ $? -ne 0 ]; then
    echo "Error: Backtest exited with an error."
    exit 1
fi

echo "Backtest completed successfully."