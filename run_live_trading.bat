@echo off
REM Script to run the live trading service

REM Check if virtual environment exists and activate it
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Check if config file exists
set CONFIG_FILE=config\live_trading_config.json
if not exist %CONFIG_FILE% (
    echo Error: Config file not found: %CONFIG_FILE%
    echo Please create the config file before running the service.
    exit /b 1
)

REM Create logs directory if it doesn't exist
if not exist logs mkdir logs

REM Run the live trading service
echo Starting live trading service...
python scripts\live_trading.py --config %CONFIG_FILE% %*

REM Check exit status
if %ERRORLEVEL% neq 0 (
    echo Error: Live trading service exited with an error.
    exit /b 1
)

echo Live trading service completed successfully.
pause