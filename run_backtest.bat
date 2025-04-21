@echo off
REM Script to run the backtesting framework

REM Check if virtual environment exists and activate it
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Check if config file exists
set CONFIG_FILE=config\backtest_config.json
if not "%1"=="" (
    set CONFIG_FILE=%1
)

if not exist %CONFIG_FILE% (
    echo Error: Config file not found: %CONFIG_FILE%
    echo Please create the config file before running the backtest.
    exit /b 1
)

REM Create logs directory if it doesn't exist
if not exist logs mkdir logs

REM Run the backtest
echo Starting backtest...
python scripts\backtest.py --config %CONFIG_FILE% %2 %3 %4 %5 %6 %7 %8 %9

REM Check exit status
if %ERRORLEVEL% neq 0 (
    echo Error: Backtest exited with an error.
    exit /b 1
)

echo Backtest completed successfully.
pause