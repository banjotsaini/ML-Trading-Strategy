#!/usr/bin/env python
"""
Live Trading Script

This script runs the real-time trading system with live market data.
"""

import os
import sys
import time
import logging
import argparse
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import Settings
from data.real_time_feed import RealTimeDataFeed
from data.stream_processor import StreamProcessor
from models.random_forest import RandomForestModel
from models.gradient_boosting import GradientBoostingModel
from models.svm import SVMModel
from strategies.ml_trading_strategy import MLTradingStrategy
from trading.signal_service import SignalGeneratorService
from utils.market_scheduler import MarketScheduler
from utils.notifications import NotificationManager
from utils.resilience import ConnectionManager

class LiveTradingService:
    """Main service for real-time trading"""
    
    def __init__(
        self,
        config_file: Optional[str] = None,
        api_key: Optional[str] = None,
        provider: str = 'alphavantage',
        tickers: Optional[List[str]] = None,
        log_level: str = 'INFO'
    ):
        """
        Initialize live trading service
        
        Parameters:
        -----------
        config_file : str, optional
            Path to configuration file
        api_key : str, optional
            API key for data provider
        provider : str
            Data provider name
        tickers : List[str], optional
            List of tickers to trade
        log_level : str
            Logging level
        """
        # Setup logging
        self._setup_logging(log_level)
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Override config with command-line arguments
        if api_key:
            self.config['api_key'] = api_key
        if provider:
            self.config['provider'] = provider
        if tickers:
            self.config['tickers'] = tickers
        
        # Validate configuration
        self._validate_config()
        
        # Initialize components
        self.connection_manager = ConnectionManager()
        self.market_scheduler = MarketScheduler()
        
        # Initialize notification manager if configured
        self.notification_manager = None
        if 'notifications' in self.config and self.config['notifications'].get('enabled', False):
            self.notification_manager = self._setup_notifications()
        
        # Service state
        self.running = False
        self.data_feed = None
        self.stream_processor = None
        self.strategy = None
        self.signal_service = None
        
        # Service status
        self.status = {
            'state': 'initialized',
            'start_time': None,
            'components': {
                'data_feed': 'not_started',
                'stream_processor': 'not_started',
                'strategy': 'not_started',
                'signal_service': 'not_started'
            }
        }
    
    def _setup_logging(self, log_level: str):
        """
        Setup logging configuration
        
        Parameters:
        -----------
        log_level : str
            Logging level
        """
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Setup logging
        log_file = f"logs/live_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration from file
        
        Parameters:
        -----------
        config_file : str, optional
            Path to configuration file
            
        Returns:
        --------
        Dict[str, Any]
            Configuration dictionary
        """
        # Default configuration
        config = {
            'provider': 'alphavantage',
            'api_key': os.environ.get('MARKET_DATA_API_KEY', ''),
            'tickers': Settings.TICKERS,
            'update_interval': 60,  # seconds
            'signal_threshold': 0.7,
            'window_size': 100,
            'notifications': {
                'enabled': False
            }
        }
        
        # Load from file if provided
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                    config.update(file_config)
                self.logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                self.logger.error(f"Error loading configuration from {config_file}: {str(e)}")
        
        return config
    
    def _validate_config(self):
        """Validate configuration"""
        # Check required fields
        required_fields = ['provider', 'api_key', 'tickers']
        for field in required_fields:
            if not self.config.get(field):
                raise ValueError(f"Missing required configuration: {field}")
        
        # Validate API key
        if not self.config['api_key']:
            raise ValueError("API key is required")
        
        # Validate tickers
        if not self.config['tickers'] or not isinstance(self.config['tickers'], list):
            raise ValueError("Tickers must be a non-empty list")
    
    def _setup_notifications(self) -> NotificationManager:
        """
        Setup notification manager
        
        Returns:
        --------
        NotificationManager
            Configured notification manager
        """
        notification_config = self.config['notifications']
        
        # Email configuration
        email_config = None
        if notification_config.get('email', {}).get('enabled', False):
            email_config = {
                'smtp_server': notification_config['email'].get('smtp_server', ''),
                'smtp_port': notification_config['email'].get('smtp_port', 587),
                'username': notification_config['email'].get('username', ''),
                'password': notification_config['email'].get('password', ''),
                'from_email': notification_config['email'].get('from_email', '')
            }
        
        # SMS configuration
        sms_config = None
        if notification_config.get('sms', {}).get('enabled', False):
            sms_config = notification_config['sms']
        
        # Create notification manager
        return NotificationManager(
            email_config=email_config,
            sms_config=sms_config,
            throttle_period=notification_config.get('throttle_period', 300)
        )
    
    def start(self):
        """Start the live trading service"""
        if self.running:
            return
        
        self.logger.info("Starting live trading service")
        
        # Update status
        self.status['state'] = 'starting'
        self.status['start_time'] = datetime.now()
        
        try:
            # 1. Initialize data feed
            self.logger.info(f"Initializing data feed ({self.config['provider']})")
            self.data_feed = RealTimeDataFeed(
                provider_name=self.config['provider'],
                api_key=self.config['api_key']
            )
            self.status['components']['data_feed'] = 'initialized'
            
            # 2. Initialize stream processor
            self.logger.info("Initializing stream processor")
            self.stream_processor = StreamProcessor(
                window_size=self.config.get('window_size', 100)
            )
            self.status['components']['stream_processor'] = 'initialized'
            
            # 3. Initialize ML models
            self.logger.info("Initializing ML models")
            models = self._initialize_models()
            
            # 4. Initialize trading strategy
            self.logger.info("Initializing trading strategy")
            self.strategy = MLTradingStrategy(
                models=models,
                initial_capital=Settings.INITIAL_CAPITAL,
                risk_percentage=Settings.RISK_PERCENTAGE,
                ensemble_method=self.config.get('ensemble_method', 'dynamic_ensemble')
            )
            self.status['components']['strategy'] = 'initialized'
            
            # 5. Initialize signal generator service
            self.logger.info("Initializing signal generator service")
            self.signal_service = SignalGeneratorService(
                data_feed=self.data_feed,
                stream_processor=self.stream_processor,
                strategy=self.strategy,
                tickers=self.config['tickers'],
                update_interval=self.config.get('update_interval', 60),
                signal_threshold=self.config.get('signal_threshold', 0.7),
                notification_manager=self.notification_manager,
                notification_recipients=self.config.get('notification_recipients', [])
            )
            self.status['components']['signal_service'] = 'initialized'
            
            # 6. Start signal generator service
            self.logger.info("Starting signal generator service")
            self.signal_service.start()
            self.status['components']['signal_service'] = 'running'
            
            # 7. Register signal callback
            self.signal_service.register_signal_callback(self._handle_signal)
            
            # Update status
            self.running = True
            self.status['state'] = 'running'
            
            # Send notification
            if self.notification_manager and self.config.get('notification_recipients'):
                self.notification_manager.send_status_update(
                    "Live trading service started",
                    {
                        'tickers': self.config['tickers'],
                        'provider': self.config['provider']
                    },
                    self.config['notification_recipients'],
                    'normal'
                )
            
            self.logger.info("Live trading service started")
            
            # Keep service running
            try:
                while self.running:
                    time.sleep(1)
            except KeyboardInterrupt:
                self.logger.info("Keyboard interrupt received")
                self.stop()
            
        except Exception as e:
            self.status['state'] = 'error'
            self.logger.error(f"Failed to start live trading service: {str(e)}")
            
            # Send notification
            if self.notification_manager and self.config.get('notification_recipients'):
                self.notification_manager.send_error_alert(
                    "Failed to start live trading service",
                    str(e),
                    self.config['notification_recipients'],
                    'high'
                )
            
            raise
    
    def _initialize_models(self) -> List[Any]:
        """
        Initialize ML models
        
        Returns:
        --------
        List[Any]
            List of initialized ML models
        """
        models = []
        
        # Random Forest model
        self.logger.info("Initializing Random Forest model")
        rf_model = RandomForestModel()
        
        # Gradient Boosting model
        self.logger.info("Initializing Gradient Boosting model")
        gb_model = GradientBoostingModel()
        
        # SVM model
        self.logger.info("Initializing SVM model")
        svm_model = SVMModel()
        
        # Add models to list
        models.extend([rf_model, gb_model, svm_model])
        
        # Load model weights if available
        model_dir = os.path.join(Settings.MODELS_DIR, 'trained')
        if os.path.exists(model_dir):
            self.logger.info(f"Loading model weights from {model_dir}")
            for model in models:
                model_file = os.path.join(model_dir, f"{model.__class__.__name__}.pkl")
                if os.path.exists(model_file):
                    try:
                        model.load(model_file)
                        self.logger.info(f"Loaded weights for {model.__class__.__name__}")
                    except Exception as e:
                        self.logger.error(f"Error loading weights for {model.__class__.__name__}: {str(e)}")
        else:
            self.logger.warning(f"Model directory {model_dir} not found. Using untrained models.")
        
        return models
    
    def _handle_signal(self, signal: Dict[str, Any]):
        """
        Handle new trading signal
        
        Parameters:
        -----------
        signal : Dict[str, Any]
            Trading signal
        """
        # This is where you would implement order execution
        # For now, we just log the signal
        self.logger.info(f"Received signal: {signal['ticker']} - {signal['signal']} "
                        f"(confidence: {signal['confidence']:.4f}, price: ${signal['price']:.2f})")
    
    def stop(self):
        """Stop the live trading service"""
        if not self.running:
            return
        
        self.logger.info("Stopping live trading service")
        
        # Update status
        self.status['state'] = 'stopping'
        
        # Stop signal generator service
        if self.signal_service:
            try:
                self.signal_service.stop()
                self.status['components']['signal_service'] = 'stopped'
            except Exception as e:
                self.logger.error(f"Error stopping signal generator service: {str(e)}")
        
        # Update status
        self.running = False
        self.status['state'] = 'stopped'
        
        # Send notification
        if self.notification_manager and self.config.get('notification_recipients'):
            self.notification_manager.send_status_update(
                "Live trading service stopped",
                {
                    'run_time': str(datetime.now() - self.status['start_time']) if self.status['start_time'] else 'N/A'
                },
                self.config['notification_recipients'],
                'normal'
            )
        
        self.logger.info("Live trading service stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get service status
        
        Returns:
        --------
        Dict[str, Any]
            Service status
        """
        # Update with signal service status if available
        if self.signal_service:
            self.status['signal_service'] = self.signal_service.get_status()
        
        return self.status
    
    def pause(self):
        """Pause trading temporarily"""
        if not self.running or not self.signal_service:
            return
        
        self.logger.info("Pausing live trading")
        
        # Update status
        self.status['state'] = 'paused'
        
        # Stop signal generator service
        self.signal_service.stop()
        self.status['components']['signal_service'] = 'stopped'
        
        self.logger.info("Live trading paused")
    
    def resume(self):
        """Resume trading after pause"""
        if self.running and self.status['state'] == 'paused' and self.signal_service:
            self.logger.info("Resuming live trading")
            
            # Start signal generator service
            self.signal_service.start()
            self.status['components']['signal_service'] = 'running'
            
            # Update status
            self.status['state'] = 'running'
            
            self.logger.info("Live trading resumed")

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Live Trading Service')
    
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--api-key', type=str, help='API key for data provider')
    parser.add_argument('--provider', type=str, default='alphavantage', help='Data provider name')
    parser.add_argument('--tickers', type=str, help='Comma-separated list of tickers')
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level')
    
    return parser.parse_args()

def main():
    """Main entry point"""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Convert tickers string to list if provided
    tickers = None
    if args.tickers:
        tickers = [ticker.strip() for ticker in args.tickers.split(',')]
    
    # Create and start live trading service
    service = LiveTradingService(
        config_file=args.config,
        api_key=args.api_key,
        provider=args.provider,
        tickers=tickers,
        log_level=args.log_level
    )
    
    try:
        service.start()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Stopping service...")
        service.stop()
    except Exception as e:
        print(f"Error: {str(e)}")
        service.stop()
        sys.exit(1)

if __name__ == "__main__":
    main()