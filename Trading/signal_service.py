import time
import threading
import logging
from typing import Dict, List, Any, Callable, Optional
from datetime import datetime

from data.real_time_feed import RealTimeDataFeed
from data.stream_processor import StreamProcessor
from strategies.ml_trading_strategy import MLTradingStrategy
from utils.market_scheduler import MarketScheduler
from utils.notifications import NotificationManager
from utils.resilience import with_retry, CircuitBreaker

class SignalGeneratorService:
    """Service for continuous signal generation"""
    
    def __init__(
        self,
        data_feed: RealTimeDataFeed,
        stream_processor: StreamProcessor,
        strategy: MLTradingStrategy,
        tickers: List[str],
        update_interval: int = 60,  # seconds
        signal_threshold: float = 0.7,  # minimum confidence for signals
        notification_manager: Optional[NotificationManager] = None,
        notification_recipients: Optional[List[str]] = None
    ):
        """
        Initialize signal generator service
        
        Parameters:
        -----------
        data_feed : RealTimeDataFeed
            Real-time market data feed
        stream_processor : StreamProcessor
            Streaming data processor
        strategy : MLTradingStrategy
            Trading strategy
        tickers : List[str]
            List of tickers to monitor
        update_interval : int
            Interval between signal updates (seconds)
        signal_threshold : float
            Minimum confidence threshold for signals
        notification_manager : NotificationManager, optional
            Notification manager for alerts
        notification_recipients : List[str], optional
            List of notification recipients
        """
        self.data_feed = data_feed
        self.stream_processor = stream_processor
        self.strategy = strategy
        self.tickers = tickers
        self.update_interval = update_interval
        self.signal_threshold = signal_threshold
        self.notification_manager = notification_manager
        self.notification_recipients = notification_recipients
        
        self.running = False
        self.thread = None
        self.signal_callbacks = []
        self.market_scheduler = MarketScheduler()
        self.logger = logging.getLogger(__name__)
        
        # Signal history
        self.signal_history = {}  # ticker -> list of signals
        
        # Service status
        self.status = {
            'state': 'initialized',
            'start_time': None,
            'last_update': None,
            'signals_generated': 0,
            'errors': 0
        }
        
    def start(self):
        """Start the signal generation service"""
        if self.running:
            return
        
        self.logger.info("Starting signal generator service")
        
        # Update status
        self.status['state'] = 'starting'
        self.status['start_time'] = datetime.now()
        
        try:
            # Connect to data feed
            self.data_feed.connect()
            
            # Subscribe to tickers
            self.data_feed.subscribe(self.tickers)
            
            # Register callback for new data
            try:
                self.data_feed.register_callback(self.process_new_data)
                self.logger.info("Registered callback for real-time data")
            except NotImplementedError:
                # If callbacks not supported, we'll poll instead
                self.logger.info("Callbacks not supported, using polling")
            
            # Start service thread
            self.running = True
            self.thread = threading.Thread(target=self._run_service)
            self.thread.daemon = True
            self.thread.start()
            
            # Update status
            self.status['state'] = 'running'
            
            # Send notification
            if self.notification_manager and self.notification_recipients:
                self.notification_manager.send_status_update(
                    "Signal generator service started",
                    {
                        'tickers': self.tickers,
                        'update_interval': self.update_interval,
                        'signal_threshold': self.signal_threshold
                    },
                    self.notification_recipients,
                    'normal'
                )
            
            self.logger.info("Signal generator service started")
            
        except Exception as e:
            self.status['state'] = 'error'
            self.status['errors'] += 1
            self.logger.error(f"Failed to start signal generator service: {str(e)}")
            
            # Send notification
            if self.notification_manager and self.notification_recipients:
                self.notification_manager.send_error_alert(
                    "Failed to start signal generator service",
                    str(e),
                    self.notification_recipients,
                    'high'
                )
            
            raise
        
    def _run_service(self):
        """Main service loop"""
        while self.running:
            try:
                # Check if market is open
                if not self.market_scheduler.is_market_open():
                    next_open = self.market_scheduler.get_next_market_open()
                    self.logger.info(f"Market closed. Next open: {next_open}")
                    
                    # Update status
                    self.status['state'] = 'waiting_for_market_open'
                    
                    # Sleep until next check (every 5 minutes when market is closed)
                    time.sleep(300)
                    continue
                
                # Update status
                if self.status['state'] != 'running':
                    self.status['state'] = 'running'
                
                # For polling-based providers, fetch latest data
                if not hasattr(self.data_feed.provider, 'register_callback'):
                    for ticker in self.tickers:
                        try:
                            latest_data = self.data_feed.get_latest_data(ticker)
                            self.process_new_data(latest_data)
                        except Exception as e:
                            self.logger.error(f"Error fetching data for {ticker}: {str(e)}")
                            self.status['errors'] += 1
                
                # Generate signals for all tickers
                self.generate_signals()
                
                # Update status
                self.status['last_update'] = datetime.now()
                
            except Exception as e:
                self.logger.error(f"Error in signal generator service: {str(e)}")
                self.status['errors'] += 1
                
                # Send notification for critical errors
                if self.notification_manager and self.notification_recipients:
                    self.notification_manager.send_error_alert(
                        "Error in signal generator service",
                        str(e),
                        self.notification_recipients,
                        'high'
                    )
            
            # Sleep until next update
            time.sleep(self.update_interval)
    
    @with_retry(max_retries=3, retry_delay=1, backoff_factor=2.0)
    def process_new_data(self, tick_data: Dict[str, Any]):
        """
        Process new market data
        
        Parameters:
        -----------
        tick_data : Dict[str, Any]
            Tick data from the data feed
        """
        try:
            # Process tick data
            self.stream_processor.process_tick_data(tick_data)
            
            # Log data received
            ticker = tick_data['ticker']
            price = tick_data['price']
            self.logger.debug(f"Received data for {ticker}: ${price:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error processing tick data: {str(e)}")
            self.status['errors'] += 1
            raise
    
    def generate_signals(self):
        """Generate trading signals for all tickers"""
        for ticker in self.tickers:
            try:
                # Get feature vector for the ticker
                feature_vector = self.stream_processor.get_feature_vector(ticker)
                if feature_vector is None:
                    continue
                
                # Generate signal
                signal_data = self.strategy.generate_signals(feature_vector)
                
                # Check if signal meets threshold
                if signal_data['confidence'] >= self.signal_threshold:
                    # Get current price
                    latest_data = self.data_feed.get_latest_data(ticker)
                    current_price = latest_data['price']
                    
                    # Calculate position size
                    position_size = self.strategy.calculate_position_size(
                        current_price, 
                        confidence=signal_data['confidence']
                    )
                    
                    # Create signal with additional information
                    signal = {
                        'ticker': ticker,
                        'signal': 'BUY' if signal_data['signal'] == 1 else 'SELL',
                        'confidence': signal_data['confidence'],
                        'price': current_price,
                        'position_size': position_size,
                        'timestamp': datetime.now().isoformat(),
                        'model_predictions': signal_data['model_predictions'],
                        'model_confidences': signal_data['model_confidences']
                    }
                    
                    # Store signal in history
                    if ticker not in self.signal_history:
                        self.signal_history[ticker] = []
                    self.signal_history[ticker].append(signal)
                    
                    # Limit history size
                    if len(self.signal_history[ticker]) > 100:
                        self.signal_history[ticker] = self.signal_history[ticker][-100:]
                    
                    # Update status
                    self.status['signals_generated'] += 1
                    
                    # Log signal
                    self.logger.info(f"Generated signal: {ticker} - {signal['signal']} "
                                    f"(confidence: {signal['confidence']:.4f})")
                    
                    # Send notification
                    if self.notification_manager and self.notification_recipients:
                        self.notification_manager.send_signal_alert(
                            ticker,
                            signal['signal'],
                            signal['confidence'],
                            current_price,
                            self.notification_recipients,
                            'normal' if signal['confidence'] < 0.9 else 'high'
                        )
                    
                    # Call signal callbacks
                    for callback in self.signal_callbacks:
                        callback(signal)
            
            except Exception as e:
                self.logger.error(f"Error generating signal for {ticker}: {str(e)}")
                self.status['errors'] += 1
    
    def register_signal_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Register callback for new signals
        
        Parameters:
        -----------
        callback : Callable
            Callback function that takes a signal dictionary
        """
        self.signal_callbacks.append(callback)
    
    def get_signal_history(self, ticker: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get signal history
        
        Parameters:
        -----------
        ticker : str, optional
            Ticker to get history for, or None for all tickers
            
        Returns:
        --------
        Dict[str, List[Dict[str, Any]]]
            Signal history
        """
        if ticker:
            return {ticker: self.signal_history.get(ticker, [])}
        return self.signal_history
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get service status
        
        Returns:
        --------
        Dict[str, Any]
            Service status
        """
        return self.status
    
    def stop(self):
        """Stop the signal generation service"""
        if not self.running:
            return
        
        self.logger.info("Stopping signal generator service")
        
        # Update status
        self.status['state'] = 'stopping'
        
        # Stop service thread
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        
        # Disconnect from data feed
        try:
            self.data_feed.disconnect()
        except Exception as e:
            self.logger.error(f"Error disconnecting from data feed: {str(e)}")
            self.status['errors'] += 1
        
        # Update status
        self.status['state'] = 'stopped'
        
        # Send notification
        if self.notification_manager and self.notification_recipients:
            self.notification_manager.send_status_update(
                "Signal generator service stopped",
                {
                    'signals_generated': self.status['signals_generated'],
                    'errors': self.status['errors'],
                    'run_time': str(datetime.now() - self.status['start_time']) if self.status['start_time'] else 'N/A'
                },
                self.notification_recipients,
                'normal'
            )
        
        self.logger.info("Signal generator service stopped")