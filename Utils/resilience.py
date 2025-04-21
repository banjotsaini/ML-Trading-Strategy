import time
import logging
import functools
import random
from typing import Callable, Any, Optional, Dict, List, Union
from datetime import datetime, timedelta

class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open"""
    pass

class CircuitBreaker:
    """
    Circuit breaker pattern implementation
    
    The circuit breaker prevents cascading failures by stopping
    operations when a service is failing repeatedly.
    """
    
    # Circuit breaker states
    CLOSED = 'closed'  # Normal operation
    OPEN = 'open'      # Circuit is open, calls fail fast
    HALF_OPEN = 'half_open'  # Testing if service is back
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exceptions: tuple = (Exception,)
    ):
        """
        Initialize circuit breaker
        
        Parameters:
        -----------
        failure_threshold : int
            Number of consecutive failures before opening circuit
        recovery_timeout : int
            Time in seconds to wait before trying recovery
        expected_exceptions : tuple
            Exceptions that count as failures
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exceptions = expected_exceptions
        
        self.state = self.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, func):
        """
        Decorator for circuit breaker
        
        Parameters:
        -----------
        func : Callable
            Function to protect with circuit breaker
            
        Returns:
        --------
        Callable
            Wrapped function
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self._call(func, *args, **kwargs)
        return wrapper
    
    def _call(self, func, *args, **kwargs):
        """
        Call the function with circuit breaker protection
        
        Parameters:
        -----------
        func : Callable
            Function to call
        *args, **kwargs
            Arguments to pass to the function
            
        Returns:
        --------
        Any
            Function result
            
        Raises:
        -------
        CircuitBreakerError
            If circuit is open
        """
        if self.state == self.OPEN:
            # Check if recovery timeout has elapsed
            if self.last_failure_time and datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout):
                self.logger.info(f"Circuit half-open for {func.__name__}")
                self.state = self.HALF_OPEN
            else:
                raise CircuitBreakerError(f"Circuit breaker open for {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            
            # If successful and in half-open state, close the circuit
            if self.state == self.HALF_OPEN:
                self.logger.info(f"Circuit closed for {func.__name__}")
                self.state = self.CLOSED
                self.failure_count = 0
            
            return result
            
        except self.expected_exceptions as e:
            # Record the failure
            self.last_failure_time = datetime.now()
            
            # If in half-open state, open the circuit again
            if self.state == self.HALF_OPEN:
                self.logger.warning(f"Circuit opened again for {func.__name__}: {str(e)}")
                self.state = self.OPEN
                raise CircuitBreakerError(f"Circuit breaker re-opened for {func.__name__}")
            
            # If in closed state, increment failure count
            self.failure_count += 1
            
            # If failure threshold reached, open the circuit
            if self.state == self.CLOSED and self.failure_count >= self.failure_threshold:
                self.logger.warning(f"Circuit opened for {func.__name__} after {self.failure_count} failures")
                self.state = self.OPEN
                raise CircuitBreakerError(f"Circuit breaker opened for {func.__name__}")
            
            # Re-raise the original exception
            raise

def with_retry(
    max_retries: int = 3,
    retry_delay: int = 1,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    expected_exceptions: tuple = (Exception,)
):
    """
    Decorator for retrying operations with exponential backoff
    
    Parameters:
    -----------
    max_retries : int
        Maximum number of retry attempts
    retry_delay : int
        Initial delay between retries in seconds
    backoff_factor : float
        Factor to increase delay for each retry
    jitter : bool
        Whether to add random jitter to delay
    expected_exceptions : tuple
        Exceptions that trigger retry
        
    Returns:
    --------
    Callable
        Decorator function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(__name__)
            
            retries = 0
            delay = retry_delay
            
            while True:
                try:
                    return func(*args, **kwargs)
                except expected_exceptions as e:
                    retries += 1
                    
                    if retries > max_retries:
                        logger.error(f"Max retries ({max_retries}) exceeded for {func.__name__}: {str(e)}")
                        raise
                    
                    # Calculate delay with optional jitter
                    if jitter:
                        # Add random jitter between 0% and 25%
                        jitter_amount = random.uniform(0, 0.25)
                        current_delay = delay * (1 + jitter_amount)
                    else:
                        current_delay = delay
                    
                    logger.warning(f"Retry {retries}/{max_retries} for {func.__name__} after {current_delay:.2f}s: {str(e)}")
                    
                    # Sleep before retry
                    time.sleep(current_delay)
                    
                    # Increase delay for next retry
                    delay *= backoff_factor
        
        return wrapper
    
    return decorator

class ConnectionManager:
    """
    Manage API connections with resilience
    
    This class combines circuit breaker and retry patterns
    for robust API connections.
    """
    
    def __init__(
        self,
        retry_config: Optional[Dict[str, Any]] = None,
        circuit_breaker_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize connection manager
        
        Parameters:
        -----------
        retry_config : Dict[str, Any], optional
            Configuration for retry mechanism
        circuit_breaker_config : Dict[str, Any], optional
            Configuration for circuit breaker
        """
        # Default retry configuration
        self.retry_config = {
            'max_retries': 3,
            'retry_delay': 1,
            'backoff_factor': 2.0,
            'jitter': True
        }
        
        # Default circuit breaker configuration
        self.circuit_breaker_config = {
            'failure_threshold': 5,
            'recovery_timeout': 60
        }
        
        # Update with provided configurations
        if retry_config:
            self.retry_config.update(retry_config)
        
        if circuit_breaker_config:
            self.circuit_breaker_config.update(circuit_breaker_config)
        
        self.logger = logging.getLogger(__name__)
        
        # Connection status tracking
        self.connection_status = {}
    
    def connect(self, service_name: str, connect_func: Callable, *args, **kwargs) -> Any:
        """
        Connect to a service with resilience
        
        Parameters:
        -----------
        service_name : str
            Name of the service to connect to
        connect_func : Callable
            Function to establish connection
        *args, **kwargs
            Arguments to pass to connect function
            
        Returns:
        --------
        Any
            Connection object or result from connect_func
        """
        # Create circuit breaker for this service
        circuit_breaker = CircuitBreaker(**self.circuit_breaker_config)
        
        # Apply circuit breaker and retry to connect function
        @circuit_breaker
        @with_retry(**self.retry_config)
        def resilient_connect():
            self.logger.info(f"Connecting to {service_name}...")
            result = connect_func(*args, **kwargs)
            self.logger.info(f"Connected to {service_name}")
            
            # Update connection status
            self.connection_status[service_name] = {
                'status': 'connected',
                'last_connected': datetime.now(),
                'failures': 0
            }
            
            return result
        
        try:
            return resilient_connect()
        except Exception as e:
            # Update connection status
            self.connection_status[service_name] = {
                'status': 'failed',
                'last_failure': datetime.now(),
                'failures': self.connection_status.get(service_name, {}).get('failures', 0) + 1,
                'last_error': str(e)
            }
            
            # Re-raise the exception
            raise
    
    def execute(self, service_name: str, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with resilience
        
        Parameters:
        -----------
        service_name : str
            Name of the service
        func : Callable
            Function to execute
        *args, **kwargs
            Arguments to pass to function
            
        Returns:
        --------
        Any
            Result from function
        """
        # Apply retry to function
        @with_retry(**self.retry_config)
        def resilient_execute():
            return func(*args, **kwargs)
        
        try:
            return resilient_execute()
        except Exception as e:
            # Update connection status
            if service_name in self.connection_status:
                self.connection_status[service_name]['failures'] = self.connection_status[service_name].get('failures', 0) + 1
                self.connection_status[service_name]['last_failure'] = datetime.now()
                self.connection_status[service_name]['last_error'] = str(e)
            
            # Re-raise the exception
            raise
    
    def health_check(self) -> Dict[str, Dict[str, Any]]:
        """
        Check health of all connections
        
        Returns:
        --------
        Dict[str, Dict[str, Any]]
            Health status for all connections
        """
        return self.connection_status