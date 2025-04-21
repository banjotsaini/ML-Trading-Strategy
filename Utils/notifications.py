import smtplib
import logging
import time
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta

class NotificationManager:
    """Manage and send notifications"""
    
    def __init__(
        self,
        email_config: Optional[Dict[str, str]] = None,
        sms_config: Optional[Dict[str, str]] = None,
        throttle_period: int = 300  # 5 minutes
    ):
        """
        Initialize notification manager
        
        Parameters:
        -----------
        email_config : Dict[str, str], optional
            Email configuration with keys:
            - smtp_server: SMTP server address
            - smtp_port: SMTP server port
            - username: Email username
            - password: Email password
            - from_email: Sender email address
        sms_config : Dict[str, str], optional
            SMS configuration (provider-specific)
        throttle_period : int
            Minimum time between notifications of the same type (seconds)
        """
        self.email_config = email_config
        self.sms_config = sms_config
        self.throttle_period = throttle_period
        
        # Notification history for throttling
        self.notification_history = {}
        
        # Logger
        self.logger = logging.getLogger(__name__)
    
    def send_signal_alert(
        self,
        ticker: str,
        signal: str,
        confidence: float,
        price: float,
        recipients: List[str],
        priority: str = 'normal'
    ) -> bool:
        """
        Send alert for new trading signals
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol
        signal : str
            Trading signal (BUY, SELL)
        confidence : float
            Signal confidence
        price : float
            Current price
        recipients : List[str]
            List of recipient email addresses or phone numbers
        priority : str
            Priority level ('high', 'normal', 'low')
            
        Returns:
        --------
        bool
            True if notification was sent, False otherwise
        """
        # Check if notification should be throttled
        notification_key = f"signal_{ticker}_{signal}"
        if self._should_throttle(notification_key, priority):
            return False
        
        # Create notification message
        subject = f"Trading Signal: {ticker} - {signal}"
        
        message = f"""
        Trading Signal Alert
        
        Ticker: {ticker}
        Signal: {signal}
        Confidence: {confidence:.4f}
        Price: ${price:.2f}
        Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        This is an automated notification from your ML Trading System.
        """
        
        # Send notification based on priority
        if priority == 'high':
            # Send both email and SMS for high priority
            email_sent = self._send_email(subject, message, recipients) if self.email_config else False
            sms_sent = self._send_sms(message, recipients) if self.sms_config else False
            return email_sent or sms_sent
        elif priority == 'normal':
            # Send only email for normal priority
            return self._send_email(subject, message, recipients) if self.email_config else False
        else:
            # Low priority - only log
            self.logger.info(f"Low priority signal alert: {ticker} - {signal}")
            return True
    
    def send_error_alert(
        self,
        error_message: str,
        error_details: Optional[str] = None,
        recipients: List[str] = None,
        priority: str = 'high'
    ) -> bool:
        """
        Send alert for errors
        
        Parameters:
        -----------
        error_message : str
            Brief error message
        error_details : str, optional
            Detailed error information
        recipients : List[str], optional
            List of recipient email addresses or phone numbers
        priority : str
            Priority level ('high', 'normal', 'low')
            
        Returns:
        --------
        bool
            True if notification was sent, False otherwise
        """
        # Check if notification should be throttled
        notification_key = f"error_{error_message[:50]}"
        if self._should_throttle(notification_key, priority):
            return False
        
        # Create notification message
        subject = f"Trading System Error: {error_message[:50]}"
        
        message = f"""
        Trading System Error Alert
        
        Error: {error_message}
        Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        {error_details if error_details else ''}
        
        This is an automated notification from your ML Trading System.
        """
        
        # Always log errors
        self.logger.error(f"Error alert: {error_message}")
        
        # Send notification based on priority
        if priority == 'high':
            # Send both email and SMS for high priority
            email_sent = self._send_email(subject, message, recipients) if self.email_config else False
            sms_sent = self._send_sms(message, recipients) if self.sms_config else False
            return email_sent or sms_sent
        elif priority == 'normal':
            # Send only email for normal priority
            return self._send_email(subject, message, recipients) if self.email_config else False
        else:
            # Low priority - only log
            return True
    
    def send_status_update(
        self,
        status_message: str,
        status_details: Optional[Dict[str, Any]] = None,
        recipients: List[str] = None,
        priority: str = 'low'
    ) -> bool:
        """
        Send periodic status updates
        
        Parameters:
        -----------
        status_message : str
            Brief status message
        status_details : Dict[str, Any], optional
            Detailed status information
        recipients : List[str], optional
            List of recipient email addresses or phone numbers
        priority : str
            Priority level ('high', 'normal', 'low')
            
        Returns:
        --------
        bool
            True if notification was sent, False otherwise
        """
        # Check if notification should be throttled
        notification_key = "status_update"
        if self._should_throttle(notification_key, priority):
            return False
        
        # Create notification message
        subject = f"Trading System Status: {status_message[:50]}"
        
        message = f"""
        Trading System Status Update
        
        Status: {status_message}
        Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        """
        
        # Add status details if provided
        if status_details:
            message += "Details:\n"
            for key, value in status_details.items():
                message += f"  {key}: {value}\n"
        
        message += "\nThis is an automated notification from your ML Trading System."
        
        # Send notification based on priority
        if priority == 'high':
            # Send both email and SMS for high priority
            email_sent = self._send_email(subject, message, recipients) if self.email_config else False
            sms_sent = self._send_sms(message, recipients) if self.sms_config else False
            return email_sent or sms_sent
        elif priority == 'normal':
            # Send only email for normal priority
            return self._send_email(subject, message, recipients) if self.email_config else False
        else:
            # Low priority - only log
            self.logger.info(f"Status update: {status_message}")
            return True
    
    def _should_throttle(self, notification_key: str, priority: str) -> bool:
        """
        Check if notification should be throttled
        
        Parameters:
        -----------
        notification_key : str
            Unique key for the notification type
        priority : str
            Priority level ('high', 'normal', 'low')
            
        Returns:
        --------
        bool
            True if notification should be throttled, False otherwise
        """
        # High priority notifications are never throttled
        if priority == 'high':
            return False
        
        # Check if notification was sent recently
        now = datetime.now()
        if notification_key in self.notification_history:
            last_sent = self.notification_history[notification_key]
            time_since_last = (now - last_sent).total_seconds()
            
            # Throttle based on priority
            if priority == 'normal' and time_since_last < self.throttle_period:
                return True
            elif priority == 'low' and time_since_last < self.throttle_period * 3:
                return True
        
        # Update notification history
        self.notification_history[notification_key] = now
        return False
    
    def _send_email(self, subject: str, message: str, recipients: List[str]) -> bool:
        """
        Send email notification
        
        Parameters:
        -----------
        subject : str
            Email subject
        message : str
            Email message
        recipients : List[str]
            List of recipient email addresses
            
        Returns:
        --------
        bool
            True if email was sent, False otherwise
        """
        if not self.email_config:
            self.logger.warning("Email configuration not provided")
            return False
        
        if not recipients:
            self.logger.warning("No recipients provided for email notification")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email_config['from_email']
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = subject
            
            # Add message body
            msg.attach(MIMEText(message, 'plain'))
            
            # Connect to SMTP server
            server = smtplib.SMTP(
                self.email_config['smtp_server'],
                int(self.email_config['smtp_port'])
            )
            server.starttls()
            server.login(
                self.email_config['username'],
                self.email_config['password']
            )
            
            # Send email
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"Email notification sent: {subject}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email notification: {str(e)}")
            return False
    
    def _send_sms(self, message: str, recipients: List[str]) -> bool:
        """
        Send SMS notification
        
        Parameters:
        -----------
        message : str
            SMS message
        recipients : List[str]
            List of recipient phone numbers
            
        Returns:
        --------
        bool
            True if SMS was sent, False otherwise
        """
        if not self.sms_config:
            self.logger.warning("SMS configuration not provided")
            return False
        
        if not recipients:
            self.logger.warning("No recipients provided for SMS notification")
            return False
        
        # This is a placeholder for SMS implementation
        # Actual implementation would depend on the SMS provider
        self.logger.info(f"SMS notification would be sent to {len(recipients)} recipients")
        return True