import pytz
import pandas as pd
from datetime import datetime, time, timedelta
from pandas.tseries.holiday import USFederalHolidayCalendar
from typing import Callable, Optional

class MarketScheduler:
    """Schedule operations during market hours"""
    
    def __init__(self, timezone: str = 'America/New_York'):
        """
        Initialize market scheduler
        
        Parameters:
        -----------
        timezone : str
            Timezone for market hours (default: 'America/New_York' for US markets)
        """
        self.timezone = pytz.timezone(timezone)
        self.calendar = USFederalHolidayCalendar()
        
        # Regular market hours (9:30 AM to 4:00 PM Eastern Time)
        self.market_open = time(9, 30)
        self.market_close = time(16, 0)
        
        # Pre-market hours (4:00 AM to 9:30 AM Eastern Time)
        self.pre_market_open = time(4, 0)
        self.pre_market_close = self.market_open
        
        # After-hours (4:00 PM to 8:00 PM Eastern Time)
        self.after_hours_open = self.market_close
        self.after_hours_close = time(20, 0)
    
    def is_market_open(self, include_extended: bool = False) -> bool:
        """
        Check if market is currently open
        
        Parameters:
        -----------
        include_extended : bool
            Whether to include pre-market and after-hours
            
        Returns:
        --------
        bool
            True if market is open, False otherwise
        """
        # Get current time in market timezone
        now = datetime.now(self.timezone)
        current_time = now.time()
        
        # Check if today is a holiday
        holidays = self.calendar.holidays(start=now.date(), end=now.date())
        if len(holidays) > 0:
            return False
        
        # Check if it's a weekend
        if now.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            return False
        
        # Check regular market hours
        if self.market_open <= current_time < self.market_close:
            return True
        
        # Check extended hours if requested
        if include_extended:
            if self.pre_market_open <= current_time < self.pre_market_close:
                return True
            if self.after_hours_open <= current_time < self.after_hours_close:
                return True
        
        return False
    
    def get_next_market_open(self) -> datetime:
        """
        Get next market open time
        
        Returns:
        --------
        datetime
            Next market open time
        """
        now = datetime.now(self.timezone)
        current_date = now.date()
        current_time = now.time()
        
        # If market is already open today, return current time
        if self.is_market_open() and current_time < self.market_close:
            return datetime.combine(current_date, current_time).replace(tzinfo=self.timezone)
        
        # Start with tomorrow
        next_date = current_date + timedelta(days=1)
        
        # Find next business day
        while True:
            next_datetime = datetime.combine(next_date, self.market_open).replace(tzinfo=self.timezone)
            
            # Check if it's a weekend
            if next_datetime.weekday() >= 5:
                next_date += timedelta(days=1)
                continue
            
            # Check if it's a holiday
            holidays = self.calendar.holidays(start=next_date, end=next_date)
            if len(holidays) > 0:
                next_date += timedelta(days=1)
                continue
            
            # Found next business day
            break
        
        return datetime.combine(next_date, self.market_open).replace(tzinfo=self.timezone)
    
    def get_next_market_close(self) -> datetime:
        """
        Get next market close time
        
        Returns:
        --------
        datetime
            Next market close time
        """
        now = datetime.now(self.timezone)
        current_date = now.date()
        current_time = now.time()
        
        # If market is open today and will close later today
        if self.is_market_open() and current_time < self.market_close:
            return datetime.combine(current_date, self.market_close).replace(tzinfo=self.timezone)
        
        # Get next market open first
        next_open = self.get_next_market_open()
        
        # Market close is on the same day as next open
        next_close_date = next_open.date()
        
        return datetime.combine(next_close_date, self.market_close).replace(tzinfo=self.timezone)
    
    def schedule_during_market_hours(self, func: Callable, *args, **kwargs):
        """
        Schedule a function to run during market hours
        
        Parameters:
        -----------
        func : Callable
            Function to schedule
        *args, **kwargs
            Arguments to pass to the function
        """
        import threading
        
        def _run_during_market_hours():
            while True:
                # Check if market is open
                if self.is_market_open():
                    # Run the function
                    func(*args, **kwargs)
                    
                    # Sleep until next check (every minute during market hours)
                    time.sleep(60)
                else:
                    # Get next market open
                    next_open = self.get_next_market_open()
                    
                    # Calculate seconds until next open
                    now = datetime.now(self.timezone)
                    seconds_until_open = (next_open - now).total_seconds()
                    
                    # Sleep until next open (or check every 5 minutes)
                    sleep_time = min(seconds_until_open, 300)
                    time.sleep(sleep_time)
        
        # Start scheduler thread
        scheduler_thread = threading.Thread(target=_run_during_market_hours)
        scheduler_thread.daemon = True
        scheduler_thread.start()
        
        return scheduler_thread