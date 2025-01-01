import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Tuple

class DataLoader:
    @staticmethod
    def fetch_historical_data(
        tickers: List[str], 
        start_date: str = '2020-01-01', 
        end_date: str = None
    ) -> dict:
        """
        Fetch historical stock data for multiple tickers
        
        Parameters:
        -----------
        tickers : List[str]
            List of stock ticker symbols
        start_date : str, optional
            Start date for historical data
        end_date : str, optional
            End date for historical data
        
        Returns:
        --------
        dict
            Dictionary of DataFrames for each ticker
        """
        if end_date is None:
            end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
        
        historical_data = {}
        
        for ticker in tickers:
            try:
                data = yf.download(ticker, start=start_date, end=end_date)
                historical_data[ticker] = data
            except Exception as e:
                print(f"Error downloading data for {ticker}: {e}")
        
        return historical_data

    def save_data(self, data, ticker):
        # Save data to local storage
        pass