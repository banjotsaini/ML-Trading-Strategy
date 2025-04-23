import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Union, Any, Callable
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PolynomialFeatures
from sklearn.decomposition import PCA, KernelPCA, FastICA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import talib
from scipy import stats, signal
import pywt
import logging
from datetime import datetime, timedelta
import re
import warnings

class FeatureStore:
    """
    Feature engineering and management for trading strategies
    """
    
    # Feature categories for organization and selection
    FEATURE_CATEGORIES = {
        'trend': [
            'MA5', 'MA10', 'MA20', 'MA50', 'MA200', 
            'EMA5', 'EMA10', 'EMA20', 'EMA50', 'EMA200',
            'DEMA', 'TEMA', 'TRIMA', 'KAMA', 'ADX', 'ADXR',
            'APO', 'AROON_UP', 'AROON_DOWN', 'AROONOSC',
            'BOP', 'CCI', 'CMO', 'DX', 'MACD', 'MACD_SIGNAL', 'MACD_HIST',
            'MOM', 'PPO', 'ROC', 'ROCR', 'TRIX'
        ],
        'momentum': [
            'RSI', 'STOCH_K', 'STOCH_D', 'STOCHF_K', 'STOCHF_D', 
            'STOCHRSI_K', 'STOCHRSI_D', 'WILLR', 'ULTOSC'
        ],
        'volatility': [
            'ATR', 'NATR', 'TRANGE', 'BB_UPPER', 'BB_MIDDLE', 'BB_LOWER',
            'BB_WIDTH', 'KC_UPPER', 'KC_MIDDLE', 'KC_LOWER', 'KC_WIDTH'
        ],
        'volume': [
            'OBV', 'AD', 'ADOSC', 'MFI', 'VWAP', 'CMF', 'EMV', 'VWMA'
        ],
        'cycle': [
            'HT_DCPERIOD', 'HT_DCPHASE', 'HT_PHASOR_INPHASE', 
            'HT_PHASOR_QUADRATURE', 'HT_SINE', 'HT_LEADSINE',
            'HT_TRENDMODE'
        ],
        'pattern': [
            'CDLENGULFING', 'CDLHAMMER', 'CDLSHOOTINGSTAR', 'CDLMORNINGSTAR',
            'CDLEVENINGSTAR', 'CDLHARAMI', 'CDLDOJI', 'CDLPIERCING'
        ],
        'price_transform': [
            'AVGPRICE', 'MEDPRICE', 'TYPPRICE', 'WCLPRICE'
        ],
        'statistical': [
            'RETURNS', 'LOG_RETURNS', 'REALIZED_VOL', 'SKEW', 'KURTOSIS',
            'ZSCORE', 'AUTOCORR_1', 'AUTOCORR_5'
        ],
        'time': [
            'DAY_OF_WEEK', 'DAY_OF_MONTH', 'WEEK_OF_YEAR', 'MONTH', 
            'QUARTER', 'YEAR', 'IS_MONTH_END', 'IS_MONTH_START',
            'IS_QUARTER_END', 'IS_QUARTER_START', 'IS_YEAR_END', 'IS_YEAR_START'
        ],
        'fundamental': [
            'PE_RATIO', 'PB_RATIO', 'DIVIDEND_YIELD', 'MARKET_CAP',
            'SECTOR_PERFORMANCE', 'INDUSTRY_PERFORMANCE'
        ],
        'cross_sectional': [
            'MARKET_REL_STRENGTH', 'SECTOR_REL_STRENGTH', 'INDUSTRY_REL_STRENGTH',
            'MARKET_BETA', 'SECTOR_BETA', 'PERCENTILE_RANK'
        ],
        'ratio': [
            'PRICE_TO_MA_RATIO', 'VOLUME_TO_MA_RATIO', 'HIGH_LOW_RATIO',
            'OPEN_CLOSE_RATIO', 'GAP_RATIO'
        ],
        'fractal': [
            'HURST_EXPONENT', 'FRACTAL_DIMENSION', 'DETRENDED_FLUCTUATION'
        ],
        'wavelet': [
            'WAVELET_A1', 'WAVELET_D1', 'WAVELET_D2', 'WAVELET_D3'
        ],
        'regime': [
            'MARKET_REGIME', 'VOLATILITY_REGIME', 'TREND_REGIME'
        ]
    }
    
    @staticmethod
    def calculate_technical_indicators(
        data: pd.DataFrame, 
        categories: Optional[List[str]] = None,
        include_patterns: bool = False,
        include_cycles: bool = False,
        include_time: bool = True,
        include_statistical: bool = True,
        include_cross_sectional: bool = False,
        include_ratio: bool = True,
        include_fractal: bool = False,
        include_wavelet: bool = False,
        include_regime: bool = False,
        market_data: Optional[pd.DataFrame] = None,
        sector_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Calculate technical indicators for feature engineering
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Historical stock price data with OHLCV columns
        categories : List[str], optional
            List of feature categories to include
        include_patterns : bool
            Whether to include candlestick patterns
        include_cycles : bool
            Whether to include cycle indicators
        include_time : bool
            Whether to include time-based features
        include_statistical : bool
            Whether to include statistical features
        include_cross_sectional : bool
            Whether to include cross-sectional features
        include_ratio : bool
            Whether to include ratio features
        include_fractal : bool
            Whether to include fractal features
        include_wavelet : bool
            Whether to include wavelet transform features
        include_regime : bool
            Whether to include market regime features
        market_data : pandas.DataFrame, optional
            Market index data for cross-sectional features
        sector_data : pandas.DataFrame, optional
            Sector index data for cross-sectional features
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with added technical indicators
        """
        # Create a copy to avoid modifying the original
        df = data.copy()
        
        # Ensure we have the required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in df.columns:
                # Try to find case-insensitive match
                matches = [c for c in df.columns if c.lower() == col.lower()]
                if matches:
                    df[col] = df[matches[0]]
                else:
                    raise ValueError(f"Required column {col} not found in data")
        
        # Ensure date is the index
        if 'date' in df.columns:
            df = df.set_index('date')
        
        # Get open, high, low, close, volume arrays for TA-Lib
        open_prices = df['Open'].values
        high_prices = df['High'].values
        low_prices = df['Low'].values
        close_prices = df['Close'].values
        volume = df['Volume'].values
        
        # Determine which categories to include
        if categories is None:
            # Default to all except patterns and cycles which can be noisy
            categories = [cat for cat in FeatureStore.FEATURE_CATEGORIES.keys() 
                         if cat != 'pattern' or include_patterns
                         if cat != 'cycle' or include_cycles
                         if cat != 'time' or include_time
                         if cat != 'statistical' or include_statistical
                         if cat != 'cross_sectional' or include_cross_sectional
                         if cat != 'ratio' or include_ratio
                         if cat != 'fractal' or include_fractal
                         if cat != 'wavelet' or include_wavelet
                         if cat != 'regime' or include_regime]
        
        # Calculate indicators based on selected categories
        for category in categories:
            if category not in FeatureStore.FEATURE_CATEGORIES:
                logging.warning(f"Unknown feature category: {category}")
                continue
                
            # Calculate indicators for this category
            if category == 'trend':
                # Moving Averages
                df['MA5'] = talib.SMA(close_prices, timeperiod=5)
                df['MA10'] = talib.SMA(close_prices, timeperiod=10)
                df['MA20'] = talib.SMA(close_prices, timeperiod=20)
                df['MA50'] = talib.SMA(close_prices, timeperiod=50)
                df['MA200'] = talib.SMA(close_prices, timeperiod=200)
                
                # Exponential Moving Averages
                df['EMA5'] = talib.EMA(close_prices, timeperiod=5)
                df['EMA10'] = talib.EMA(close_prices, timeperiod=10)
                df['EMA20'] = talib.EMA(close_prices, timeperiod=20)
                df['EMA50'] = talib.EMA(close_prices, timeperiod=50)
                df['EMA200'] = talib.EMA(close_prices, timeperiod=200)
                
                # Other trend indicators
                df['DEMA'] = talib.DEMA(close_prices, timeperiod=20)
                df['TEMA'] = talib.TEMA(close_prices, timeperiod=20)
                df['TRIMA'] = talib.TRIMA(close_prices, timeperiod=20)
                df['KAMA'] = talib.KAMA(close_prices, timeperiod=20)
                
                # Average Directional Index
                df['ADX'] = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)
                df['ADXR'] = talib.ADXR(high_prices, low_prices, close_prices, timeperiod=14)
                
                # Absolute Price Oscillator
                df['APO'] = talib.APO(close_prices, fastperiod=12, slowperiod=26)
                
                # Aroon Oscillator
                aroon_down, aroon_up = talib.AROON(high_prices, low_prices, timeperiod=14)
                df['AROON_DOWN'] = aroon_down
                df['AROON_UP'] = aroon_up
                df['AROONOSC'] = talib.AROONOSC(high_prices, low_prices, timeperiod=14)
                
                # Balance of Power
                df['BOP'] = talib.BOP(open_prices, high_prices, low_prices, close_prices)
                
                # Commodity Channel Index
                df['CCI'] = talib.CCI(high_prices, low_prices, close_prices, timeperiod=14)
                
                # Chande Momentum Oscillator
                df['CMO'] = talib.CMO(close_prices, timeperiod=14)
                
                # Directional Movement Index
                df['DX'] = talib.DX(high_prices, low_prices, close_prices, timeperiod=14)
                
                # MACD
                macd, macd_signal, macd_hist = talib.MACD(
                    close_prices, fastperiod=12, slowperiod=26, signalperiod=9
                )
                df['MACD'] = macd
                df['MACD_SIGNAL'] = macd_signal
                df['MACD_HIST'] = macd_hist
                
                # Momentum
                df['MOM'] = talib.MOM(close_prices, timeperiod=10)
                
                # Percentage Price Oscillator
                df['PPO'] = talib.PPO(close_prices, fastperiod=12, slowperiod=26)
                
                # Rate of Change
                df['ROC'] = talib.ROC(close_prices, timeperiod=10)
                df['ROCR'] = talib.ROCR(close_prices, timeperiod=10)
                
                # Triple Exponential Moving Average Oscillator
                df['TRIX'] = talib.TRIX(close_prices, timeperiod=30)
            
            elif category == 'momentum':
                # Relative Strength Index
                df['RSI'] = talib.RSI(close_prices, timeperiod=14)
                
                # Stochastic
                slowk, slowd = talib.STOCH(
                    high_prices, low_prices, close_prices,
                    fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0
                )
                df['STOCH_K'] = slowk
                df['STOCH_D'] = slowd
                
                # Fast Stochastic
                fastk, fastd = talib.STOCHF(
                    high_prices, low_prices, close_prices,
                    fastk_period=5, fastd_period=3, fastd_matype=0
                )
                df['STOCHF_K'] = fastk
                df['STOCHF_D'] = fastd
                
                # Stochastic RSI
                try:
                    fastk, fastd = talib.STOCHRSI(
                        close_prices, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0
                    )
                    df['STOCHRSI_K'] = fastk
                    df['STOCHRSI_D'] = fastd
                except:
                    # STOCHRSI can fail if there are NaN values
                    pass
                
                # Williams' %R
                df['WILLR'] = talib.WILLR(high_prices, low_prices, close_prices, timeperiod=14)
                
                # Ultimate Oscillator
                df['ULTOSC'] = talib.ULTOSC(
                    high_prices, low_prices, close_prices,
                    timeperiod1=7, timeperiod2=14, timeperiod3=28
                )
            
            elif category == 'volatility':
                # Average True Range
                df['ATR'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
                df['NATR'] = talib.NATR(high_prices, low_prices, close_prices, timeperiod=14)
                df['TRANGE'] = talib.TRANGE(high_prices, low_prices, close_prices)
                
                # Bollinger Bands
                upper, middle, lower = talib.BBANDS(
                    close_prices, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
                )
                df['BB_UPPER'] = upper
                df['BB_MIDDLE'] = middle
                df['BB_LOWER'] = lower
                df['BB_WIDTH'] = (upper - lower) / middle
                
                # Keltner Channels
                typical_price = (high_prices + low_prices + close_prices) / 3
                atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
                ema20 = talib.EMA(close_prices, timeperiod=20)
                
                df['KC_MIDDLE'] = ema20
                df['KC_UPPER'] = ema20 + (2 * atr)
                df['KC_LOWER'] = ema20 - (2 * atr)
                df['KC_WIDTH'] = (df['KC_UPPER'] - df['KC_LOWER']) / df['KC_MIDDLE']
                
                # Historical Volatility
                df['HIST_VOL_10'] = df['Close'].pct_change().rolling(window=10).std() * np.sqrt(252)
                df['HIST_VOL_20'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
                df['HIST_VOL_30'] = df['Close'].pct_change().rolling(window=30).std() * np.sqrt(252)
                
                # Volatility Ratio
                df['VOL_RATIO_10_30'] = df['HIST_VOL_10'] / df['HIST_VOL_30']
            
            elif category == 'volume':
                # On Balance Volume
                df['OBV'] = talib.OBV(close_prices, volume)
                
                # Chaikin A/D Line
                df['AD'] = talib.AD(high_prices, low_prices, close_prices, volume)
                
                # Chaikin A/D Oscillator
                df['ADOSC'] = talib.ADOSC(
                    high_prices, low_prices, close_prices, volume,
                    fastperiod=3, slowperiod=10
                )
                
                # Money Flow Index
                df['MFI'] = talib.MFI(high_prices, low_prices, close_prices, volume, timeperiod=14)
                
                # Volume-Weighted Average Price (VWAP)
                df['VWAP'] = FeatureStore._calculate_vwap(df)
                
                # Chaikin Money Flow
                df['CMF'] = FeatureStore._calculate_cmf(df, period=20)
                
                # Ease of Movement
                df['EMV'] = FeatureStore._calculate_emv(df)
                
                # Volume-Weighted Moving Average
                df['VWMA'] = FeatureStore._calculate_vwma(df, period=20)
                
                # Volume Oscillator
                vol_5 = talib.SMA(volume, timeperiod=5)
                vol_10 = talib.SMA(volume, timeperiod=10)
                df['VOL_OSC'] = ((vol_5 - vol_10) / vol_10) * 100
                
                # Price-Volume Trend
                df['PVT'] = (df['Close'].pct_change() * df['Volume']).cumsum()
                
                # Negative Volume Index
                df['NVI'] = FeatureStore._calculate_nvi(df)
                
                # Positive Volume Index
                df['PVI'] = FeatureStore._calculate_pvi(df)
                
                # Volume Ratio
                df['VOL_RATIO'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
            
            elif category == 'cycle' and include_cycles:
                # Hilbert Transform - Dominant Cycle Period
                try:
                    df['HT_DCPERIOD'] = talib.HT_DCPERIOD(close_prices)
                    
                    # Hilbert Transform - Dominant Cycle Phase
                    df['HT_DCPHASE'] = talib.HT_DCPHASE(close_prices)
                    
                    # Hilbert Transform - Phasor Components
                    inphase, quadrature = talib.HT_PHASOR(close_prices)
                    df['HT_PHASOR_INPHASE'] = inphase
                    df['HT_PHASOR_QUADRATURE'] = quadrature
                    
                    # Hilbert Transform - SineWave
                    sine, leadsine = talib.HT_SINE(close_prices)
                    df['HT_SINE'] = sine
                    df['HT_LEADSINE'] = leadsine
                    
                    # Hilbert Transform - Trend vs Cycle Mode
                    df['HT_TRENDMODE'] = talib.HT_TRENDMODE(close_prices)
                except:
                    # Hilbert Transform can fail with certain data
                    pass
                
                # Ehlers' Fisher Transform
                try:
                    df['FISHER_TRANSFORM'] = FeatureStore._calculate_fisher_transform(df, period=10)
                except:
                    pass
                
                # Ehlers' Sine Wave Indicator
                try:
                    df['SINE_WAVE'], df['LEAD_SINE_WAVE'] = FeatureStore._calculate_sine_wave(df, period=10)
                except:
                    pass
            
            elif category == 'pattern' and include_patterns:
                # Candlestick Patterns
                df['CDLENGULFING'] = talib.CDLENGULFING(
                    open_prices, high_prices, low_prices, close_prices
                )
                df['CDLHAMMER'] = talib.CDLHAMMER(
                    open_prices, high_prices, low_prices, close_prices
                )
                df['CDLSHOOTINGSTAR'] = talib.CDLSHOOTINGSTAR(
                    open_prices, high_prices, low_prices, close_prices
                )
                df['CDLMORNINGSTAR'] = talib.CDLMORNINGSTAR(
                    open_prices, high_prices, low_prices, close_prices, penetration=0
                )
                df['CDLEVENINGSTAR'] = talib.CDLEVENINGSTAR(
                    open_prices, high_prices, low_prices, close_prices, penetration=0
                )
                df['CDLHARAMI'] = talib.CDLHARAMI(
                    open_prices, high_prices, low_prices, close_prices
                )
                df['CDLDOJI'] = talib.CDLDOJI(
                    open_prices, high_prices, low_prices, close_prices
                )
                df['CDLPIERCING'] = talib.CDLPIERCING(
                    open_prices, high_prices, low_prices, close_prices
                )
                
                # Additional patterns
                df['CDLMARUBOZU'] = talib.CDLMARUBOZU(
                    open_prices, high_prices, low_prices, close_prices
                )
                df['CDLHANGINGMAN'] = talib.CDLHANGINGMAN(
                    open_prices, high_prices, low_prices, close_prices
                )
                df['CDLDARKCLOUDCOVER'] = talib.CDLDARKCLOUDCOVER(
                    open_prices, high_prices, low_prices, close_prices, penetration=0
                )
                
                # Pattern Recognition Score (sum of pattern signals)
                pattern_columns = [col for col in df.columns if col.startswith('CDL')]
                if pattern_columns:
                    df['PATTERN_SCORE'] = df[pattern_columns].sum(axis=1)
            
            elif category == 'price_transform':
                # Average Price
                df['AVGPRICE'] = talib.AVGPRICE(
                    open_prices, high_prices, low_prices, close_prices
                )
                
                # Median Price
                df['MEDPRICE'] = talib.MEDPRICE(high_prices, low_prices)
                
                # Typical Price
                df['TYPPRICE'] = talib.TYPPRICE(
                    high_prices, low_prices, close_prices
                )
                
                # Weighted Close Price
                df['WCLPRICE'] = talib.WCLPRICE(
                    high_prices, low_prices, close_prices
                )
                
                # Log Price
                df['LOG_PRICE'] = np.log(close_prices)
                
                # Price Difference
                df['PRICE_DIFF'] = df['Close'] - df['Open']
                
                # High-Low Range
                df['HL_RANGE'] = df['High'] - df['Low']
                
                # True Range Percentage
                df['TR_PCT'] = df['TRANGE'] / df['Close'].shift(1) * 100 if 'TRANGE' in df.columns else np.nan
            
            elif category == 'statistical' and include_statistical:
                # Returns
                df['RETURNS'] = df['Close'].pct_change()
                df['LOG_RETURNS'] = np.log(df['Close'] / df['Close'].shift(1))
                
                # Realized Volatility (20-day rolling standard deviation of returns)
                df['REALIZED_VOL'] = df['RETURNS'].rolling(window=20).std() * np.sqrt(252)
                
                # Skewness and Kurtosis (20-day rolling)
                df['SKEW'] = df['RETURNS'].rolling(window=20).skew()
                df['KURTOSIS'] = df['RETURNS'].rolling(window=20).kurt()
                
                # Z-Score (20-day)
                df['ZSCORE'] = (df['Close'] - df['Close'].rolling(window=20).mean()) / df['Close'].rolling(window=20).std()
                
                # Autocorrelation
                df['AUTOCORR_1'] = df['RETURNS'].rolling(window=20).apply(
                    lambda x: pd.Series(x).autocorr(lag=1), raw=False
                )
                df['AUTOCORR_5'] = df['RETURNS'].rolling(window=20).apply(
                    lambda x: pd.Series(x).autocorr(lag=5), raw=False
                )
                
                # Jarque-Bera Test for Normality
                def jarque_bera(x):
                    n = len(x)
                    if n < 3:
                        return np.nan
                    skew = stats.skew(x)
                    kurtosis = stats.kurtosis(x)
                    jb = n/6 * (skew**2 + kurtosis**2/4)
                    return jb
                
                df['JB_TEST'] = df['RETURNS'].rolling(window=30).apply(jarque_bera, raw=True)
                
                # Variance Ratio Test
                df['VAR_RATIO'] = FeatureStore._calculate_variance_ratio(df['Close'], period=10)
                
                # Runs Test for Randomness
                df['RUNS_TEST'] = df['RETURNS'].rolling(window=30).apply(
                    lambda x: FeatureStore._runs_test(x), raw=True
                )
            
            elif category == 'time' and include_time:
                # Reset index to get date as a column
                if isinstance(df.index, pd.DatetimeIndex):
                    df['DATE'] = df.index
                else:
                    try:
                        df['DATE'] = pd.to_datetime(df.index)
                    except:
                        logging.warning("Could not convert index to datetime for time features")
                        continue
                
                # Extract time features
                df['DAY_OF_WEEK'] = df['DATE'].dt.dayofweek
                df['DAY_OF_MONTH'] = df['DATE'].dt.day
                df['WEEK_OF_YEAR'] = df['DATE'].dt.isocalendar().week
                df['MONTH'] = df['DATE'].dt.month
                df['QUARTER'] = df['DATE'].dt.quarter
                df['YEAR'] = df['DATE'].dt.year
                df['IS_MONTH_END'] = df['DATE'].dt.is_month_end.astype(int)
                df['IS_MONTH_START'] = df['DATE'].dt.is_month_start.astype(int)
                df['IS_QUARTER_END'] = df['DATE'].dt.is_quarter_end.astype(int)
                df['IS_QUARTER_START'] = df['DATE'].dt.is_quarter_start.astype(int)
                df['IS_YEAR_END'] = df['DATE'].dt.is_year_end.astype(int)
                df['IS_YEAR_START'] = df['DATE'].dt.is_year_start.astype(int)
                
                # Seasonal decomposition
                try:
                    df['MONTH_SIN'] = np.sin(2 * np.pi * df['MONTH'] / 12)
                    df['MONTH_COS'] = np.cos(2 * np.pi * df['MONTH'] / 12)
                    df['DAY_SIN'] = np.sin(2 * np.pi * df['DAY_OF_MONTH'] / 31)
                    df['DAY_COS'] = np.cos(2 * np.pi * df['DAY_OF_MONTH'] / 31)
                    df['WEEKDAY_SIN'] = np.sin(2 * np.pi * df['DAY_OF_WEEK'] / 7)
                    df['WEEKDAY_COS'] = np.cos(2 * np.pi * df['DAY_OF_WEEK'] / 7)
                except:
                    pass
                
                # Drop the temporary DATE column
                df = df.drop('DATE', axis=1)
            
            elif category == 'cross_sectional' and include_cross_sectional:
                # Check if market data is provided
                if market_data is not None:
                    # Align market data with stock data
                    market_aligned = market_data.reindex(df.index, method='ffill')
                    
                    if 'Close' in market_aligned.columns:
                        # Market Relative Strength
                        df['MARKET_REL_STRENGTH'] = df['Close'] / df['Close'].shift(20) / (
                            market_aligned['Close'] / market_aligned['Close'].shift(20)
                        )
                        
                        # Market Beta (60-day rolling)
                        df['MARKET_RETURNS'] = market_aligned['Close'].pct_change()
                        
                        def calculate_beta(stock_returns, market_returns):
                            if len(stock_returns) < 2:
                                return np.nan
                            cov = np.cov(stock_returns, market_returns)[0, 1]
                            var = np.var(market_returns)
                            return cov / var if var != 0 else np.nan
                        
                        df['MARKET_BETA'] = df['RETURNS'].rolling(window=60).apply(
                            lambda x: calculate_beta(x, df.loc[x.index, 'MARKET_RETURNS']),
                            raw=False
                        )
                        
                        # Correlation with market
                        df['MARKET_CORR'] = df['RETURNS'].rolling(window=60).corr(df['MARKET_RETURNS'])
                        
                        # Clean up
