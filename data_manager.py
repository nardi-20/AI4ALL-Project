import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import ta

class DataManager:
    def __init__(self):
        self.historical_data = {}
        self.load_historical_data()
    
    def load_historical_data(self):
        """Load historical CSV data"""
        try:
            self.historical_data = {
                'AAPL': pd.read_csv('AI4ALL Project Datasets/AAPL.csv'),
                'MSFT': pd.read_csv('AI4ALL Project Datasets/MSFT.csv'),
                'INTC': pd.read_csv('AI4ALL Project Datasets/INTC.csv'),
                'IBM': pd.read_csv('AI4ALL Project Datasets/IBM.csv'),
                'SP500': pd.read_csv('AI4ALL Project Datasets/GSPC.csv'),
                'InterestRates': pd.read_csv('AI4ALL Project Datasets/federalReserveInterestRates.csv')
            }
            
            for ticker in ['AAPL', 'MSFT', 'INTC', 'IBM', 'SP500']:
                self.historical_data[ticker]['Date'] = pd.to_datetime(
                    self.historical_data[ticker]['Date'], utc=True
                ).dt.tz_localize(None)
            
            self.historical_data['InterestRates'] = self.historical_data['InterestRates'].interpolate()
            self.historical_data['InterestRates']['Date'] = pd.to_datetime(
                self.historical_data['InterestRates'][['Year', 'Month', 'Day']]
            )
            
            return True
        except Exception as e:
            print(f"Error loading historical data: {e}")
            return False
    
    def fetch_real_time_data(self, ticker, period='30d'):
        """Fetch real-time data using yfinance"""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period, interval='1d')
            data.reset_index(inplace=True)
            data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)
            return data
        except Exception as e:
            print(f"Error fetching real-time data for {ticker}: {e}")
            return None
    
    def get_historical_data(self, ticker):
        """Get historical data for a ticker"""
        return self.historical_data.get(ticker)
    
    def calculate_correlation_matrix(self, selected_ticker, price_type='Close'):
        """Calculate correlation matrix for selected ticker and other features"""
        stock_data = {}
        for ticker in ['AAPL', 'MSFT', 'INTC', 'IBM']:
            df = self.historical_data[ticker]
            stock_data[ticker] = df[['Date', price_type]].rename(columns={price_type: ticker})
        
        sp500 = self.historical_data['SP500']
        stock_data['SP500'] = sp500[['Date', price_type]].rename(columns={price_type: 'SP500'})
        
        interest_rates = self.historical_data['InterestRates']
        stock_data['Interest_Rate'] = interest_rates[['Date', 'Effective Federal Funds Rate']]
        
        merged = None
        for name, df in stock_data.items():
            if merged is None:
                merged = df
            else:
                merged = pd.merge(merged, df, on='Date', how='outer')
        
        merged.sort_values('Date', inplace=True)
        merged.interpolate(method='linear', inplace=True)
        merged.dropna(inplace=True)
        
        corr_matrix = merged.drop(columns='Date').corr()
        
        return corr_matrix, merged
    
    def generate_features(self, df):
        """Generate features for model training/prediction"""
        df = df.copy()
        df['Close_Lag1'] = df['Close'].shift(1)
        df['Close_Lag2'] = df['Close'].shift(2)
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['Momentum_5'] = df['Close'] - df['Close'].shift(5)
        df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
        df['Daily_Return'] = df['Close'].pct_change()
        df['Volume_Lag1'] = df['Volume'].shift(1)
        df['OBV'] = ta.volume.OnBalanceVolumeIndicator(
            close=df['Close'], volume=df['Volume']
        ).on_balance_volume()
        
        return df
    
    def generate_labels(self, df, column='Close', days_ahead=5, threshold=0.005):
        """Generate labels for training"""
        df = df.copy()
        df['FuturePrice'] = df[column].shift(-days_ahead)
        df['Return'] = (df['FuturePrice'] - df[column]) / df[column]
        df['Action'] = df['Return'].apply(
            lambda x: 1 if x > threshold else (-1 if x < -threshold else 0)
        )
        return df