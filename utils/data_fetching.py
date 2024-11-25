import yfinance as yf
import pandas as pd
import numpy as np

class YahooFinanceDataFetcher:
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date

    def fetch_data(self):
        data = yf.download(
            tickers=self.tickers,
            start=self.start_date,
            end=self.end_date,
            adjusted=True
        )
        return data['Adj Close']
    
    

