import yfinance as yf
import pandas as pd
import numpy as np


class YfDataLoader:
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date

    def fetch_data(self):
        data = yf.download(
            tickers=self.tickers,
            start=self.start_date,
            end=self.end_date
        )
        return data['Adj Close']


class FinancialDataProcessor:
    @staticmethod
    def calculate_returns(price_data):
        """
        Calcule les rendements quotidiens des prix donnés.

        Parameters:
        - price_data (pd.DataFrame): Données des prix.

        Returns:
        - pd.DataFrame: Rendements quotidiens.
        """
        return price_data.pct_change().dropna()

    @staticmethod
    def calculate_covariance(returns_data):
        """
        Calcule la matrice de covariance des rendements.

        Parameters:
        - returns_data (pd.DataFrame): Rendements des actifs.

        Returns:
        - pd.DataFrame: Matrice de covariance.
        """
        return returns_data.cov()


class Portfolio:
    def __init__(self, returns_data:pd.Dataframe, weights=None):
        
        self.returns = returns_data
        self.assets = returns_data.columns  # Asset names
        self.dynamic_weights = None  # Dynamic weights over time





        
    

