import yfinance as yf
import pandas as pd
import numpy as np
from utils.optimization import diversification_ratio, get_weights_md


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
    def __init__(self, returns_data:pd.DataFrame, weights=None):
        self.returns = returns_data
        self.assets = returns_data.columns  # Asset names
        self.n_assets_ = len(self.assets)
        
        if weights :
            self.dynamic_weights = weights
        else :
            self.dynamic_weights = pd.DataFrame(np.ones_like(self.returns) / self.n_assets_, index=returns_data.index, columns=self.assets)

    def estimate_asset_vol(self, rolling_window=50, win_type=None):
        self.volatilities = self.returns.rolling(window=rolling_window, win_type=win_type).std().dropna()
        return self.volatilities

    def optimize_portfolio(self, optimizer= "default", bounds=None, rolling_window=50, win_type=None):
        """
        calcule les covariances sur des fenêres de t à (t - 49) puis calcule les poids à partir de la covariance 
        """
        
        cov_rolling = self.returns.rolling(window=rolling_window, win_type=win_type).cov().dropna()
        dates = cov_rolling.index.get_level_values('Date').unique()
        n_dates = len(dates)

        self.dynamic_weights = pd.DataFrame(np.zeros(shape=(n_dates, self.n_assets_)), index=dates, columns=cov_rolling.columns)

        if optimizer == "default":

            for i in range(n_dates - 1):
                cov = cov_rolling.loc[dates[i]]
                self.dynamic_weights.loc[dates[i]] = get_weights_md(cov, bounds=bounds)

    def get_portfolio_returns(self):
        weighted_returns = self.returns.loc[self.dynamic_weights.index] * self.dynamic_weights
        self.portfolio_returns = weighted_returns.sum(axis=1)
        return self.portfolio_returns
    
    def get_portfolio_vol(self, rolling_window=50, win_type=None):
        weighted_returns = self.returns * self.dynamic_weights
        portfolio_volatility = weighted_returns.rolling(window=rolling_window, win_type=win_type).std().dropna()
        return portfolio_volatility
    
    def get_sharpe_ratio(self, start_date, end_date):
        return self.portfolio_returns.loc[start_date:end_date].mean() / self.portfolio_returns.loc[start_date:end_date].std() 


        
    

        


        










        
    

