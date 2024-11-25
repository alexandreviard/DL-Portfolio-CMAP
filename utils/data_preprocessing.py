import pandas as pd
import numpy as np

class FinancialDataProcessor:
    def calculate_returns(self, price_data):
        returns = price_data.pct_change().dropna()
        return returns

    def calculate_mean_returns(self, returns_data):
        mean_returns = returns_data.mean()
        return mean_returns

    def calculate_covariance(self, returns_data):
        cov_matrix = returns_data.cov()
        return cov_matrix
