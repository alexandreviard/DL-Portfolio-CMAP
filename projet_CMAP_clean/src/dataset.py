import yfinance as yf
import pandas as pd
import torch
import numpy as np 
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime, timedelta
from typing import Union, List, Dict, Tuple


# A DISCUTER - POUR L'INSTANT ON PREND LE PARTI DE RENVOYER DES TENSEURS SANS LES DONNÉES DE PRIX 

class FinancialDataset:
    """
    Classe gérant la récupération et la préparation de données 
    contient un DataFrame pandas 

    Paramètres 
    ----------
    tickers : list of str, optional
        Nombre de points (jours) à générer pour la version synthétique.
    synthetic : bool, optional
        Si True, génère des données synthétiques basées sur la covariance et la moyenne
        des rendements historiques. Sinon, télécharge les données réelles 
    n_synthetic : int, optional
        Nombre de points (jours) à générer pour la version synthétique.
    n_simul : int, optional
        Nombre de dataset simulés. Par défaut '1'
    start_date : str or datetime, optional
        Date de début des données. Par défaut '2006-03-01'.
        Peut être une chaîne (ex: '2006-03-01') ou un objet datetime.
    end_date : str or datetime, optional
        Date de fin des données. Par défaut '2020-12-31'.
        Peut être une chaîne (ex: '2020-12-31') ou un objet datetime.
    randomstate : int, optional
        seed pour le rng
    """

    def __init__(
        self,
        tickers: List[str] = ['MC.PA', 'AIR.PA'],  # Airbus et LVMH
        synthetic: bool = True,
        n_synthetic: int = 2000,
        n_simul: int = 1,
        start_date: str = '2006-03-01',
        end_date: str = '2020-12-31',
        log_returns: bool = False, 
        randomstate: Union[int, None] = 42
    ) -> None:
        self.tickers = tickers
        self.synthetic = synthetic
        self.n_synthetic = n_synthetic
        self.n_simul = n_simul
        self.start_date = start_date
        self.end_date = end_date
        self.randomstate = randomstate
        
        self._raw_data = self._load_yf_data(log_returns)

        if synthetic:
            self.dataset = self._get_synthetic_data()
        else : 
            self.dataset = self._get_market_data()

    
    def _load_yf_data(self, log_return: bool = False) -> Dict[str, pd.DataFrame]:
        """
        télécharge les prix et calcule les returns (ou les log returns)
        et les stocke dans un dictionnaire
        """

        data_dict = {}

        prices = yf.download(
            self.tickers,
            start=self.start_date,
            end=self.end_date,
            interval="1d"
            )['Close']
        
        # Supprime la time zone éventuelle
        prices.index = prices.index.tz_localize(None).floor('D')
        
        # calcul des returns, attention, on drop une date 
        if log_return:
            returns = np.log(prices / prices.shift(1)).dropna()
        else : 
            returns = prices.pct_change().dropna()

        all_cols_returns = [f"{col}_rt" for col in prices.columns]
        returns.columns = all_cols_returns

        data_dict['prices'] = prices
        data_dict['returns'] = returns
        return data_dict

    def _get_market_data(self) -> torch.Tensor:
        """ 
        renvoie un tenseur de dim (1, n_dates, n_assets)
        """
        returns = self._raw_data['returns'].values
        tensor_returns = torch.tensor(returns, dtype=torch.float32)

        return tensor_returns.unsqueeze(0)

    def _get_synthetic_data(self) -> torch.Tensor:
        """
        data simulée selon une loi normale multivariée calibrée sur les données 
        renvoie un tenseur de dim (n_simul, n_dates, n_assets)
        """
        returns = self._raw_data['returns'].values

        # échelle journalier
        mu = returns.mean(axis=0)
        Sigma = np.cov(returns, rowvar=False)  

        synthetic_returns = np.random.multivariate_normal(mu, Sigma, size=(self.n_synthetic, self.n_simul)) # dim (n_assets, n_dates, n_simul)

        # on bascule les dimensions dans l'ordre canonique (n_simul, n_dates, n_assets)
        synthetic_returns = np.transpose(synthetic_returns, (1, 0, 2))

        # POUR L'INSTANT ON PREND LE PARTI DE RENVOYER DES TENSEURS SANS LES DONNÉES DE PRIX 
        # on crée un date index en cohérence avec le cas données rélles 
        # date_index = pd.date_range(
        #     pd.to_datetime(self.start_date),
        #     pd.to_datetime(self.start_date) + timedelta(self.n_synthetic - 1),
        #     freq='D'
        # )
        tensor_returns = torch.tensor(synthetic_returns, dtype=torch.float32)

        return tensor_returns
    
    def prices(self) -> pd.DataFrame:
        """
        Retourne le DataFrame des prix (close) de yf 
        """
        return self._raw_data["prices"]
    
    def returns(self) -> pd.DataFrame:
        """
        Retourne le DataFrame des rendements journaliers de yf 
        """
        return self._raw_data["returns"] 
    

class DataHandler:
    def __init__(self, dataset: torch.Tensor, start_date: str) -> None:
        """
        Gestionnaire des données financières pour l'entraînement d'un modèle.
        
        Paramètres :
        - dataset : torch.Tensor : Tenseur (n_simul, n_dates, n_assets).
        - start_date : str : Date de début des données (ex: '2006-03-01').
        """
        self.dataset = dataset
        self.start_date = start_date
        self.date_range = pd.date_range(start=start_date, periods=dataset.shape[1], freq="D")

    def get_periods(self, initial_train_years:int = 4, retrain_years:int = 2, unit:str = "years") -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """Génère les périodes d'entraînement et de test."""
        unit_mapping = {"years": 252, "months": 21, "weeks": 5, "days": 1}
        period_length = unit_mapping[unit]

        if initial_train_years * period_length >= len(self.date_range):
            raise ValueError("Pas assez de données pour l'entraînement initial.")

        periods = []
        train_start = self.date_range[0]
        train_end = self.date_range[initial_train_years * period_length]
        
        while train_end in self.date_range:
            test_start = train_end
            test_end_idx = self.date_range.get_loc(test_start) + retrain_years * period_length
            
            if test_end_idx >= len(self.date_range):
                break  # Stop si pas assez de données restantes

            test_end = self.date_range[test_end_idx]
            periods.append((train_start, train_end, test_start, test_end))

            train_start = test_start
            train_end = test_end

        return periods

        



