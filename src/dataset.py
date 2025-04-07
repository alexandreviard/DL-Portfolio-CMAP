import yfinance as yf
import pandas as pd
import torch
import numpy as np 
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime, timedelta
from typing import Union, List, Dict, Tuple


# Fincial dataset ne fait que recuperer les données brutes de YF et ensuite cree des returns synthetiques 
# (eventuellement plusieurs simulations)

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
        n_simul: int = 1,
        start_date: str = '2006-03-01',
        end_date: str = '2020-12-31',
        log_returns: bool = False, 
        n_synthetic: int = None,
        calibrated: bool = True,
        mean: np.ndarray = np.zeros(2),
        cov: np.ndarray = np.eye(2),
        randomstate: Union[int, None] = 42,
    ) -> None:

        
        self.tickers = tickers
        self.synthetic = synthetic
        self.n_synthetic = n_synthetic
        self.n_simul = n_simul
        self.start_date = start_date
        self.end_date = end_date
        self.randomstate = randomstate
        self._raw_data = self._load_yf_data(log_returns)
        self.dataset = self._get_market_data()

        """GÉRER LES N_SIMULS POUR LA PROCHAINE FOIS"""
        if synthetic:
            if not self.n_synthetic:
                self.n_synthetic = self.dataset.shape[1]

            if calibrated:
                self.dataset_synthetic = self._get_synthetic_data_calibrated()
            else : 
                self.mean = mean 
                self.cov = cov
                self.dataset_synthetic = self._get_synthetic_data()
    
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

    def _get_synthetic_data_calibrated(self) -> torch.Tensor:
        """
        data simulée selon une loi normale multivariée calibrée sur les données 
        renvoie un tenseur de dim (n_simul, n_dates, n_assets)
        """
        np.random.seed(self.randomstate)
        returns = self._raw_data['returns'].values

        # échelle journalier
        self.mu = returns.mean(axis=0)
        self.Sigma = np.cov(returns, rowvar=False)  

        synthetic_returns = np.random.multivariate_normal(self.mu, self.Sigma, size=(self.n_synthetic, self.n_simul)) # dim (n_assets, n_dates, n_simul) ??

        # on bascule les dimensions dans l'ordre canonique (n_simul, n_dates, n_assets) 
        synthetic_returns = np.transpose(synthetic_returns, (1, 0, 2))

        tensor_returns = torch.tensor(synthetic_returns, dtype=torch.float32)

        return tensor_returns 
    
    def _get_synthetic_data(self) -> torch.Tensor:
        """
        data simulée selon une loi normale multivariée selon une mean et une cov définies par l'utilisateur
        renvoie un tenseur de dim (n_simul, n_dates, n_assets)
        """
        np.random.seed(self.randomstate)

        # échelle journalier
        synthetic_returns = np.random.multivariate_normal(self.mu, self.Sigma, size=(self.n_synthetic, self.n_simul)) # dim (n_assets, n_dates, n_simul) ??

        # on bascule les dimensions dans l'ordre canonique (n_simul, n_dates, n_assets) 
        synthetic_returns = np.transpose(synthetic_returns, (1, 0, 2))

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
 
# DataHandler 
# genere des periodes de training (couples d'index) 
# compute data = helper pour la creation du dataloader  
# utilise un dataloader pytorch 
    
class DataHandler:
    def __init__(self,
                 dataset: FinancialDataset,
                 initial_train_years: int = 4,
                 retrain_years: int = 2,
                 rolling_window: int =50,
                 batch_size: int=32,
                 overlap: bool=True,
                 shuffle: bool=True,
                 verbose:bool =True,
                 on_synthetic:bool = False) -> None:
        
        self.dataset = dataset 
        self.on_synthetic = on_synthetic
        if self.on_synthetic:
            self.n_simul = self.dataset.dataset_synthetic.shape[0]
            self.n_obs = self.dataset.dataset_synthetic.shape[1]
        else:
            self.n_simul = self.dataset.dataset.shape[0]
            self.n_obs = self.dataset.dataset.shape[1]
            
        self.n_assets = len(self.dataset.tickers)
        self.initial_train_years= initial_train_years
        self.retrain_years= retrain_years
        self.rolling_window = rolling_window
        self.start_index = self.rolling_window+1
        self.periods_train, self.periods_invest = self._generate_training_periods()
        self.batch_size = batch_size
        self.overlap = overlap
        self.shuffle = shuffle
        self.verbose = verbose

    def _generate_training_periods(self):
        
        training_periods = []
        test_periods = []

        if self.on_synthetic:
            n_dates = self.dataset.dataset_synthetic.shape[1]
        else:
            n_dates = self.dataset.dataset.shape[1]

        train_len = self.initial_train_years * 252
        retrain_len = self.retrain_years * 252

        training_periods.append((self.start_index, train_len+self.start_index))
        test_periods.append((train_len+self.start_index, train_len+self.start_index+retrain_len))

        current_start = train_len+self.start_index

        while current_start + 2 * retrain_len <= n_dates:
            train_start = current_start
            train_end = train_start + retrain_len

            test_start = train_end
            test_end = test_start + retrain_len

            training_periods.append((train_start, train_end))
            test_periods.append((test_start, test_end))

            current_start = test_start

        return training_periods, test_periods

    
    def _compute_data(self, start, end, training=True) -> List:
        

        rolling_data = []
        
        if self.on_synthetic: 
            data = self.dataset.dataset_synthetic
        else:
            data = self.dataset.dataset

        for sim in range(self.n_simul):
            
            if training:
                if self.overlap:
                    for i in range(start, end):

                        X = data[sim, i - self.rolling_window: i, :]
                        Y = data[sim, i - self.rolling_window+1: i+1, :]
                        rolling_data.append((X, Y))
                else:
                    for i in range(start, end, self.rolling_window):
                        X = data[sim, i - self.rolling_window: i, :]
                        Y = data[sim, i - self.rolling_window+1: i+1, :]
                        rolling_data.append((X, Y))
            else:
                for i in range(start, end):
                    X = data[sim, i - self.rolling_window: i, :]
                    rolling_data.append(X)
        
        return rolling_data

    def load_period(self, period_index=0):

        start_training, end_training = self.periods_train[period_index]
        start_invest, end_invest = self.periods_invest[period_index]

        if self.verbose:
            print(f'Training period from {start_training} to {end_training}')
            print(f'Investment period from {start_invest} to {end_invest}')

        data_training = self._compute_data(start=start_training, end=end_training, training=True)
        data_invest = self._compute_data(start=start_invest, end=end_invest, training=False)

        X_array = np.array([df[0] for df in data_training])
        Y_array = np.array([df[1] for df in data_training])
        X_test_array = np.array([df for df in data_invest])

        X_tensor = torch.tensor(X_array, dtype=torch.float32)
        Y_tensor = torch.tensor(Y_array, dtype=torch.float32)
        X_test = torch.tensor(X_test_array, dtype=torch.float32)

        X_tensor = X_tensor.view(-1, self.rolling_window, self.n_assets)
        Y_tensor = Y_tensor.view(-1, self.rolling_window, self.n_assets)
        X_test = X_test.view(-1, self.rolling_window, self.n_assets)

        dataset = TensorDataset(X_tensor, Y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)

        return dataloader, X_test, (start_training, end_training, start_invest, end_invest)
    
    
    