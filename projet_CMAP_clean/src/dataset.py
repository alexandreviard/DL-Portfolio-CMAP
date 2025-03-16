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
        np.random.seed(self.randomstate)
        returns = self._raw_data['returns'].values

        # échelle journalier
        mu = returns.mean(axis=0)
        Sigma = np.cov(returns, rowvar=False)  

        synthetic_returns = np.random.multivariate_normal(mu, Sigma, size=(self.n_simul, self.n_synthetic)) # dim (n_assets, n_dates, n_simul)

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

    def _generate_training_periods(self, initial_train_years=4, retrain_years=2, is_synthetic=False):
        """
        Génère les périodes d'entraînement et de test en fonction du type de dataset.
        - Si `is_synthetic=True`, utilise un `date_range` artificiel basé sur `start_date`.
        - Sinon, utilise les indices réels du dataset.
        """

        training_periods = []
        test_periods = []

        # Choix de la source des dates
        if is_synthetic:
            date_source = pd.date_range(start=self.start_date, periods=self.dataset.shape[1], freq="B")
        else:
            date_source = self.dataset["returns"].index

        # Définition de la première période d'entraînement
        last_date_1st_training = date_source[initial_train_years * 252] # on met la derniere date d'entrainement egale à 252*nbre d'année car apres dans la bcle d'entrainement on va l'exclure (ou pas XD)
        training_periods.append((date_source[0], last_date_1st_training))

        # Première période de test
        first_invest_date = last_date_1st_training
        end_date_first_invest = date_source[date_source.get_loc(first_invest_date) + retrain_years * 252]

        test_periods.append((first_invest_date, end_date_first_invest))

        # Nombre total de périodes possibles
        n_periods = (len(date_source[date_source >= last_date_1st_training])) // (retrain_years * 252)

        training_date = first_invest_date

        for i in range(n_periods - 1):
            training_date, end_training_date = training_date, date_source[date_source.get_loc(training_date) + retrain_years * 252]
            invest_date, end_invest_date = end_training_date, date_source[date_source.get_loc(end_training_date) + retrain_years * 252]

            training_periods.append((training_date, end_training_date))
            test_periods.append((invest_date, end_invest_date))

            training_date = invest_date

        return training_periods, test_periods
    
    class DataHandler:
    def __init__(self, dataset: torch.Tensor, start_date: str, is_synthetic: bool = False) -> None:
        """
        Gestionnaire des données financières pour plusieurs simulations.
        
        Paramètres :
        - dataset : torch.Tensor : Tenseur (n_simul, n_dates, n_assets).
        - start_date : str : Date de début des données (ex: '2006-03-01').
        - is_synthetic : bool : Indique si les données sont synthétiques (True) ou réelles (False).
        """
        self.dataset = dataset  # (n_simul, n_dates, n_assets)
        self.start_date = start_date
        self.is_synthetic = is_synthetic
        self.n_simul, self.n_dates, self.n_assets = dataset.shape

        # Génération de la plage de dates
        self.date_range = pd.date_range(start=self.start_date, periods=self.n_dates, freq="B")

    def _generate_training_periods(self, initial_train_years=4, retrain_years=2) -> Tuple[List, List]:
        """
        Génère les périodes d'entraînement et de test en fonction du dataset.

        Retourne :
        - `training_periods` : Liste des périodes d'entraînement pour chaque simulation.
        - `test_periods` : Liste des périodes de test pour chaque simulation.
        """

        training_periods = []
        test_periods = []

        # Définition de la première période d'entraînement
        last_date_1st_training = self.date_range[initial_train_years * 252]
        training_periods.append((self.date_range[0], last_date_1st_training))

        # Première période de test
        first_invest_date = last_date_1st_training
        end_date_first_invest = self.date_range[self.date_range.get_loc(first_invest_date) + retrain_years * 252]
        test_periods.append((first_invest_date, end_date_first_invest))

        # Nombre total de cycles possibles
        n_periods = (len(self.date_range[self.date_range >= last_date_1st_training])) // (retrain_years * 252)

        training_date = first_invest_date

        for _ in range(n_periods - 1):
            training_date, end_training_date = training_date, self.date_range[self.date_range.get_loc(training_date) + retrain_years * 252]
            invest_date, end_invest_date = end_training_date, self.date_range[self.date_range.get_loc(end_training_date) + retrain_years * 252]

            training_periods.append((training_date, end_training_date))
            test_periods.append((invest_date, end_invest_date))

            training_date = invest_date

        return training_periods, test_periods

    def _compute_data(self, start, end, rolling_window=50, overlap=True, training=True) -> List[Tuple]:
        """
        Calcule les fenêtres glissantes pour toutes les simulations.

        Retourne une liste de tuples `(X, Y)` pour chaque simulation.
        shape du return pour TRAINING de la forme suivante: [
        [ (X1_sim1, Y1_sim1), (X2_sim1, Y2_sim1), ... ],  # Simulation 1
        [ (X1_sim2, Y1_sim2), (X2_sim2, Y2_sim2), ... ],  # Simulation 2
        ...
        [ (X1_simN, Y1_simN), (X2_simN, Y2_simN), ... ]   # Simulation N
        ]
        (n_simul, n_rolling_windows, 2, rolling_window, n_assets) 

        Pour TEST : y a que des X
        """

        rolling_data = []
        idx_start, idx_end = self.date_range.get_loc(start), self.date_range.get_loc(end)

        for sim in range(self.n_simul):  # Boucle sur chaque simulation
            data = self.dataset[sim, idx_start:idx_end, :].clone()

            sim_rolling_data = []
            if training:
                if overlap:
                    for i in range(len(data), rolling_window + 1, -1):
                        sim_rolling_data.append((data[i - rolling_window - 1: i - 1, :], data[i - rolling_window: i, :]))
                else:
                    for i in range(len(data), rolling_window + 1, -rolling_window):
                        sim_rolling_data.append((data[i - rolling_window - 1: i - 1, :], data[i - rolling_window: i, :]))
            else:
                for i in range(idx_start, idx_end):
                    sim_rolling_data.append(data[i - rolling_window: i, :])

            rolling_data.append(sim_rolling_data[::-1])

        return rolling_data
    
    # à supprmier et faut dé-commenter la premiere ligne de la fct loader period
    def load_training_periods(self, initial_train_years=4, retrain_years=2):
        """Charge les périodes d'entraînement et de test pour toutes les simulations."""
        self.periods_train, self.periods_invest = self._generate_training_periods(initial_train_years, retrain_years)

    def loader_period(self, period_index=0, rolling_window=50, batch_size=32, overlap=True, shuffle=True, verbose=False):
        """
        Crée un DataLoader pour toutes les simulations sur une période donnée.

        Retourne :
        - `dataloader` : DataLoader PyTorch pour l'entraînement.
        - `X_test` : Tenseur contenant les données de test.
        - `(start_training, end_training, start_invest, end_invest)` : Périodes correspondantes.
        """
        #self.periods_train, self.periods_invest = self._generate_training_periods(initial_train_years, retrain_years)
        start_training, end_training = self.periods_train[period_index]
        start_invest, end_invest = self.periods_invest[period_index]

        if verbose:
            print(f'Training period from {start_training} to {end_training}')
            print(f'Investment period from {start_invest} to {end_invest}')

        # Génération des données d'entraînement et de test pour toutes les simulations
        data_training = self._compute_data(start=start_training, end=end_training, rolling_window=rolling_window, training=True, overlap=overlap)
        data_invest = self._compute_data(start=start_invest, end=end_invest, rolling_window=rolling_window, training=False)

        # Conversion en tenseurs PyTorch
        X_tensor = torch.cat([torch.tensor([df[0].numpy() for df in sim_data], dtype=torch.float32) for sim_data in data_training], dim=0)
        Y_tensor = torch.cat([torch.tensor([df[1].numpy() for df in sim_data], dtype=torch.float32) for sim_data in data_training], dim=0)
        X_test = torch.cat([torch.tensor([df.numpy() for df in sim_data], dtype=torch.float32) for sim_data in data_invest], dim=0)

        # shapes : 
        # X_tensor.shape = (n_simul * n_rolling_windows, rolling_window, n_assets)
        # Y_tensor.shape = (n_simul * n_rolling_windows, rolling_window, n_assets)
        # simulations stacked one next to the other
        # X_test.shape = (n_simul * n_rolling_windows, rolling_window, n_assets)
        dataset = TensorDataset(X_tensor, Y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return dataloader, X_test, (start_training, end_training, start_invest, end_invest)


        



