import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from tqdm import trange

from markowitz import MaxSharpe


# A QUOI SERT LE SCHEDULER 

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange
from typing import List, Tuple
import pandas as pd
from dataset import DataHandler
#from src.dataset import DataHandler

class PortfolioTrainer:
    def __init__(
            self, 
            model,
            data_handler: DataHandler,
            device: str,
            lr: float = 1e-4,
            weight_decay: float = 0.2,
            # scheduler_gamma: float = 0.5,
            permute_assets: bool = False,
            epochs: int = 200,
            verbose: bool = True
            ) -> None:
        self.model = model
        self.dataset = data_handler.dataset # tenseur 3D (n_simul, n_obs, n_assets)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        #self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=scheduler_gamma)
        self.permute_assets = permute_assets
        self.epochs = epochs
        self.data_handler = data_handler
        self.verbose = verbose

    def _train_epoch(
            self,
            dataloader,
            permute_assets: bool 
        ) -> float:
        """
        Epoque d'entrainement 
        on effectue une permutation pour chaque batch si permutation 
        """
        total_loss = 0 
        count = 0 

        if isinstance(self.model, torch.nn.Module):
            self.model.train() 

        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            # on permute chaque batch au sein d'une epoch  
            if permute_assets:
                perm = torch.randperm(batch_x.shape[-1])
                batch_x = batch_x[:, :, perm]  
                batch_y = batch_y[:, :, perm]
            
            self.optimizer.zero_grad()
            loss = self.model(batch_x, batch_y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            count += 1 

        loss_epoch = total_loss / count
        print(f"Loss_epoch:= {loss_epoch}") 
        return loss_epoch

    def train(self) -> None:
        """
        
        """
        permute_assets = self.permute_assets
        epochs = self.epochs
        data_handler = self.data_handler
        n_period_train = len(data_handler.periods_train) # periods_train liste de tuples
        n_obs, n_assets = data_handler.n_dates, data_handler.n_assets
        
        # initialisation des poids 
        self.weights = np.zeros((n_obs, n_assets))

        # On boucle sur le nombre de periode d'entrainement 
        # a chaque periode on calcule la derniere allocation de la periode de test 

        

        for i in range(n_period_train):
            # AJOUTER LES LOGS 

            # on récupère le gestionaire de batch et le X_test pour la période 
            dataloader, X_test, periods = data_handler.load_period(i) #
            
            # entraîne toutes les epochs au sein d'une periode de train 
            # récupère les loss sur chaque époque
            loss_epochs = [
                self._train_epoch(dataloader, permute_assets)
                for _ in (trange(epochs) if self.verbose else range(epochs)) 
            ]
            
            # on récupère le dernier poids de la fenêtre 
            with torch.no_grad():
                alloc_test = self.model.get_alloc_last(X_test.to(self.device)).cpu()


            start_invest_period_i, end_invest_period_i = data_handler.periods_invest[i]
            
            if i == 0 : 
                self.weights[:start_invest_period_i, :] = np.ones(n_assets) / n_assets

            self.weights[start_invest_period_i:end_invest_period_i, :] = alloc_test


        #### AJOUT DU MARKOWITZ #### 

        dataset_marko = self.dataset.dataset.squeeze(0).numpy()
        batch_size = data_handler.batch_size 
        model_marko = MaxSharpe(batch_size=batch_size, overlap=1, max_sharpe=True)
        model_marko.train(dataset_marko)
        self.weights_markowitz = model_marko.weights

        #############  

def simulate_rets_portfolio(self, returns: np.ndarray) -> np.ndarray:
        """
        Calcule le rendement du portefeuille jour par jour en utilisant self.weights,
        selon la convention B (on applique les poids du jour t-1 au jour t).
        """
        n_obs, _ = returns.shape
        port_rets = np.zeros(n_obs)
        port_rets[0] = returns[0] @ self.weights[0, :]
        
        for t in range(1, n_obs):
            port_rets[t] = returns[t] @ self.weights[t - 1, :]

        return port_rets





    


