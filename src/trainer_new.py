import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from tqdm import trange
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from src.markowitz import MaxSharpe
from typing import Union, List, Dict, Tuple


# A QUOI SERT LE SCHEDULER 

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange
from typing import List, Tuple
import pandas as pd
from src.dataset import DataHandler
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
        self.weight_decay = weight_decay
        self.model = model
        self.dataset = data_handler.dataset # tenseur 3D (n_simul, n_obs, n_assets)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        #self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=scheduler_gamma)
        self.permute_assets = permute_assets
        self.epochs = epochs
        self.data_handler = data_handler
        self.verbose = verbose
        self.markowitz = MaxSharpe()
        self.on_synthetic = self.data_handler.on_synthetic
        self.batch_size = self.data_handler.batch_size
        if self.on_synthetic:
            self.th_weights_sharpe = self.markowitz._max_sharpe_torch(self.data_handler.dataset.Sigma, self.data_handler.dataset.mu)
            self.th_weights_marko = self.markowitz._markowitz_opt(self.data_handler.dataset.Sigma, self.data_handler.dataset.mu)

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
            if permute_assets:

                perm = torch.randperm(self.data_handler.n_assets)
                batch_x = batch_x[:, :, perm]  
                batch_y = batch_y[:, :, perm]
            
            self.optimizer.zero_grad()
            loss = self.model(batch_x, batch_y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            count += 1 

        loss_epoch = total_loss / count
        
        return loss_epoch

    def train(self, compute_marko_weights: List=None) -> None:

        permute_assets = self.permute_assets
        epochs = self.epochs
        data_handler = self.data_handler
        n_period_train = len(data_handler.periods_train) # periods_train liste de tuples
        n_assets = data_handler.n_assets
        n_obs = data_handler.n_obs
        n_simul = data_handler.n_simul
        self.nb_wsim_computed = min(8, n_simul)
        # initialisation des poids 
        self.weights_model = np.zeros((n_simul, n_obs, n_assets))
        
        if compute_marko_weights:
             #on compute les weights pour max 8 simulations
            if 'sharpe' in compute_marko_weights:
                self.weights_sharpe = np.zeros((self.nb_wsim_computed, n_obs, n_assets))
            if 'marko' in compute_marko_weights:
                self.weights_markow = np.zeros((self.nb_wsim_computed, n_obs, n_assets))
            if 'sharpe_torch' in compute_marko_weights:
                self.weights_sharpe = np.zeros((self.nb_wsim_computed, n_obs, n_assets))
                

        # On boucle sur le nombre de periode d'entrainement 
        # a chaque periode on calcule la dern
        # iere allocation de la periode de test 

        

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
            
            start, end = periods[2], periods[3]
            print(np.mean(loss_epochs))
            
            # on récupère le dernier poids de la fenêtre 
            with torch.no_grad():
                alloc_test = self.model.get_alloc_last(X_test[0:self.nb_wsim_computed*(end-start)].to(self.device)).cpu() # (n_simul*(start-end), n_assets)
                self.alloc = alloc_test
                
            if compute_marko_weights:
                ws_sharpe = []
                ws_marko = []
                print(X_test.shape)
                self.xtest = X_test
                
                print(self.nb_wsim_computed*(end-start), self.nb_wsim_computed, end, start)
                for j in trange(self.nb_wsim_computed*(end-start)): #on compute pas pour + de 8 simulations (trop couteux)
                    if 'sharpe' in compute_marko_weights:
                        w_sharpe = self.markowitz._compute_weights(batch=X_test[j].numpy(), method='sharpe')
                        ws_sharpe.append(w_sharpe)
                    if 'sharpe_torch' in compute_marko_weights:
                        w_sharpe = self.markowitz._compute_weights(batch=X_test[j].numpy(), method='sharpe_torch')
                        ws_sharpe.append(w_sharpe)
                    if 'marko' in compute_marko_weights:
                        w_marko = self.markowitz._compute_weights(batch=X_test[j].numpy(), method='marko')
                        ws_marko.append(w_marko)
                
                ws_sharpe = np.asarray(ws_sharpe)
                ws_marko = np.asarray(ws_marko)
        
                
            
            for sim in trange(self.nb_wsim_computed):
                self.weights_model[sim, start:end, :] = alloc_test[sim*(end-start) : (sim+1)*(end-start), :].numpy()
                if compute_marko_weights:
                    self.weights_sharpe[sim, start:end, :] = ws_sharpe[sim*(end-start) : (sim+1)*(end-start), :]
                    self.weights_markow[sim, start:end, :] = ws_marko[sim*(end-start) : (sim+1)*(end-start), :]


    def plot_weights(self, type_w='model', th_weights=None):
        
        if not hasattr(self, 'weights_model'):
            raise ValueError("Le modèle doit d'abord être entraîné pour accéder aux poids")

        if type_w == 'model':
            weights = self.weights_model
        elif type_w == 'sharpe':
            weights = self.weights_sharpe
        elif type_w == 'marko':
            weights = self.weights_markow
        else:
            raise ValueError("type_w doit être 'model', 'sharpe' ou 'marko'")

        if self.on_synthetic:

            ncols = 4
            nrows = int(np.ceil(self.nb_wsim_computed / ncols))
            width_per_subplot = 4
            height_per_subplot = 3
            figsize = (ncols * width_per_subplot, nrows * height_per_subplot)
            fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)
            ax = ax.flatten()

            for sim in range(self.nb_wsim_computed):
                w = weights[sim]
                T = w.shape[0]
                time = np.arange(T)
                ax[sim].stackplot(time, w.T, labels=self.data_handler.dataset.tickers)
                ax[sim].set_title(f"Sim {sim}")
                ax[sim].set_xlabel("Time")
                ax[sim].set_ylabel("Poids")
                ax[sim].legend(prop={'size':7})

                if th_weights=='sharpe':
                    cumsum_weights = self.th_weights_sharpe.copy()
                    cumsum_weights.sort()
                    cumsum_weights = cumsum_weights.cumsum()
                    self.tata = cumsum_weights
                    for w in cumsum_weights:
                        ax[sim].axhline(w, color='black', linewidth=2, linestyle="--")
                        
                elif th_weights=='marko':
                    cumsum_weights = self.th_weights_marko.copy()
                    cumsum_weights.sort()
                    cumsum_weights = cumsum_weights.cumsum()
                    self.tata = cumsum_weights
                    for w in cumsum_weights:
                        ax[sim].axhline(w, color='black', linewidth=2, linestyle="--")
                    
            if type_w == 'model':
                fig.suptitle(f"Poids pour {type_w}", fontsize=16)

                axes_positions = [ax.get_position() for ax in fig.get_axes()]
                max_y1 = max(pos.y1 for pos in axes_positions) if axes_positions else 0.9
                text_y = min(max_y1 + 0.022, 0.95)
                fig.subplots_adjust(top=text_y - 0.03)
                params_text = (
                    f"hidden_size: {self.model.hidden_size}, batch_size: {self.batch_size}, num_layers: {self.model.num_layers}, permute : {self.permute_assets}\n"
                    f"rolling_window: {self.data_handler.rolling_window}, overlap: {self.data_handler.overlap}, epochs : {self.epochs}, "
                    f"shuffle: {self.data_handler.shuffle}, weight_decay: {self.weight_decay}"
                )
                fig.text(0.5, text_y, params_text, ha='center', fontsize=10)

            elif type_w == 'marko':
                weights_str = '\n'.join(
                    f'{ticker} : {weight:.2%}' 
                    for ticker, weight in zip(self.data_handler.dataset.tickers, self.th_weights_marko)
                )
                fig.suptitle(f'Poids pour {type_w}\nPoids optimaux théoriques :\n{weights_str}')

            elif type_w == 'sharpe':
                weights_str = '\n'.join(
                    f'{ticker} : {weight:.2%}' 
                    for ticker, weight in zip(self.data_handler.dataset.tickers, self.th_weights_sharpe)
                )
                fig.suptitle(f'Poids pour {type_w}\nPoids optimaux théoriques :\n{weights_str}')
                            
            for i in range(self.nb_wsim_computed, len(ax)):
                fig.delaxes(ax[i])
            
            fig.tight_layout()

        else:
            w = weights[0] 
            T = w.shape[0]
            time = np.arange(T)
            plt.figure(figsize=(10, 5))
            plt.stackplot(time, w.T)
            plt.title("Poids du portefeuille sur données réelles")
            plt.xlabel("Time")
            plt.ylabel("Poids")
            plt.tight_layout()
            plt.show()
        
        

        

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
