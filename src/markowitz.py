import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import scipy.optimize as opt 
import torch


class MaxSharpe():
    def __init__(self, risk_aversion:float = 1, batch_size:int = 100, overlap:int = 1, max_sharpe:bool = True) -> None:
        self.risk_aversion = risk_aversion
        self.overlap = overlap
        self.batch_size = batch_size
        self.max_sharpe = max_sharpe

    def _compute_weights(self, batch:np.ndarray, method:str='sharpe') -> np.ndarray:
        """ 
        Computes the markowitz solution over the batch 
        data is assumed to be on a daily basis 
        therefore we annualize the mean and the cov 
        """
        mean_vector = np.mean(batch, axis=0)
        cov_matrix = np.cov(batch, rowvar=False)
        if method=='sharpe':
            w = self._max_sharpe_opt(cov_matrix, mean_vector)
        elif method=="sharpe_torch" :
            w = self._max_sharpe_torch(cov_matrix, mean_vector)
        elif method=="marko" :
            w = self._markowitz_opt(cov_matrix, mean_vector)
        return w 
    
    def _markowitz_opt(self, cov_matrix:np.ndarray, mean_vector:np.ndarray) -> float:
        """
        solves the Markowitz problem with scipy 
        weights constraints : equal to one, no short selling
        """
        def target_func(weights:np.ndarray) -> float:
            f = mean_vector @ weights - (self.risk_aversion / 2) * weights.T @ cov_matrix @ weights
            return -f # on veut un problème de minimisation

        n_assets = len(mean_vector)  
        
        constraints = [{'type': 'eq', 'fun': lambda x : np.sum(x) - 1}] 
        bounds = [(0, None)] * n_assets

        x0 = np.ones(n_assets) / n_assets
        res = opt.minimize(
            target_func, 
            x0, 
            constraints=constraints,
            bounds=bounds
            )

        return res.x  
    
    def _max_sharpe_opt(self, cov_matrix:np.ndarray, mean_vector:np.ndarray) -> float:
        
        def target_func(weights:np.ndarray) -> float:
            f = mean_vector @ weights / weights.T @ cov_matrix @ weights
            return -f # on veut un problème de minimisation
        
        n_assets = len(mean_vector)  
        
        constraints = [{'type': 'eq', 'fun': lambda x : np.sum(x) - 1}] 
        bounds = [(0, None)] * n_assets

        x0 = np.ones(n_assets) / n_assets
        res = opt.minimize(
            target_func, 
            x0, 
            constraints=constraints,
            bounds=bounds
            )

        return res.x 
    
    def _max_sharpe_torch(self, cov_matrix, mean_vector):
        class MaxSharpeLongOnly(torch.nn.Module):
            def __init__(self, d):
                    super().__init__()
                    self.w = torch.nn.Parameter(torch.ones(d))

            @property
            def weights(self):
                return torch.softmax(self.w, dim=0)

        def sharpe_loss(weights, cov_matrix, mean_vector):
            returns = torch.einsum("d, d -> ", weights, mean_vector)
            risk    = torch.sqrt(torch.einsum("d, dc, c -> ", weights, cov_matrix, weights))
            sharpe  = returns / risk
            return -sharpe
        d=len(mean_vector)
        model = MaxSharpeLongOnly(d)
        adam  = torch.optim.Adam(model.parameters(), lr=1e-1)

        torch_mu  = torch.from_numpy(mean_vector).float()
        torch_cov = torch.from_numpy(cov_matrix).float()

        num_steps = 100
        sharpe_log = np.zeros(shape=num_steps)
        weights_log = np.zeros(shape=(num_steps, d))

        for s in range(num_steps):
            adam.zero_grad()
            loss = sharpe_loss(model.weights, torch_cov, torch_mu)
            loss.backward()
            adam.step()

            sharpe_log[s]  = -loss
            weights_log[s] = model.weights.detach().numpy()
            
        return weights_log[-1]


    def train(self, returns: np.ndarray) -> None:
        """
        trains the model 
        """
        n_obs, n_assets = returns.shape
        self.weights = np.zeros((n_obs, n_assets))

        # Initialisation des poids (avant la première fenêtre)
        for t in range(self.batch_size):
            self.weights[t, :] = np.ones(n_assets) / n_assets

        # On avance par pas de self.overlap jours :
        for t in range(self.batch_size, n_obs, self.overlap):
            window_returns = returns[t - self.batch_size : t, :]
            w = self._compute_weights(window_returns)

            for day in range(t, min(t + self.overlap, n_obs)):
                self.weights[day, :] = w

    def train_with_permutations(self, returns: np.ndarray) -> None:
        """
        trains the model with permutation of assets at each training 
        """
        n_obs, n_assets = returns.shape
        self.weights = np.zeros((n_obs, n_assets))

        # Initialisation des poids (avant la première fenêtre)
        for t in range(self.batch_size):
            self.weights[t, :] = np.ones(n_assets) / n_assets

        # On avance par pas de self.overlap jours :
        # On permute les assets à chaque calcul de poids 
        index = np.arange(n_assets)
        for t in range(self.batch_size, n_obs, self.overlap):
            window_returns = returns[t - self.batch_size : t, :]
            index = np.random.permutation(index) 
            window_returns_permutation = window_returns[:, index]
            w = self._compute_weights(window_returns_permutation)

            for day in range(t, min(t + self.overlap, n_obs)):
                self.weights[day, :] = w
    
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