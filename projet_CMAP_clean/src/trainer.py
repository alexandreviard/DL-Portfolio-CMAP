import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from tqdm import trange


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
    def __init__(self, model, data_handler: DataHandler, device: str = "cuda", lr: float = 0.001, weight_decay: float = 0.2, scheduler_gamma: float = 0.5, permute_assets: bool = False, epochs: int = 200):
        """
        Classe responsable de l'entraînement du modèle de portefeuille.

        Paramètres :
        - model : Modèle PyTorch
        - device : "cuda" ou "cpu"
        - lr : Taux d'apprentissage
        - weight_decay : Régularisation L2 
        - scheduler_gamma : Réduction du taux d’apprentissage 
        """
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=scheduler_gamma)
        self.permute_assets = permute_assets
        self.epochs = epochs
        self.dataset = data_handler.dataset 
        self.data_handler = data_handler
        self.result = self.train(epochs=epochs)

    def _train_epoch(self, dataloader: DataLoader, permute_assets) -> float:
        """Effectue une époque d'entraînement"""
        total_loss = 0
        count = 0
        self.model.train()

        for batch_x, batch_y in dataloader:
            print(f"Shape of batch_x: {batch_x.shape}")  # Debugging
            print(f"Shape of batch_y: {batch_y.shape}")  # Debugging
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

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

        return total_loss / count if count > 0 else 0

    def train(self, epochs=200, verbose=False):
        """
        Train the model over multiple training periods.

        Parameters:
        - epochs : int, Number of training epochs
        - verbose : bool, Show training progress
        - permute_assets : bool, Whether to permute asset order during training

        Returns:
        - result : pd.DataFrame, Portfolio returns with allocations
        """
        self.logs = {}
        self.model.to(self.device)

        result = self.dataset.returns().copy()

        # Add allocation columns to the result DataFrame
        for col in result.columns:
            if not col.endswith("_alloc"):
                result[f"{col}_alloc"] = np.nan

        for i in range(len(self.data_handler.periods_train)):  # Using existing DataHandler
            self.logs[i] = {}

            dataloader, X_test, periods = self.data_handler.loader_period(i)

            # Train for multiple epochs
            loss_epochs = [self._train_epoch(dataloader, self.permute_assets) for _ in (trange(epochs) if verbose else range(epochs))]
            self.scheduler.step()

            # Compute allocations
            with torch.no_grad():
                alloc_test = self.model.get_alloc_last(X_test.to(self.device)).cpu()

            original_data = self.dataset.returns()
            filtered_data = original_data[(original_data.index >= periods[2]) & (original_data.index < periods[3])]

            alloc_columns = [f"{col}_alloc" for col in result.columns if not col.endswith("_alloc")]
            returns_columns = [col for col in result.columns if not col.endswith("_alloc")]

            result.loc[filtered_data.index, alloc_columns] = alloc_test.numpy()
            self.logs[i]["loss"] = loss_epochs

        # Compute portfolio returns
        result['return_pf'] = sum(result[alloc_col] * result[ret_col] for alloc_col, ret_col in zip(alloc_columns, returns_columns))

        return result