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

class PortfolioTrainer:
    def __init__(self, model, device: str = "cuda", lr: float = 0.001, weight_decay: float = 0.2, scheduler_gamma: float = 0.5):
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

    def _train_epoch(self, dataloader: DataLoader) -> float:
        """Effectue une époque d'entraînement"""
        total_loss = 0
        count = 0
        self.model.train()

        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            self.optimizer.zero_grad()

            loss = self.model(batch_x, batch_y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            count += 1

        return total_loss / count if count > 0 else 0

    def train(self, dataset: torch.Tensor, periods: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]], 
              date_range: pd.DatetimeIndex, epochs: int = 200, batch_size: int = 64, rolling_window: int = 50, verbose: bool = False):
        """
        Entraîne le modèle sur plusieurs périodes.

        Paramètres :
        - dataset : torch.Tensor contenant les données (n_simul, n_dates, n_assets)
        - periods : Liste des périodes d'entraînement/test [(train_start, train_end, test_start, test_end)]
        - date_range : Index des dates correspondant à dataset
        - epochs : Nombre d’époques d'entraînement
        - batch_size : Taille des batchs pour l'entraînement
        - rolling_window : Nombre de jours pour la validation
        - verbose : Affiche les logs d'entraînement
        """

        self.model.to(self.device)

        for train_start, train_end, test_start, test_end in periods:
            train_idx_start, train_idx_end = date_range.get_loc(train_start), date_range.get_loc(train_end)
            test_idx_start, test_idx_end = date_range.get_loc(test_start), date_range.get_loc(test_end)

            train_data = dataset[:, train_idx_start:train_idx_end, :]
            test_data = dataset[:, test_idx_start:test_idx_end, :]

            # Créer un DataLoader
            tensor_dataset = TensorDataset(train_data, test_data)
            dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True)

            # Entraînement
            loss_epochs = []
            for _ in trange(epochs) if verbose else range(epochs):
                loss_epochs.append(self._train_epoch(dataloader))

            self.scheduler.step()

        print("✅ Entraînement terminé avec succès !")

    # def get_alloc_last(self, x: torch.Tensor) -> torch.Tensor:
    #     """Récupère la dernière allocation du modèle"""
    #     with torch.no_grad():
    #         return self.model._get_alloc(x)[:, -1, :]
