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
    def __init__(self, model, data_handler: DataHandler, device: str = "cuda", lr: float = 0.001, weight_decay: float = 0.2, scheduler_gamma: float = 0.5, permute_assets: bool = False, epochs: int = 200, verbose: bool = True):
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
        self.verbose = verbose 
        self.result = self.train(epochs=epochs)

    def _train_epoch(self, dataloader: DataLoader, permute_assets) -> float:
        """Effectue une époque d'entraînement"""
        total_loss = 0
        count = 0
        self.model.train()

        for batch_x, batch_y in dataloader:
            #print(f"Shape of batch_x: {batch_x.shape}")  # Debugging
            #print(f"Shape of batch_y: {batch_y.shape}")  # Debugging
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

        loss_epoch =total_loss / count
        print(f"Loss_epoch:= {loss_epoch}") 
        return loss_epoch

    def train(self, epochs=200):
        """
        Train the model over multiple training periods and store allocations.

        Parameters:
        - epochs : int, Number of training epochs

        Returns:
        - result : pd.DataFrame, Portfolio allocations only
        """
        self.logs = {}
        self.model.to(self.device)

        # Dictionary to store allocations per simulation
        allocations_dict = {sim: pd.DataFrame() for sim in range(self.data_handler.n_simul)}

        # Generate an empty DataFrame for allocations
        result = pd.DataFrame(index=self.data_handler.date_range)
        asset_names = self.dataset.returns().columns  

        # Add allocation columns for each simulation
        alloc_columns = {sim: [f"{col}_alloc_sim{sim+1}" for col in asset_names] for sim in range(self.data_handler.n_simul)}
        for sim in alloc_columns:
            for col in alloc_columns[sim]:
                result[col] = np.nan  # Initialize as NaN

        # Iterate over each training period
        for i in range(len(self.data_handler.periods_train)):
            self.logs[i] = {}

            # Load training data & test data
            dataloader, X_test, periods = self.data_handler.loader_period(i)

            # Train for multiple epochs
            loss_epochs = [self._train_epoch(dataloader, self.permute_assets) for _ in (trange(epochs) if self.verbose else range(epochs))]
            self.scheduler.step()

            # Compute allocations for the test period
            with torch.no_grad():
                alloc_test = self.model.get_alloc_last(X_test.to(self.device)).cpu()

            print(f"alloc_test shape: {alloc_test.shape}")  # Debugging

            # Identify the test period start and end
            start_invest, end_invest = periods[2], periods[3]

            if self.data_handler.is_synthetic:
                # Get index for synthetic dataset
                filtered_index = (self.data_handler.date_range >= start_invest) & (self.data_handler.date_range < end_invest)
                filtered_dates = self.data_handler.date_range[filtered_index]
            else:
                # Get index for real dataset
                filtered_dates = result.index[(result.index >= start_invest) & (result.index < end_invest)]

            print(f"Filtered dates count: {len(filtered_dates)}, Expected alloc_test count: {alloc_test.shape[0]}")  # Debugging
            
            # number of test_days = len(filtered_dates)
            n_test_days = len(filtered_dates)
            n_simul = self.data_handler.n_simul

            # Ensure `alloc_test` shape matches `n_simul * n_test_days`
            if alloc_test.shape[0] != n_simul * n_test_days:
                print("⚠️ WARNING: Unexpected mismatch in allocation shape!")

            alloc_test = alloc_test.view(n_simul, n_test_days, -1)
            print(f"alloc_test shape: {alloc_test.shape}")

            # Assign each simulation’s allocations sequentially
            for sim in range(n_simul):
                sim_dates = filtered_dates  # Dates remain the same for this period
                sim_alloc = alloc_test[sim].numpy()  # Get allocations for this simulation

                # Create DataFrame for this simulation's allocations
                sim_alloc_df = pd.DataFrame(sim_alloc, index=sim_dates, columns=alloc_columns[sim])

                # **Instead of overwriting, we append allocations for each simulation**
                allocations_dict[sim] = pd.concat([allocations_dict[sim], sim_alloc_df])

            self.logs[i]["loss"] = loss_epochs  # Save loss for debugging

        # Merge all simulations' allocations into a single result DataFrame
        result = pd.concat(allocations_dict.values(), axis=1)

        return result  # Return dataset with sequential allocations for each simulation