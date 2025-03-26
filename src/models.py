import torch
import torch.nn as nn
import cvxpy as cp

from typing import Union

class NN_Sharpe(nn.Module):
    def __init__(
        self,
        input_size: int = 2,
        hidden_size: int = 64,
        output_size: int = 2,
        num_layers: int = 1,
        model_name: str = 'LSTM',
        temperature: float = 0.1
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.model_name = model_name
        self.temperature = temperature

        # A DISCUTER - PEUT ETRE RETIRER LE CHOIX 
        rnn_dict = {
            'LSTM': nn.LSTM,
            'RNN': nn.RNN,
            'GRU': nn.GRU
        }

        if model_name not in rnn_dict:
            raise ValueError("model_name doit être parmi ['LSTM', 'RNN', 'GRU']")

        self.model = rnn_dict[model_name](
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_size, output_size)
        self.temperature = temperature


    def get_alloc(self, x: torch.Tensor) -> torch.Tensor:
        if self.model_name == "LSTM":
            output, (hn,cn) = self.model(x)

        elif self.model_name in ['GRU', 'RNN']:
            output, hn = self.model(x)

        unnormalized_weights = self.linear(output)
        scaled_weights = unnormalized_weights / self.temperature

        normalized_weights = torch.softmax(scaled_weights, dim=-1)
        return normalized_weights

    def get_alloc_last(self, x: torch.Tensor) -> torch.Tensor:
        alloc = self.get_alloc(x)
        return alloc[:, -1, :]

    def sharpe_loss(
        self,
        weights: torch.Tensor,
        returns: torch.Tensor,
        eps: float = 1e-12
    ) -> torch.Tensor:
        # weights : [batch_size, seq_length, output_size]
        #print(weights.shape)
        #print(returns.shape)
        weighted_returns = (weights * returns).sum(dim=-1)
        mean_returns = weighted_returns.mean(dim=-1)
        std_returns = weighted_returns.std(dim=-1) + eps

        return -mean_returns / std_returns

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        allocations = self.get_alloc(x)
        loss_batch = self.sharpe_loss(allocations, y)
        return loss_batch.mean()
