import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from datetime import timedelta
from torch.utils.data import DataLoader, TensorDataset


class NN_Sharpe(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers, model_name='LSTM'):
        super(NN_Sharpe, self).__init__()

        self.num_layers = num_layers
        self.model_name = model_name

        if self.model_name == 'LSTM':
            self.model = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        elif self.model_name == 'RNN':
            self.model = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        else:
            self.model = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
            
        self.linear = nn.Linear(hidden_size, output_size)
    
    def get_alloc(self, x):


        # x : shape[batch_size, seq_length, input_size]

        # output : [batch_size, sequ_length, hidden_size]
        # cn : [num_layers, batch_size, output_size], it's the final state of each layer if proj_size>0 (only for LSTM)
        # hn : [num_layers, batch_size, output_size], it's the final state of each layer

        if self.model_name=="LSTM":
            # if num_layers >1, we only keep the last layer for computing loss

            output, (hn,cn) = self.model(x)
            output, (hn,cn) = output[:,:,:], (hn[:, :, :], cn[:, : , :])
            
        elif self.model_name in ['GRU', 'RNN']:

            output, hn = self.model(x)
            output, hn = output[:,:,:], hn[:, :, :]


        # shape of weights = [batch_size, seq_length, output_size]
        unnormalized_weights = self.linear(output) 
        normalized_weights = torch.softmax(unnormalized_weights, dim=-1)

        return normalized_weights
    
    def get_alloc_last(self, x):

        x = self.get_alloc(x)
        return x[:, -1, :]
    

    def sharpe_loss(self, weights, y):

        # weights : [batch_size, seq_length, output_size]
        # y : [batch_size, seq_length, output_size]

        # returns : [batch_size, seq_length]
        returns = torch.sum(weights*y, dim=-1)

        # mean_returns/std_returns : [batch_size]
        mean_returns = returns.mean(dim=-1)
        std_returns = returns.std(dim=-1) + 1e-12

        mean_sharpe_ratio = mean_returns/std_returns # shape [batch_size]

        return -mean_sharpe_ratio #maximizing the sharpe ratio
    
    def forward(self, x, y):

        x = self.get_alloc(x)
        loss = self.sharpe_loss(x, y).mean()

        return loss