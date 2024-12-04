import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import nbformat

from datetime import timedelta
from torch.utils.data import DataLoader, TensorDataset

"""
We implement a Sharpe Neural Network aiming to bypass the return prediction, to do so we use 3 different architectures:
-  RNN : Recurrent Neural Network
-  LSTM : Long Short-Term Memory
-  GRU : Gated Recurrent Unit

We use return in percent in order to avoid convergence problems when dealing with values close to zero.
"""

class NN_Sharpe(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers, model_name='LSTM'):
        #super(NN_Sharpe, self).__init__() # semble inutile non ? 

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


def generate_training_periods(data, initial_train_years=4, retrain_years=2):

    # We compute each index for the training & test dates (start and end)

    training_periods = []
    test_periods = []

    last_date_1st_training = data.index[initial_train_years*252]
    training_periods.append((data.index[0], last_date_1st_training))

    first_invest_date = last_date_1st_training
    end_date_first_invest = data.index[data.index.get_loc(first_invest_date) + retrain_years*252]

    test_periods.append((first_invest_date, end_date_first_invest))

    n_periods = (data[data.index >= last_date_1st_training].shape[0]) // (retrain_years*252)

    training_date = first_invest_date

    for i in range(n_periods-1):

        training_date, end_training_date = training_date, data.index[data.index.get_loc(training_date)+retrain_years*252]
        invest_date, end_invest_date = end_training_date, data.index[data.index.get_loc(end_training_date)+retrain_years*252]

        training_periods.append((training_date, end_training_date))
        test_periods.append((invest_date, end_invest_date))

        #print(training_date, end_training_date)
        #print(invest_date, end_invest_date)

        training_date = invest_date


    return training_periods, test_periods


def compute_data(data, start, end, training=True, overlap=True, rolling_window=50):

    rolling_data = []
    idx_start, idx_end = data.index.get_loc(start), data.index.get_loc(end)
    
    #print(len(data))
    if training: 
        data = data.iloc[idx_start:idx_end, :].copy()

        if overlap:

            for i in range(len(data), rolling_window+1, -1):
                    
                #print(i-rolling_window, i)
                rolling_data.append((data.iloc[i - rolling_window - 1: i-1, :], data.iloc[i - rolling_window: i, :]))

            return rolling_data[::-1]
        
        else:

            for i in range(len(data), rolling_window+1, -rolling_window):
                    
                #print(i-rolling_window, i)
                rolling_data.append((data.iloc[i - rolling_window - 1: i-1, :], data.iloc[i - rolling_window: i, :]))

            return rolling_data[::-1]
    

    else:
        
        for i in range(idx_start, idx_end):
                
            rolling_data.append((data.iloc[i-rolling_window: i, :]))

        return rolling_data


def training(data_used, input_size,
              hidden_size, output_size, num_layers, model_name, 
              initial_train_years=4, retrain_years=2, rolling_window=50, 
              shuffle=False, batch_size=64, epoch=50, overlap=True):
    
    # We make a copy of the datafram
    result = data_used.copy()
    for col in result.columns:
        alloc_col = f"{col}_alloc"
        result[alloc_col] = np.nan

    model = NN_Sharpe(input_size, hidden_size, output_size, num_layers, model_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.2)
    scheduler_global = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    periods_train, periods_invest = generate_training_periods(data_used, initial_train_years, retrain_years)

    
    for i in range(len(periods_train)):

        start_training, end_training = periods_train[i][0], periods_train[i][1]
        start_invest, end_invest = periods_invest[i][0], periods_invest[i][1]

        print(f'training from {start_training} to {end_training}')
        print(f'invest from {start_invest} to {end_invest}')

        data_training = compute_data(data=returns, start=start_training, end=end_training, rolling_window=rolling_window, training=True, overlap=overlap)
        data_invest = compute_data(data=returns, start=start_invest, end=end_invest, training=False, rolling_window=rolling_window)

        X_tensor = torch.tensor([df[0].values for df in data_training], dtype=torch.float32)
        print(X_tensor.shape)
        Y_tensor = torch.tensor([df[1].values for df in data_training], dtype=torch.float32)
        X_test = torch.tensor([df.values for df in data_invest], dtype=torch.float32)
        dataset = TensorDataset(X_tensor, Y_tensor)

        dataloader = DataLoader(dataset, batch_size, shuffle)
        global_loss = []

        for epoch in range(epoch): 

            loss = []
        
            for k, (batch_x, batch_y) in enumerate(dataloader):
                optimizer.zero_grad()
                outputs = model(batch_x, batch_y)
                loss.append(outputs.item())
                outputs.backward()
                optimizer.step()
            
            loss_epoch = sum(loss) / len(loss)
            print(f'epoch {epoch}, loss = {loss_epoch}')
            global_loss.append(global_loss)


        with torch.no_grad():
            alloc_test = model.get_alloc_last(X_test)

        #scheduler_global.step()

        filtered_data = data_used[(data_used.index >= start_invest) & (data_used.index < end_invest)].copy()

        #print("Filtered data index shape:", len(filtered_data.index))
        #print("Alloc test shape:", alloc_test.numpy().shape)
        #print("Alloc columns:", [f"{col}_alloc" for col in result.columns if not col.endswith("_alloc")])


        alloc_columns = [f"{col}_alloc" for col in result.columns if not col.endswith("_alloc")]
        returns_columns = [col for col in result.columns if not col.endswith("_alloc")]

        result.loc[filtered_data.index, alloc_columns] = alloc_test.numpy()

        #t.append(data_training)
        #j.append(data_invest)

    rt_pf = 0
    for i in range(len(returns_columns)):
        rt_pf += result[alloc_columns[i]] * result[returns_columns[i]]
    result['return_pf'] = rt_pf


    return result



        #for data in data_training:

        #    data_X = torch.tensor(data[0].values, dtype=torch.float32).view(1, rolling_window, input_size)
        #    data_Y = torch.tensor(data[1].values, dtype=torch.float32).view(1, rolling_window, input_size)
        #    print(model(data_X, data_Y))

            
def get_result(dataframe_result, prices):

    fig_prices = px.line(prices[prices.index.isin(dataframe_result.dropna().index)], 
                        title="prices")
    fig_prices.show()

    alloc_columns = [col for col in dataframe_result.columns if col.endswith('_alloc')]
    fig_allocations = px.line(dataframe_result.dropna(), y=alloc_columns, 
                            title="allocations")
    fig_allocations.show()

    N = dataframe_result.dropna().shape[0]
    log_returns_sum = np.sum(np.log(1+dataframe_result['return_pf']/100))
    annual_rt = np.exp((252/N) * log_returns_sum) - 1
    sharpe = dataframe_result['return_pf'].mean() / dataframe_result['return_pf'].std() * np.sqrt(252)

    print(f"nb de jours d'investissement: {N}\nannualized return: {annual_rt}")
    print(f"sharpe ratio: {sharpe}")
    print(f"std deviation: {(dataframe_result['return_pf']/100).std() * np.sqrt(252)}")
    print(f"downside_risk:  { ((dataframe_result[ dataframe_result['return_pf']<0 ]['return_pf']) / 100).std() * np.sqrt(252)} ")




        




        