import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from datetime import timedelta
from torch.utils.data import DataLoader, TensorDataset
from models.base import NN_Sharpe

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

    for _ in range(n_periods-1):

        end_training_date = data.index[data.index.get_loc(training_date)+retrain_years*252]
        invest_date, end_invest_date = end_training_date, data.index[data.index.get_loc(end_training_date)+retrain_years*252]

        training_periods.append((training_date, end_training_date))
        test_periods.append((invest_date, end_invest_date))

        #print(training_date, end_training_date)
        #print(invest_date, end_invest_date)

        training_date = invest_date


    return training_periods, test_periods

def compute_data(data, start, end, is_training=True, overlap=True, rolling_window=50):

    rolling_data = []
    idx_start, idx_end = data.index.get_loc(start), data.index.get_loc(end)
    
    if is_training: 
        data = data.iloc[idx_start:idx_end, :].copy()

        if overlap:

            for i in range(len(data), rolling_window+1, -1):
                    
                rolling_data.append((data.iloc[i - rolling_window - 1: i-1, :], data.iloc[i - rolling_window: i, :]))

            return rolling_data[::-1]
        
        else:

            for i in range(len(data), rolling_window+1, -rolling_window):
                    
                rolling_data.append((data.iloc[i - rolling_window - 1: i-1, :], data.iloc[i - rolling_window: i, :]))

            return rolling_data[::-1]
    

    else:
        
        for i in range(idx_start, idx_end):
                
            rolling_data.append((data.iloc[i-rolling_window: i, :]))

        return rolling_data
    

def training_with_model(data_used, model, optimizer, initial_train_years=4, retrain_years=2, rolling_window=50, 
              shuffle=False, batch_size=64, epoch=50, overlap=True):
    
    # We make a copy of the datafram
    result = data_used.copy()
    for col in result.columns:
        alloc_col = f"{col}_alloc"
        result[alloc_col] = np.nan

    #model = NN_Sharpe(input_size, hidden_size, output_size, num_layers, model_name)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.2)
    
    #scheduler_global = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    periods_train, periods_invest = generate_training_periods(data_used, initial_train_years, retrain_years)

    for i in range(len(periods_train)):

        start_training, end_training = periods_train[i][0], periods_train[i][1]
        start_invest, end_invest = periods_invest[i][0], periods_invest[i][1]

        print(f'training from {start_training} to {end_training}')
        print(f'invest from {start_invest} to {end_invest}')

        data_training = compute_data(data=data_used, start=start_training, end=end_training, rolling_window=rolling_window, is_training=True, overlap=overlap)
        data_invest = compute_data(data=data_used, start=start_invest, end=end_invest, is_training=False, rolling_window=rolling_window)

        X_tensor = torch.tensor([df[0].values for df in data_training], dtype=torch.float32)
        print(X_tensor.shape)
        Y_tensor = torch.tensor([df[1].values for df in data_training], dtype=torch.float32)
        X_test = torch.tensor([df.values for df in data_invest], dtype=torch.float32)
        dataset = TensorDataset(X_tensor, Y_tensor)

        dataloader = DataLoader(dataset, batch_size, shuffle)
        global_loss = []

        for epoch in range(epoch): 

            loss = []
        
            for _, (batch_x, batch_y) in enumerate(dataloader):
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

        filtered_data = data_used[(data_used.index >= start_invest) & (data_used.index < end_invest)].copy()

        alloc_columns = [f"{col}_alloc" for col in result.columns if not col.endswith("_alloc")]

        returns_columns = [col for col in result.columns if not col.endswith("_alloc")]

        result.loc[filtered_data.index, alloc_columns] = alloc_test.numpy()
        

    rt_pf = 0
    for i in range(len(returns_columns)):
        rt_pf += result[alloc_columns[i]] * result[returns_columns[i]]
    result['return_pf'] = rt_pf

    #return only the columns of the allocations and corresponding pf returns
    selected_columns = [col for col in result.columns if col.endswith("_alloc") or col == "return_pf"]

    return result[selected_columns]

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
    #scheduler_global = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    periods_train, periods_invest = generate_training_periods(data_used, initial_train_years, retrain_years)

    for i in range(len(periods_train)):

        start_training, end_training = periods_train[i][0], periods_train[i][1]
        start_invest, end_invest = periods_invest[i][0], periods_invest[i][1]

        print(f'training from {start_training} to {end_training}')
        print(f'invest from {start_invest} to {end_invest}')

        data_training = compute_data(data=data_used, start=start_training, end=end_training, rolling_window=rolling_window, is_training=True, overlap=overlap)
        data_invest = compute_data(data=data_used, start=start_invest, end=end_invest, is_training=False, rolling_window=rolling_window)

        X_tensor = torch.tensor([df[0].values for df in data_training], dtype=torch.float32)
        print(X_tensor.shape)
        Y_tensor = torch.tensor([df[1].values for df in data_training], dtype=torch.float32)
        X_test = torch.tensor([df.values for df in data_invest], dtype=torch.float32)
        dataset = TensorDataset(X_tensor, Y_tensor)

        dataloader = DataLoader(dataset, batch_size, shuffle)
        global_loss = []

        for epoch in range(epoch): 

            loss = []
        
            for _, (batch_x, batch_y) in enumerate(dataloader):
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
            print('printing alloc_test')
            print(alloc_test)

        filtered_data = data_used[(data_used.index >= start_invest) & (data_used.index < end_invest)].copy()
        print('filtered data')
        print(filtered_data)

        alloc_columns = [f"{col}_alloc" for col in result.columns if not col.endswith("_alloc")]
        print("alloc columns")
        print(alloc_columns)
        returns_columns = [col for col in result.columns if not col.endswith("_alloc")]

        result.loc[filtered_data.index, alloc_columns] = alloc_test.numpy()
        print("filtered_data.index")
        print(filtered_data.index)

        print(f"Result after update (filtered index): {result.loc[filtered_data.index, alloc_columns]}")
        print(f"Results are : {result}")

    rt_pf = 0
    for i in range(len(returns_columns)):
        rt_pf += result[alloc_columns[i]] * result[returns_columns[i]]
    result['return_pf'] = rt_pf


    return result, periods_train, periods_invest