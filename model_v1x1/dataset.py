import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from model_v1x1 import input

dataframe = pd.read_csv(input.stock_file_loc, usecols=['Date', 'Close'],
                        index_col='Date', parse_dates=True)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(dataframe)

train_length = int(input.train_fraction * len(scaled_data))
train_data = scaled_data[0:train_length, :]
test_data = scaled_data[train_length:len(scaled_data), :]


def create_xy_data(dataset, timestep):
    x_data, y_data = [], []
    for i in range(len(dataset) - timestep - 1):
        x_data.append(dataset[i:(i + timestep), 0])
        y_data.append(dataset[i + timestep, 0])
    x_data = np.array(x_data)
    x_data = x_data.reshape((x_data.shape[0], 1, x_data.shape[1]))
    y_data = np.array(y_data)
    return x_data, y_data
