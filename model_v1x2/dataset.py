import pandas as pd
import numpy as np
from model_v1x2 import input
from sklearn.preprocessing import MinMaxScaler
import requests

# url = 'https://api.tiingo.com/tiingo/daily/AMZN/prices?startDate=2005-01-01&endDate=2021-01-01&token=08e61d77a015c0708723e8d49884c435c5bb5e86'
# response = requests.get(url)
# dataframe = pd.DataFrame.from_dict(response.json())
# dataframe = dataframe.drop(
#     ['open', 'high', 'low', 'adjOpen', 'adjHigh', 'adjLow', 'adjClose', 'volume', 'adjVolume',
#      'divCash', 'splitFactor'], axis=1)
# dataframe['date'] = pd.to_datetime(dataframe['date'])
# dataframe = dataframe.set_index('date')

dataframe = pd.read_csv(input.stock_file_loc, usecols=['Date', 'Close'],
                        index_col='Date', parse_dates=True)

scaler = MinMaxScaler()


def get_scaled_data(dataset):
    scaled_data = scaler.fit_transform(dataset)
    return scaled_data


def train_test_data(dataset):
    train_length = int(len(dataset) * input.train_fraction)
    train_data = dataset[0:train_length, :]
    test_data = dataset[train_length:len(dataset), :]
    return train_data, test_data


def get_xy_data(dataset, timestep):
    x_data, y_data = [], []
    for i in range(len(dataset) - timestep - 1):
        x_data.append(dataset[i:i + timestep, 0])
        y_data.append(dataset[i + timestep, 0])
    x_data = np.array(x_data)
    x_data = x_data.reshape((x_data.shape[0], 1, x_data.shape[1]))
    y_data = np.array(y_data)
    return x_data, y_data
