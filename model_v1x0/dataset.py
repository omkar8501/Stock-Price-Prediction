import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
import numpy as np

train_data_fraction = 0.95
scaler = MinMaxScaler()


def get_data(stock_symbol, start_date, end_date):
    base_url = f'https://api.tiingo.com/tiingo/daily/{stock_symbol}/prices?'
    api_token = '08e61d77a015c0708723e8d49884c435c5bb5e86'
    payload = {
        'startDate': start_date,
        'endDate': end_date,
        'token': api_token
    }
    response_url = requests.get(base_url, params=payload)
    dataframe = pd.DataFrame.from_dict(response_url.json())
    dataframe = dataframe.drop(
        ['open', 'high', 'low', 'adjOpen', 'adjHigh', 'adjLow', 'adjClose', 'volume', 'adjVolume',
         'divCash', 'splitFactor'], axis=1)
    return dataframe


def change_index(dataframe):
    dataframe['date'] = pd.to_datetime(dataframe['date'])
    dataframe = dataframe.set_index('date', inplace=True)
    return dataframe


def get_scaled_data(dataframe):
    data_scaled = scaler.fit_transform(dataframe)
    return data_scaled


def create_test_train_dataset(data_scaled):
    train_length = int(len(data_scaled) * train_data_fraction)
    train_data = data_scaled[0:train_length, :]
    test_data = data_scaled[train_length:len(data_scaled), :]
    return train_data, test_data


def create_xy_dataset(dataset, timestep):
    x_data, y_data = [], []
    for i in range(len(dataset) - timestep - 1):
        x_data.append(dataset[i:(i + timestep), 0])
        y_data.append(dataset[i + timestep, 0])
    x_data = np.array(x_data)
    x_data = x_data.reshape((x_data.shape[0], 1, x_data.shape[1]))
    y_data = np.array(y_data)
    return x_data, y_data
