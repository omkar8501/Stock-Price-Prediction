from model_v1x0 import model
from model_v1x0.dataset import scaler
import numpy as np
import matplotlib.pyplot as plt

train_predict_plot = np.empty_like(model.scaled_data)
train_predict_plot[:, :] = np.nan
train_predict_plot[1:len(model.train_prediction) + 1, :] = model.train_prediction

test_predict_plot = np.empty_like(model.scaled_data)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(model.train_prediction) + 2 + 1: len(model.scaled_data) - 1, :] = model.test_prediction

plt.plot(model.netflix.index, scaler.inverse_transform(model.scaled_data), label='historical data')
plt.plot(model.netflix.index, train_predict_plot, label='training prediction data')
plt.plot(model.netflix.index, test_predict_plot, label='testing prediction data')
plt.legend()
plt.grid()
plt.show()
