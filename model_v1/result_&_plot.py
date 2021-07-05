from model_v1 import model
from model_v1.dataset import scaler
import numpy as np
import matplotlib.pyplot as plt

train_predict_plot = np.empty_like(model.scaled_data)
train_predict_plot[:, :] = np.nan
train_predict_plot[1:len(model.train_prediction) + 1, :] = model.train_prediction

test_predict_plot = np.empty_like(model.scaled_data)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(model.train_prediction) + 3: len(model.scaled_data)-1, :] = model.test_prediction

plt.plot(scaler.inverse_transform(model.scaled_data))
plt.plot(train_predict_plot)
plt.plot(test_predict_plot)
plt.legend()
plt.show()