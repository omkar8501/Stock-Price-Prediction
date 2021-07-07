from model_v1x1 import model, dataset
import numpy as np
import matplotlib.pyplot as plt

train_predict_plot = np.empty_like(dataset.scaled_data)
train_predict_plot[:, :] = np.nan
train_predict_plot[1:len(model.train_prediction) + 1, :] = model.train_prediction

test_predict_plot = np.empty_like(dataset.scaled_data)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(model.train_prediction) + 2 + 1: len(dataset.scaled_data) - 1, :] = model.test_prediction

# plt.plot(dataset.dataframe.index, dataset.scaler.inverse_transform(dataset.scaled_data), label='historical value')
# plt.plot(dataset.dataframe.index, train_predict_plot, label='train prediction')
# plt.plot(dataset.dataframe.index, test_predict_plot, label='test prediction')
# plt.grid()
# plt.legend()
# plt.show()

print(train_predict_plot)
