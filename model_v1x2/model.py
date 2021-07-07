from model_v1x2 import dataset, input
from keras.models import Sequential
from keras.layers import LSTM, Dense

scaled_data = dataset.get_scaled_data(dataset.dataframe)
data_train, data_test = dataset.train_test_data(scaled_data)
x_train, y_train = dataset.get_xy_data(data_train, input.timestep)
x_test, y_test = dataset.get_xy_data(data_test, input.timestep)

model = Sequential()
model.add(LSTM(256, input_shape=(1, 1)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=input.epochs, verbose=1, batch_size=input.batch_size)
score = model.evaluate(x_train, y_train, verbose=0)

train_prediction = model.predict(x_train)
test_prediction = model.predict(x_test)

train_prediction = dataset.scaler.inverse_transform(train_prediction)
y_train = dataset.scaler.inverse_transform([y_train])

test_prediction = dataset.scaler.inverse_transform(test_prediction)
y_test = dataset.scaler.inverse_transform([y_test])
