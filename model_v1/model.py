from model_v1 import dataset
from keras.models import Sequential
from keras.layers import LSTM, Dense

netflix = dataset.get_data('NFLX', '2016-01-01', '2021-04-01')
dataset.change_index(netflix)
scaled_data = dataset.get_scaled_data(netflix)
netflix_train, netflix_test = dataset.create_test_train_dataset(scaled_data)
x_train, y_train = dataset.create_xy_dataset(netflix_train, 1)
x_test, y_test = dataset.create_xy_dataset(netflix_test, 1)

model = Sequential()
model.add(LSTM(256, input_shape=(1, 1)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=150, verbose=1, batch_size=1)
score = model.evaluate(x_train, y_train, verbose=0)

train_prediction = model.predict(x_train)
test_prediction = model.predict(x_test)

train_prediction = dataset.scaler.inverse_transform(train_prediction)
y_train = dataset.scaler.inverse_transform([y_train])

test_prediction = dataset.scaler.inverse_transform(test_prediction)
y_test = dataset.scaler.inverse_transform([y_test])
