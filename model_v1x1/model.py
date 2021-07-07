from model_v1x1 import dataset, input
from keras.layers import Dense, LSTM
from keras.models import Sequential

train_data = dataset.train_data
test_data = dataset.test_data
x_train, y_train = dataset.create_xy_data(train_data, input.timestep)
x_test, y_test = dataset.create_xy_data(test_data, input.timestep)

model = Sequential()
model.add(LSTM(256, input_shape=(1, 1)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1, verbose=1, batch_size=1)
score = model.evaluate(x_train, y_train, verbose=0)

train_prediction = model.predict(x_train)
test_prediction = model.predict(x_test)

train_prediction = dataset.scaler.inverse_transform(train_prediction)
y_train = dataset.scaler.inverse_transform([y_train])

test_prediction = dataset.scaler.inverse_transform(test_prediction)
y_test = dataset.scaler.inverse_transform([y_test])

print(train_prediction)