from numpy import array
from keras.models import Sequential
from keras.layers import LSTM, Dense

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data = array([
	[0.1, 1.0],
	[0.2, 0.9],
	[0.3, 0.8],
	[0.4, 0.7],
	[0.5, 0.6],
	[0.6, 0.5],
	[0.7, 0.4],
	[0.8, 0.3],
	[0.9, 0.2],
	[1.0, 0.1]])
data = data.reshape(1, 10, 2)
print(data.shape)

y = array([0, 0, 0, 1, 0, 1, 0, 0, 0, 0])
y = y.reshape(1, 10, 1)

model = Sequential()
model.add(LSTM(32, input_shape=(10, 2)))
model.add(Dense(2))

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit(data, y)