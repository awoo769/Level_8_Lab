import keras.layers as L
import keras.models as M
import tensorflow as tf

import numpy as np

data_x = np.array([
	[
		[1, 2, 3],
		[4, 5, 6]
	],
	[
		[7, 8, 9],
		[10, 11, 12]
	]
])

data_y = np.array([
	[
		[101, 102, 103, 104],
		[105, 106, 107, 108]
	],
	[
		[201, 202, 203, 204],
		[205, 206, 207, 208]
	]
])

model_input = L.Input(shape=(2,3))

model_output = L.LSTM(4, return_sequences=True)(model_input)

model = M.Model(input=model_input, output=model_output)

# sgd = stocastic gradient descent
model.compile(loss='mean_squared_error', optimizer='sgd')

model.fit(data_x, data_y)