import tensorflow

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
import numpy as np
from numpy import array
from random import random
from numpy import cumsum
from matplotlib import pyplot
from pandas import DataFrame

from matplotlib import pyplot as plt

# create a sequence classification instance
def get_sequence(n_timesteps):
	# create a sequence of random numbers in [0,1]
	X = array([random() for _ in range(10)])

	# calculate cut-off value to change class values
	limit = n_timesteps/4.0

	# determine the class outcome for each item in cumulative sequence
	y = array([0 if x < limit else 1 for x in cumsum(X)])
	
	# Expected structure of an LSTM has the dimensions [samples, timesteps, features]
	# reshape input and output data to be suitable for LSTMs
	X = X.reshape(1, n_timesteps, 1)
	y = y.reshape(1, n_timesteps, 1)

	return X, y

def get_lstm_model(n_timesteps, backwards):
	model = Sequential()
	model.add(LSTM(20, input_shape=(n_timesteps, 1), return_sequences=True, go_backwards=backwards))
	model.add(TimeDistributed(Dense(1, activation='sigmoid')))
	model.compile(loss='binary_crossentropy', optimizer='adam')

	return model

def get_bi_lstm_model(n_timesteps, mode):
	model = Sequential()
	model.add(Bidirectional(LSTM(20, return_sequences=True), input_shape=(n_timesteps, 1), merge_mode=mode))
	model.add(TimeDistributed(Dense(1, activation='sigmoid')))
	model.compile(loss='binary_crossentropy', optimizer='adam')

	return model

def train_model(model, n_timesteps):
	loss = list()
	for _ in range(250):
		# generate new random sequence
		X, y = get_sequence(n_timesteps)

		# fit model for one epoch on this sequence
		hist = model.fit(X, y, epochs=1, batch_size=1, verbose=0)
		loss.append(hist.history['loss'][0])

	return loss

'''
# define problem properties
n_timesteps = 10
results = DataFrame()

# lstm forwards
model = get_lstm_model(n_timesteps, False)
results['lstm_forw'] = train_model(model, n_timesteps)

# lstm backwards
model = get_lstm_model(n_timesteps, True)
results['lstm_back'] = train_model(model, n_timesteps)

# bidirectional concat
model = get_bi_lstm_model(n_timesteps, 'concat')
results['bilstm_con'] = train_model(model, n_timesteps)

# line plot of results
results.plot()
pyplot.show()
'''
if __name__ == '__main__':
	# Load datasets and real outputs
	dataset = np.load(file="C:\\Users\\alexw\\Dropbox\\ABI\\Level_8_Lab\\Vertical GRF from IMU\\dataset.npy", allow_pickle=True)
	HS_TO = np.load(file="C:\\Users\\alexw\\Dropbox\\ABI\\Level_8_Lab\\Vertical GRF from IMU\\HS_TO.npy", allow_pickle=True)

	# Train on 90 % of data, test on 10 %
	# For now, use bottom 10 %.

	n_samples = len(dataset)

	n_training = int(n_samples * 0.9)

	training = dataset[:n_training]
	validation = dataset[n_training:]

	training_truths = HS_TO[:n_training]
	validation_truths = HS_TO[n_training:]

	# Each timestep has 6 "features" (ax_L, ay_L, az_L, ax_R, ay_R, az_R)
	# Shape of dataset = (n, 636, 6)

	# To plot:
	# plt.plot(training[0])
	# plt.plot(training_truths[0][0],(training[0][:])[training_truths[0][0]],'o','g') # HS
	# plt.plot(training_truths[0][1],(training[0][:])[training_truths[0][1]],'o','r') # TO

	a = 1

	# Constrained peak detection algorithm for RNN - structured prediction model
	# IC and TO event of the same foot are speparated by at least 35 ms and at most 200 ms
	# TO and IC event of opposing feet are separated by at least 160 ms and at most 350 ms
