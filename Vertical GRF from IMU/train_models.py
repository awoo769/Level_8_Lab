import tensorflow as tf
from tensorflow import keras

#from time import time
#from tensorflow.python.keras.callbacks import TensorBoard


import numpy as np
from matplotlib import pyplot
import sys

from matplotlib import pyplot as plt

# Disables the tensorflow AVX2 warning, doesn't enable AVX2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from utils import construct_model, train_model, save_model

if __name__ == '__main__':

	# Load datasets and true outputs
	dataset = np.load(file="C:\\Users\\alexw\\Dropbox\\ABI\\Level_8_Lab\\Vertical GRF from IMU\\dataset.npy", allow_pickle=True)
	HS_TO = np.load(file="C:\\Users\\alexw\\Dropbox\\ABI\\Level_8_Lab\\Vertical GRF from IMU\\HS_TO.npy", allow_pickle=True)

	# Train on 90 % of data, test on 10 %
	# For now, use bottom 10 %.

	n_samples = len(dataset)

	n_training = int(n_samples * 0.9)

	training_data = dataset[:n_training]
	validation_data = dataset[n_training:]

	training_truths = HS_TO[:n_training]
	validation_truths = HS_TO[n_training:]

	# Each timestep has 6 "features" (ax_L, ay_L, az_L, ax_R, ay_R, az_R)
	# Shape of dataset = (n, 636, 6)

	# To plot:
	#plt.plot(training_data[0,:,1]) # Left y acceleration
	#plt.plot(training_data[0,:,4]) # Right y acceleration
	#plt.plot(training_truths[0,:,0]*100,'g') # HS
	#plt.plot(training_truths[0,:,1]*100, 'r') # TO
	#plt.show()

	# Contstruct the model
	model = construct_model(input_shape=dataset.shape[1:], output_dim=1)

	#tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

	model_type = 'TO' # HS or TO

	if model_type == 'HS':
		history = train_model(model, training_data, training_truths[:,:,0], 32, 20)
	
	elif model_type == 'TO':
		history = train_model(model, training_data, training_truths[:,:,1], 32, 20)

	# Save the model
	save_dir = 'C:\\Users\\alexw\\Dropbox\\ABI\\Level_8_Lab\\Vertical GRF from IMU\\'
	save_model(model, save_dir, model_type)

	# Constrained peak detection algorithm for RNN - structured prediction model TODO - see if this is needed first
	# IC and TO event of the same foot are speparated by at least 35 ms and at most 200 ms
	# TO and IC event of opposing feet are separated by at least 160 ms and at most 350 ms

	#likelihood = model.predict(validation_data)
	#eval_prediction(likelihood, validation_truths, 'test')
