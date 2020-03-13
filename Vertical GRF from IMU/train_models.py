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

from utils import construct_model, train_model, save_model, plot_history

if __name__ == '__main__':

	# Load datasets and true outputs
	dataset = np.load(file="C:\\Users\\alexw\\Dropbox\\ABI\\Level_8_Lab\\Vertical GRF from IMU\\data\\dataset.npy", allow_pickle=True)
	HS_TO = np.load(file="C:\\Users\\alexw\\Dropbox\\ABI\\Level_8_Lab\\Vertical GRF from IMU\\data\\HS_TO.npy", allow_pickle=True)

	# Train on 90 % of data, test on 10 %
	# For now, use bottom 10 %.

	n_samples = len(dataset)

	n_training = int(n_samples * 0.9)

	training_data = dataset[:n_training]
	validation_data = dataset[n_training:]

	training_truths = HS_TO[:n_training]
	validation_truths = HS_TO[n_training:]

	# Another option. 3 classes. 0 = no event, 1 = foot strike, 2 = foot off
	y = HS_TO[:,:,0] * 1 + HS_TO[:,:,1] * 2
	y_true = keras.utils.to_categorical(y)

	# Weighting for each event
	weights = np.array([636/632, 636/2, 636/2])

	y_training = y_true[:n_training]
	y_validation = y_true[n_training:]


	# Each timestep has 6 "features" (ax_L, ay_L, az_L, ax_R, ay_R, az_R)
	# Shape of dataset = (n, 636, 6)

	# To plot:
	#plt.plot(training_data[0,:,1]) # Left y acceleration
	#plt.plot(training_data[0,:,4]) # Right y acceleration
	#plt.plot(training_truths[0,:,0]*100,'g') # HS
	#plt.plot(training_truths[0,:,1]*100, 'r') # TO
	#plt.show()

	# Contstruct the model
	model = construct_model(input_shape=dataset.shape[1:], output_dim=3, weights=weights)

	#tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

	model_type = 'neither' # HS or TO or both

	history = train_model(model, training_data, y_training, 32, 15, validation=True, validation_data=validation_data, validation_truths=y_validation)

	if model_type == 'HS':
		history = train_model(model, training_data, training_truths[:,:,0], 32, 4, validation=True, validation_data=validation_data, validation_truths=validation_truths[:,:,0])
	
	elif model_type == 'TO':
		history = train_model(model, training_data, training_truths[:,:,1], 32, 30, validation=True, validation_data=validation_data, validation_truths=validation_truths[:,:,1])
	


	# Save the model
	save_dir = 'C:\\Users\\alexw\\Dropbox\\ABI\\Level_8_Lab\\Vertical GRF from IMU\\models\\'
	save_model(model, save_dir, model_type)

	plot_history(history, model_type, save_dir)

	likelihood = model.predict(validation_data)

	plt.plot(likelihood[0][0], label='No event probability')
	plt.plot(likelihood[0][1], label='Foot strike probability')
	plt.plot(likelihood[0][2], label='Foot off probability')
	plt.plot(y_validation[0][0], label='No event true')
	plt.plot(y_validation[0][1], label='Foot strike true')
	plt.plot(y_validation[0][2], label='Foot off true')
	plt.xlabel('time (ms)')
	plt.ylabel('probability (unitless)')
	plt.show()

	plt.plot(likelihood[-1][0], label='No event probability')
	plt.plot(likelihood[-1][1], label='Foot strike probability')
	plt.plot(likelihood[-1][2], label='Foot off probability')
	plt.plot(y_validation[-1][0], label='No event true')
	plt.plot(y_validation[-1][1], label='Foot strike true')
	plt.plot(y_validation[-1][2], label='Foot off true')
	plt.show()

	a = 1