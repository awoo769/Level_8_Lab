import tensorflow as tf
from tensorflow import keras

#from time import time
#from tensorflow.python.keras.callbacks import TensorBoard


import numpy as np
from matplotlib import pyplot
import sys

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

# Disables the tensorflow AVX2 warning, doesn't enable AVX2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from utils import construct_model, train_model, save_model, plot_history

if __name__ == '__main__':

	# data folder
	data_folder = 'C:\\Users\\alexw\\Dropbox\\ABI\\Level_8_Lab\\Vertical GRF from IMU\\data\\'

	# models folder
	models_folder = 'C:\\Users\\alexw\\Dropbox\\ABI\\Level_8_Lab\\Vertical GRF from IMU\\models\\'

	# Load datasets and true outputs
	dataset = np.load(file=data_folder + 'dataset.npy', allow_pickle=True)
	HS_TO = np.load(file= data_folder + 'HS_TO.npy', allow_pickle=True)

	# Train on 80 % of data, test on 20 %

	# Another option. 3 classes. 0 = no event, 1 = foot strike, 2 = foot off
	y = HS_TO[:,:,0] * 1 + HS_TO[:,:,1] * 2
	y_true = keras.utils.to_categorical(y)

	X_train, X_test, y_train, y_test = train_test_split(dataset, y_true, test_size=0.2)

	# Save datasets
	np.save(models_folder + 'X_train.npy', X_train)
	np.save(models_folder + 'y_train.npy', y_train)
	np.save(models_folder + 'X_test.npy', X_test)
	np.save(models_folder + 'y_test.npy', y_test)

	# Weighting for each event
	weights = np.array([10, 300, 350])
	np.save(models_folder + 'weights.npy', weights)

	# Each timestep has 6 "features" (ax_L, ay_L, az_L, ax_R, ay_R, az_R)
	# Shape of dataset = (n, 636, 6)

	# To plot:
	#plt.plot(X_train[0,:,1]) # Left y acceleration
	#plt.plot(X_train[0,:,4]) # Right y acceleration
	#plt.plot(y_train[0,:,0]*100,'g') # HS
	#plt.plot(y_train[0,:,1]*100, 'r') # TO
	#plt.show()

	# Contstruct the model
	model = construct_model(input_shape=dataset.shape[1:], output_dim=3, weights=weights)

	#tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

	history = train_model(model, X_train, y_train, 32, nepochs=20, validation=True, validation_data=X_test, validation_truths=y_test)

	# Save the model
	name = 'foot_events.h5'
	save_model(model, models_folder, name)

	plot_history(history, name, models_folder)

	likelihood = model.predict(X_test)

	plt.plot(likelihood[0,:,0], label='No event probability')
	plt.plot(likelihood[0,:,1], label='Foot strike probability')
	plt.plot(likelihood[0,:,2], label='Foot off probability')
	plt.plot(y_test[0,:,0], label='No event true')
	plt.plot(y_test[0,:,1], label='Foot strike true')
	plt.plot(y_test[0,:,2], label='Foot off true')
	plt.xlabel('time (ms)')
	plt.ylabel('probability (unitless)')
	plt.legend()
	plt.show()

	plt.plot(likelihood[-1,:,0], label='No event probability')
	plt.plot(likelihood[-1,:,1], label='Foot strike probability')
	plt.plot(likelihood[-1,:,2], label='Foot off probability')
	plt.plot(y_test[-1,:,0], label='No event true')
	plt.plot(y_test[-1,:,1], label='Foot strike true')
	plt.plot(y_test[-1,:,2], label='Foot off true')
	plt.xlabel('time (ms)')
	plt.ylabel('probability (unitless)')
	plt.legend()
	plt.show()