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
	X_train = np.load(file=data_folder + 'X_train.npy', allow_pickle=True)
	y_train = np.load(file=data_folder + 'y_train.npy', allow_pickle=True)

	X_test = np.load(file=data_folder + 'X_test.npy', allow_pickle=True)
	y_test = np.load(file=data_folder + 'y_test.npy', allow_pickle=True)

	# Weighting for each event
	# [no event, FS, FO]
	weights = np.array([10, 350, 350])

	if not os.path.exists('{}weights_{}_{}_{}\\'.format(models_folder, weights[0], weights[1], weights[2])):
		os.makedirs('{}weights_{}_{}_{}\\'.format(models_folder, weights[0], weights[1], weights[2]))
	
	models_folder = '{}weights_{}_{}_{}\\'.format(models_folder, weights[0], weights[1], weights[2])

	np.save('{}weights_{}_{}_{}.npy'.format(models_folder, weights[0], weights[1], weights[2]), weights)

	# Each timestep has 6 "features" (ax_L, ay_L, az_L, ax_R, ay_R, az_R)
	# Shape of dataset = (n, 636, 6)

	# To plot:
	#plt.plot(X_train[0,:,1]) # Left y acceleration
	#plt.plot(X_train[0,:,4]) # Right y acceleration
	#plt.plot(y_train[0,:,0]*100,'g') # HS
	#plt.plot(y_train[0,:,1]*100, 'r') # TO
	#plt.show()

	# Contstruct the model
	model = construct_model(input_shape=X_train.shape[1:], output_dim=3, weights=weights)

	#tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
	nepochs = 5

	history = train_model(model, X_train, y_train, 32, nepochs=nepochs, validation=True, validation_data=X_test, validation_truths=y_test)

	# Save the model
	name = 'foot_events_{}_{}_{}'.format(weights[0], weights[1], weights[2])
	save_model(model, models_folder, '{}.h5'.format(name))

	plot_history(history, name, models_folder)