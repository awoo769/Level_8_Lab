import tensorflow as tf
from tensorflow import keras

#from time import time
#from tensorflow.python.keras.callbacks import TensorBoard


import numpy as np
from matplotlib import pyplot
import sys
import h5py
import pickle

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

# Disables the tensorflow AVX2 warning, doesn't enable AVX2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from utils import construct_model, train_model, save_model, plot_history

if __name__ == '__main__':

	# data folder
	data_folder = 'C:\\Users\\alexw\\Dropbox\\ABI\\Level_8_Lab\\Vertical_GRF_from_IMU\\data\\'

	# models folder
	models_folder = 'C:\\Users\\alexw\\Dropbox\\ABI\\Level_8_Lab\\Vertical_GRF_from_IMU\\models\\'

	with open(data_folder + "dataset.pkl", "rb") as f:
		dataset = pickle.load(f)


	X_train = np.array([])
	y_train = np.array([])

	counter = 0
	ntrials = len(dataset)

	for keys in dataset.keys():
		counter += 1

		if counter < ntrials: # Train on 11, test on 1
			if X_train.size == 0:
				X_train = dataset[keys]['X']
				y_train = dataset[keys]['y']
			else:
				X_train = np.vstack((X_train, dataset[keys]['X']))
				y_train = np.vstack((y_train, dataset[keys]['y']))

	'''
	hf = h5py.File(data_folder + 'dataset.h5' ,'r')

	# Load datasets and true outputs
	X_train = hf[('X_train')]
	y_train = hf[('y_train')]

	X_test = hf[('X_test')]
	y_test = hf[('y_test')]
	'''
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
	shape = (X_train.shape[1], X_train.shape[2] - 2) # Not training on first 2 columns (id and time)
	model = construct_model(input_shape=shape, output_dim=3, weights=weights)

	#tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
	nepochs = 10

	diff_accuracy = 100
	delta = 0.001
	accuracy = 1
	counter = 1

	history_list = []

	while diff_accuracy > delta: # While diff accuracy is greater than delta
		print('Epoch #{}'.format(counter))
		history = train_model(model, X_train[:,:,2:], y_train[:], 32, nepochs=1)

		prev_accuracy = accuracy
		accuracy = history.history['accuracy'][0]
		print('Accuracy = {}'.format(accuracy))
		diff_accuracy = abs((accuracy - prev_accuracy) / prev_accuracy * 100) # Change in accuracy
	
		print('Change in accuracy = {} %\n'.format(diff_accuracy))

		history_list.append(history)

		counter += 1
	
	# Save the model
	name = 'foot_events_{}_{}_{}'.format(weights[0], weights[1], weights[2])
	save_model(model, models_folder, '{}.h5'.format(name))

	plot_history(history, name, models_folder)