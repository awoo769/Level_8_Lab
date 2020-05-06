from utils import eval_prediction, weighted_categorical_crossentropy, show_results

import tensorflow as tf
from tensorflow import keras

import numpy as np
from matplotlib import pyplot
import sys
import os

from matplotlib import pyplot as plt
import h5py
import pickle

# Disables the tensorflow AVX2 warning, doesn't enable AVX2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
keras.losses.weighted_categorical_crossentropy = weighted_categorical_crossentropy


if __name__ == '__main__':

	# data folder
	data_folder = 'C:\\Users\\alexw\\Dropbox\\ABI\\Level_8_Lab\\Vertical_GRF_from_IMU\\data\\'

	# models folder
	models_folder = 'C:\\Users\\alexw\\Dropbox\\ABI\\Level_8_Lab\\Vertical_GRF_from_IMU\\models\\'

	weights = np.array([10, 350, 350])

	models_folder = '{}weights_{}_{}_{}\\'.format(models_folder, weights[0], weights[1], weights[2])

	with open(data_folder + "dataset.pkl", "rb") as f:
		dataset = pickle.load(f)

	X_test = np.array([])
	y_test = np.array([])

	counter = 0
	ntrials = len(dataset)

	for keys in dataset.keys():
		counter += 1

		if counter == ntrials: # Train on 11, test on 1
			X_test = dataset[keys]['X']
			y_test = dataset[keys]['y']

	for f in os.listdir(models_folder):
		if f.endswith('.h5'):
			model_name = models_folder + f

	'''
	hf = h5py.File(data_folder + 'dataset.h5' ,'r')
	
	# Load datasets and true outputs
	X_test = hf[('X_test')]
	y_test = hf[('y_test')]
	'''
	# Load model
	model = keras.models.load_model(model_name, custom_objects={'loss': weighted_categorical_crossentropy(weights)})
	likelihood = model.predict(X_test[:,:,2:])

	# Join events together from same timeseries
	
	# Add time and uid to the likelihood array
	time = (X_test[:,:,1])[:, np.newaxis]
	time = np.swapaxes(time, 1, -1)

	uid = (X_test[:,:,0])[:, np.newaxis]
	uid = np.swapaxes(uid, 1, -1)

	y_est = np.concatenate((uid, time, likelihood),-1)

	max_time = int(max(y_est[-1,:,1]) * 1000) # in ms

	time_series = np.zeros((max_time+1, 4))
	time_series[:,0] = np.arange(0,max_time+1)

	y_true = np.zeros((max_time+1, 4))
	y_true[:,0] = np.arange(0,max_time+1)

	for i in range(len(time)):
		time_series[(np.squeeze(time[i]) * 1000).astype(int),1] = np.sqrt(time_series[(np.squeeze(time[i]) * 1000).astype(int),1]**2 + likelihood[i,:,0]**2)
		time_series[(np.squeeze(time[i]) * 1000).astype(int),2] = np.sqrt(time_series[(np.squeeze(time[i]) * 1000).astype(int),2]**2 + likelihood[i,:,1]**2)
		time_series[(np.squeeze(time[i]) * 1000).astype(int),3] = np.sqrt(time_series[(np.squeeze(time[i]) * 1000).astype(int),3]**2 + likelihood[i,:,2]**2)

		y_true[(np.squeeze(time[i]) * 1000).astype(int),1] = np.sqrt(y_true[(np.squeeze(time[i]) * 1000).astype(int),1]**2 + y_test[i,:,0]**2)
		y_true[(np.squeeze(time[i]) * 1000).astype(int),2] = np.sqrt(y_true[(np.squeeze(time[i]) * 1000).astype(int),2]**2 + y_test[i,:,1]**2)
		y_true[(np.squeeze(time[i]) * 1000).astype(int),3] = np.sqrt(y_true[(np.squeeze(time[i]) * 1000).astype(int),3]**2 + y_test[i,:,2]**2)

	sdist = eval_prediction(likelihood, y_test[:], 'test', plot=False)

	show_results(sdist=sdist, name=model_name)
