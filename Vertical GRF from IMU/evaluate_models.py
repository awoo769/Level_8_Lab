from utils import eval_prediction, weighted_categorical_crossentropy, show_results

import tensorflow as tf
from tensorflow import keras

import numpy as np
from matplotlib import pyplot
import sys
import os

from matplotlib import pyplot as plt

# Disables the tensorflow AVX2 warning, doesn't enable AVX2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
keras.losses.weighted_categorical_crossentropy = weighted_categorical_crossentropy


if __name__ == '__main__':

	# data folder
	data_folder = 'C:\\Users\\alexw\\Dropbox\\ABI\\Level_8_Lab\\Vertical GRF from IMU\\data\\'

	# models folder
	models_folder = 'C:\\Users\\alexw\\Dropbox\\ABI\\Level_8_Lab\\Vertical GRF from IMU\\models\\'

	weights = np.array([10, 350, 300])

	models_folder = '{}weights_{}_{}_{}\\'.format(models_folder, weights[0], weights[1], weights[2])

	for f in os.listdir(models_folder):
		if f.endswith('.h5'):
			model_name = models_folder + f

	# Load datasets and true outputs
	X_test = np.load(file=data_folder + "X_test.npy", allow_pickle=True)
	y_test = np.load(file=data_folder + "y_test.npy", allow_pickle=True)

	# Load model
	model = keras.models.load_model(model_name, custom_objects={'loss': weighted_categorical_crossentropy(weights)})
	likelihood = model.predict(X_test)

	sdist = eval_prediction(likelihood, y_test, 'test', plot=False)

	show_results(sdist=sdist, name=model_name)
