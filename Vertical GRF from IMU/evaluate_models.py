from utils import weighted_binary_crossentropy, eval_prediction, weighted_categorical_crossentropy

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

keras.losses.weighted_binary_crossentropy = weighted_binary_crossentropy
keras.losses.weighted_categorical_crossentropy = weighted_categorical_crossentropy


def get_models(directory: str = None):
	if not os.path.exists(directory + 'models\\HS.h5'):
		print('Models do not exist in selected directory')

		return -1
	else:
		model_HS = keras.models.load_model(directory + 'models\\HS.h5')#, custom_objects={'weighted_binary_crossentropy': weighted_binary_crossentropy})
		model_TO = keras.models.load_model(directory + 'models\\TO.h5')#, custom_objects={'weighted_binary_crossentropy': weighted_binary_crossentropy})

		return model_HS, model_TO


if __name__ == '__main__':

	# data folder
	data_folder = 'C:\\Users\\alexw\\Dropbox\\ABI\\Level_8_Lab\\Vertical GRF from IMU\\data\\'

	# models folder
	models_folder = 'C:\\Users\\alexw\\Dropbox\\ABI\\Level_8_Lab\\Vertical GRF from IMU\\models\\'

	# Load datasets and true outputs
	X_test = np.load(file=data_folder + "X_test.npy", allow_pickle=True)
	y_test = np.load(file=data_folder + "y_test", allow_pickle=True)
	weights = np.load(file=models_folder + "weights.npy", allow_pickle=True)

	# Load model
	model = keras.models.load_model(models_folder + 'foot_events.h5', custom_objects={'loss': weighted_categorical_crossentropy(weights)})
	likelihood = model.predict(y_test)

	a = 1
