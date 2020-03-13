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

	# Directory
	directory = "C:\\Users\\alexw\\Dropbox\\ABI\\Level_8_Lab\\Vertical GRF from IMU\\"

	# Load datasets and true outputs
	dataset = np.load(file=directory + "data\\dataset.npy", allow_pickle=True)
	HS_TO = np.load(file=directory + "data\\HS_TO.npy", allow_pickle=True)

	# Load models
	#model_HS, model_TO = get_models(directory)

	n_samples = len(dataset)
	n_training = int(n_samples * 0.9)

	training_data = dataset[:n_training]
	validation_data = dataset[n_training:]

	training_truths = HS_TO[:n_training]
	validation_truths = HS_TO[n_training:]

	#likelihood_HS = model_HS.predict(validation_data)
	#likelihood_TO = model_TO.predict(validation_data)

	
	y_true = HS_TO[:,:,0] + HS_TO[:,:,1] * -1
	y_training = y_true[:n_training]
	y_validation = y_true[n_training:]


	#eval_prediction(likelihood_HS, likelihood_TO, validation_truths, 'testing')
	weights = np.array([636/632, 636/2, 636/2])
	model = keras.models.load_model(directory + '\\models\\neither_new.h5', custom_objects={'loss': weighted_categorical_crossentropy(weights)})
	likelihood = model.predict(validation_data)

	a =1
