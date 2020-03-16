import tensorflow as tf
from tensorflow import keras

import numpy as np
from matplotlib import pyplot
import sys

from matplotlib import pyplot as plt


# Disables the tensorflow AVX2 warning, doesn't enable AVX2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Construct the model
def construct_model(units: int = 32, lstm_layers: int = 2, input_shape: int = (636, 6), output_dim: int = 1, weights: np.ndarray = None):
	model = keras.Sequential()
	model.add(keras.layers.Bidirectional(keras.layers.LSTM(units, return_sequences=True, input_shape=input_shape), input_shape=input_shape))
	
	for _ in range(lstm_layers-1):
		model.add(keras.layers.Bidirectional(keras.layers.LSTM(units, return_sequences=True)))
		#model.add(keras.layers.LSTM(units, return_sequences=True))

	loss = weighted_categorical_crossentropy(weights)
	
	model.add(keras.layers.TimeDistributed(keras.layers.Dense(output_dim, activation='softmax'))) #activation=sigmoid
	model.compile(loss=loss, optimizer='sgd', metrics=['accuracy'])

	return model


# Create loss function
def weighted_binary_crossentropy(y_true, y_pred):
	# Play around with this - see if it is neccesary to have both HS and TO

	# adding 0.01 so nothing is multiplied by 0
	l = keras.backend.mean(keras.backend.binary_crossentropy(y_pred[:,0], y_true[:,0]) * (y_true[:,0] + 0.01), axis=-1)
	
	return l


def weighted_categorical_crossentropy(weights):
	"""
	A weighed version of keras.objectives.categorical_crossentropy

	Variables:
		weights: numpy array of shape (C,) where C is the number of classes
	
	Usage:
		weights: np.array([0.5, 2, 10]) # Class 1 at 0.5, class 2 twice the normal weights, class 3 10x
		loss = weighted_categorical_crossentropy(weights)
		model.compile(loss=loss,optimizer='sgd')

	"""
	
	# We have 3 categories. y_true[:,0] = no event, y_true[:,1] = FS, y_true[:,2] = FO
	# We want there to be a significantly higher weighting for the FS and FO than the no event.
	weights = keras.backend.variable(weights)

	def loss(y_true, y_pred):
		# Scale predictions so that the class probabilities of each sample sum to 1
		y_pred /= keras.backend.sum(y_pred, axis=-1, keepdims=True)
		# clibp to prevent NaN's and Inf's
		y_pred = keras.backend.clip(y_pred, keras.backend.epsilon(), 1 - keras.backend.epsilon())

		# calc
		loss = y_true * keras.backend.log(y_pred) * weights
		loss = -keras.backend.sum(loss, -1)

		return loss
	
	return loss


# Train the model
def train_model(model = None, x: np.ndarray = None, y: np.ndarray = None, batch_size: int = 32, nepochs: int = None, validation: str = False, validation_data: np.ndarray = None, validation_truths: np.ndarray = None):
	# x shape = (n_samples, n_timesteps, n_features)
	# y shape = (n_samples, n_timesteps) - binary_results
	if validation:
		history = model.fit(x=x, y=y, batch_size=batch_size, epochs=nepochs, validation_data=(validation_data, validation_truths))
	else:
		history = model.fit(x=x, y=y, batch_size=batch_size, epochs=nepochs) #, callbacks=[tensorboard])

	return history


# Save the model
def save_model(model = None, save_dir: str = None, name: str = None):
	model.save(save_dir + name)


def predict_model(model = None, x: np.ndarray = None):
	# Make sure that the input data is in the correct shape
	# (n, 636, 6)
	if x.shape[1:] != (636, 6):
		print('Input data does not have the correct shape')
	
	else:
		output = model.predict(x)
		peak_ind = peak_det(output, 0.5)
	
	return peak_ind


# Plot training history
def plot_history(history, name: str = None, save_dir: str = None):
	nepoch = len(history.history['loss'])

	plt.plot(range(nepoch),history.history['loss'],'r')
	plt.plot(range(nepoch),history.history['val_loss'],'b')

	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'validation'], loc='upper right')
	plt.show()

	plt.savefig(save_dir + name + '_loss.png')

	plt.plot(range(nepoch),history.history['accuracy'],'r')
	plt.plot(range(nepoch),history.history['val_accuracy'],'b')

	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'validation'], loc='upper right')
	plt.show()

	plt.savefig(save_dir + name + '_accuracy.png')

# Detect peaks
def peak_det(likelihood: np.ndarray, cutoff: int = 0.2):
	"""
	This function returns the most likely foot strike and foot off event indices, as detected
	by the model and contrained by this function.
	
	IC and TO event of the same foot are speparated by at least 35 ms and at most 200 ms
	TO and IC event of opposing feet are separated by at least 160 ms and at most 350 ms

	The peaks must have a likelhood of above cutoff

	"""

	from scipy.signal import argrelextrema

	out = np.zeros(likelihood.shape)

	FS_initial = argrelextrema(likelihood[:,0], np.greater)[0]
	FO_initial = argrelextrema(likelihood[:,1], np.greater)[0]

	FS_initial = FS_initial[(likelihood[:,0])[FS_initial] > cutoff]
	FO_initial = FO_initial[(likelihood[:,1])[FO_initial] > cutoff]

	# Run through each of the FS and FO arrays to see if there are any peaks that should be combined (less than 35 ms apart)
	FS_initial2 = []
	do_not_append = []
	for i in range(1, len(FS_initial)):
		if FS_initial[i] - FS_initial[i-1] < 35:
			# Remove the previous first event and replace with one 3/4s between the two same events
			temp = int((FS_initial[i] - FS_initial[i-1]) * 3/4) + FS_initial[i-1]
			FS_initial2.append(temp)
			# Add both to the do not append list
			do_not_append.append(FS_initial[i])
			do_not_append.append(FS_initial[i-1])

		else:
			if FS_initial[i-1] not in do_not_append:
				FS_initial2.append(FS_initial[i-1])
			else:
				pass
	
	if FS_initial[-1] not in do_not_append and FS_initial[-1] not in FS_initial2:
		FS_initial2.append(FS_initial[-1])
	
	FO_initial2 = []
	do_not_append = []
	for i in range(1, len(FO_initial)):
		if FO_initial[i] - FO_initial[i-1] < 35:
			# Remove the previous first event and replace with one 3/4s between the two same events
			temp = int((FO_initial[i] - FO_initial[i-1]) * 3/4) + FO_initial[i-1]
			FO_initial2.append(temp)
			# Add both to the do not append list
			do_not_append.append(FO_initial[i])
			do_not_append.append(FO_initial[i-1])

		else:
			if FO_initial[i-1] not in do_not_append:
				FO_initial2.append(FO_initial[i-1])
			else:
				pass

	if FO_initial[-1] not in do_not_append and FO_initial[-1] not in FO_initial2:
		FO_initial2.append(FO_initial[-1])

	FS_initial = np.array(FS_initial2)
	FO_initial = np.array(FO_initial2)

	if len(FS_initial) <= len(FO_initial):
		length = len(FS_initial)
		smallest = FS_initial
		longest = FO_initial
	else:
		length = len(FO_initial)
		smallest = FO_initial
		longest = FS_initial
	
	difference = abs(len(FS_initial) - len(FO_initial))

	FO = 0
	FS = 0

	# Check which comes first
	if FS_initial[0] < FO_initial[0]:
		first = FS_initial
		second = FO_initial
		FS = 1
	
	else:
		first = FO_initial
		second = FS_initial
		FO = 1

	# See if there are any FO's which have another FO before the next FS. If there is, take the second FO
	FO_diff = []
	for i in range(len(FO_initial)):
		FO_diff.append(FS_initial - FO_initial[i])

	# If FO = 1, the 1st row of FO_diff should contain 0 negatives, 2nd 1, 3rd 2 and etc...
	if FO != 1:
		pass # TODO
	else:
		for i in range(len(FO_diff)):
			neg_count = len(list(filter(lambda x: (x < 0), FO_diff[i].tolist())))
			if neg_count != i:
				# Then the value of FO previous is wrong
				FO_initial = np.delete(FO_initial, i-1)

	for i in range(length):
		# Each foot strike should be followed by a foot off.
		if first[i] < second[i]:
			if FS:
				out[first[i],0] = 1 # FS occurs here
				out[second[i],1] = 1 # FO occurs here
			elif FO:
				out[first[i],1] = 1 # FO orrurs here
				out[second[i],0] = 1 # FS orrurs here
		else:
			pass
	
	# Make sure that there aren't any other events missed
	if difference > 0:
		if smallest[i] > longest[i+1]:
			if longest[i+1] - longest[i] < 35:
				temp = int((longest[i+1] - longest[i]) * 3/4) + longest[i]


			for j in range(difference):
				if len(FO_initial) < len(FS_initial):



					a = 1
	return out

# Compare predicted to true
def peak_cmp(true, predicted):
	'''
	This function returns the difference between a true event and a predicted event

	'''

	FS_true = np.where(true[:,0] == 1)[0]
	FO_true = np.where(true[:,1] == 1)[0]

	FS_est = np.where(predicted[:,0] == 1)[0]
	FO_est = np.where(predicted[:,1] == 1)[0]

	if len(FS_true) != len(FS_est):
		return -1
	
	if len(FO_true) != len(FO_est):
		return -1
	
	temp1 = []
	for i in range(len(FS_true)):
		temp1.append(FS_true[i] - FS_est[i])
	
	temp2 = []
	for i in range(len(FO_true)):
		temp2.append(FO_true[i] - FO_est[i])
	
	dist = (temp1, temp2)

	return dist

# Evaluate the model predition
def eval_prediction(likelihood, true, patient, plot = True):
	sdist = []
	
	n_samples = likelihood.shape[0]

	for i in range(n_samples):
		if i == 174:
			stop = True
		est_events = peak_det(likelihood[i,:,1:], 0.2) # Foot strike and foot off

		sdist.append(peak_cmp(true[i,:,1:], est_events))

		if plot:
			plt.plot(est_events) # continous likelihood process
			plt.plot(true[i,:,1:]) # spikes on events
			plt.title(patient)
			plt.show()

	return sdist
