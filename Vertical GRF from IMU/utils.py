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
	#l1 = keras.backend.mean(keras.backend.binary_crossentropy(y_pred[:,0], y_true[:,0]) * (y_true[:,0] + 0.01), axis=-1)
	#l2 = keras.backend.mean(keras.backend.binary_crossentropy(y_pred[:,1], y_true[:,1]) * (y_true[:,1] + 0.01), axis=-1)
	#l = l1 + l2
	
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
def save_model(model = None, save_dir: str = None, model_type: str = None):
	model.save(save_dir + model_type + '_new.h5')


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
def plot_history(history, model_type: str = None, save_dir: str = None):
	nepoch = len(history.history['loss'])

	plt.plot(range(nepoch),history.history['loss'],'r')
	plt.plot(range(nepoch),history.history['val_loss'],'b')

	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'validation'], loc='upper right')
	plt.show()

	plt.savefig(save_dir + model_type + '_loss.png')

	plt.plot(range(nepoch),history.history['accuracy'],'r')
	plt.plot(range(nepoch),history.history['val_accuracy'],'b')

	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'validation'], loc='upper right')
	plt.show()

	plt.savefig(save_dir + model_type + '_accuracy.png')

# Detect peaks
def peak_det(v, delta, x = None):
	"""
	Converted from MATLAB script at http://billauer.co.il/peakdet.html
	
	Returns two arrays
	
	function [maxtab, mintab]=peakdet(v, delta, x)
	%PEAKDET Detect peaks in a vector
	%        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
	%        maxima and minima ("peaks") in the vector V.
	%        MAXTAB and MINTAB consists of two columns. Column 1
	%        contains indices in V, and column 2 the found values.
	%      
	%        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
	%        in MAXTAB and MINTAB are replaced with the corresponding
	%        X-values.
	%
	%        A point is considered a maximum peak if it has the maximal
	%        value, and was preceded (to the left) by a value lower by
	%        DELTA.
	
	% Eli Billauer, 3.4.05 (Explicitly not copyrighted).
	% This function is released to the public domain; Any use is allowed.
	
	"""

	# Constrained peak detection algorithm for RNN - structured prediction model TODO - see if this is needed first
	# IC and TO event of the same foot are speparated by at least 35 ms and at most 200 ms
	# TO and IC event of opposing feet are separated by at least 160 ms and at most 350 ms

	maxtab = []
	mintab = []
	   
	if x is None:
		x = np.arange(len(v))
	
	v = np.asarray(v)
	
	if len(v) != len(x):
		sys.exit('Input vectors v and x must have same length')
	
	if not np.isscalar(delta):
		sys.exit('Input argument delta must be a scalar')
	
	if delta <= 0:
		sys.exit('Input argument delta must be positive')
	
	mn, mx = np.Inf, -np.Inf
	mnpos, mxpos = np.NaN, np.NaN
	
	lookformax = True
	
	for i in np.arange(len(v)):
		this = v[i]
		if this > mx:
			mx = this
			mxpos = x[i]
		if this < mn:
			mn = this
			mnpos = x[i]
		
		if lookformax:
			if this < mx-delta:
				maxtab.append((mxpos, mx))
				mn = this
				mnpos = x[i]
				lookformax = False
		else:
			if this > mn+delta:
				mintab.append((mnpos, mn))
				mx = this
				mxpos = x[i]
				lookformax = True

	return np.array(maxtab), np.array(mintab)


# Compare predicted to true
def peak_cmp(annotated, predicted):
	dist = []
	if len(predicted) == 0 or len(annotated) == 0:
		return -1
	if len(predicted) != len(annotated):
		return -1
	
	for a in annotated:
		dist = dist + [min(np.abs(predicted - a))]
	if not len(dist):
		return -1
	return min(dist)


# Evaluate the model predition
def eval_prediction(likelihood_HS, likelihood_TO, true, patient, plot = True, shift = 0):
	sdist = []
	
	n_samples = likelihood_HS.shape[0]
	peakind = peak_det(likelihood_HS[0],0.5)
	for k,_ in peakind[0]:
		if plot:
			plt.axvline(x=k)
	sdist.append(peak_cmp(np.where(true[:,0] > 0.5)[0], [k + shift for k,_ in peakind[0]]))

	if plot:
		plt.plot(likelihood_HS) # continous likelihood process
		plt.plot(true) # spikes on events
		plt.title(patient)
		axes = plt.gca()
		axes.set_xlim([0,true.shape[0]])
		plt.show()

	return sdist
