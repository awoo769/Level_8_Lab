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
def construct_model(units: int = 32, lstm_layers: int = 2, input_shape: int = (636, 6), output_dim: int = 1):
	model = keras.Sequential()
	model.add(keras.layers.LSTM(units, input_shape=input_shape, return_sequences=True))
	
	for _ in range(lstm_layers-1):
		model.add(keras.layers.LSTM(units, return_sequences=True))
	
	model.add(keras.layers.TimeDistributed(keras.layers.Dense(output_dim, activation='sigmoid')))
	model.compile(loss=weighted_binary_crossentropy, optimizer='sgd', metrics=['accuracy'])

	return model

# Create loss function
def weighted_binary_crossentropy(y_true, y_pred):
	# Play around with this - see if it is neccesary to have both HS and TO

	# adding 0.01 so nothing is multiplied by 0
	l = keras.backend.mean(keras.backend.binary_crossentropy(y_pred[:,0], y_true[:,0]) * (y_true[:,0] + 0.01), axis=-1)

	return l


# Train the model
def train_model(model = None, x: np.ndarray = None, y: np.ndarray = None, batch_size: int = 32, nepochs: int = None):
	# x shape = (n_samples, n_timesteps, n_features)
	# y shape = (n_samples, n_timesteps) - binary_results
	history = model.fit(x=x, y=y, batch_size=32, epochs=20) #, callbacks=[tensorboard])#, validation_data=(validation_data, validation_truths))

	return history


# Save the model
def save_model(model = None, save_dir: str = None, model_type: str = None):
	model.save(save_dir + model_type + '.h5')


# Plot training history
def plot_history(history):
	nepoch = len(history.history['loss'])

	plt.plot(range(nepoch),history.history['loss'],'r')
	plt.plot(range(nepoch),history.history['val_loss'],'b')

	axes = plt.gca()
	axes.set_ylim([0.001,0.005])

	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'validation'], loc='upper right')
	plt.show()

# Detect peaks
def peakdet(v, delta, x = None):
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
def eval_prediction(likelihood, true, patient, plot = True, shift = 0):
	sdist = []
	
	peakind = peakdet(likelihood[:,0],0.5)
	for k,_ in peakind[0]:
		if plot:
			plt.axvline(x=k)
	sdist.append(peak_cmp(np.where(true[:,0] > 0.5)[0], [k + shift for k,_ in peakind[0]]))

	if plot:
		plt.plot(likelihood) # continous likelihood process
		plt.plot(true) # spikes on events
		plt.title(patient)
		axes = plt.gca()
		axes.set_xlim([0,true.shape[0]])
		plt.show()

	return sdist
