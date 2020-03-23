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

	loss = weighted_categorical_crossentropy(weights)
	
	model.add(keras.layers.TimeDistributed(keras.layers.Dense(output_dim, activation='softmax'))) #activation=sigmoid
	model.compile(loss=loss, optimizer='sgd', metrics=['accuracy'])

	return model



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
	plt.savefig(save_dir + name + '_loss.png')
	plt.show()	

	plt.plot(range(nepoch),history.history['accuracy'],'r')
	plt.plot(range(nepoch),history.history['val_accuracy'],'b')

	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'validation'], loc='upper right')
	plt.savefig(save_dir + name + '_accuracy.png')
	plt.show()
	

# Detect peaks
def peak_det(likelihood: np.ndarray, cutoff: int = 0.15):
	"""
	This function returns the most likely foot strike and foot off event indices, as detected
	by the model and contrained by this function.
	
	IC and TO event of the same foot are speparated by at least 160 ms and at most 350 ms
	TO and IC event of opposing feet are separated by at least 35 ms and at most 200 ms

	The peaks must have a likelhood of above cutoff

	"""

	from scipy.signal import argrelextrema

	out = np.zeros(likelihood.shape)

	FS_initial = argrelextrema(likelihood[:,0], np.greater)[0]
	FO_initial = argrelextrema(likelihood[:,1], np.greater)[0]

	FS_initial = FS_initial[(likelihood[:,0])[FS_initial] > cutoff]
	FO_initial = FO_initial[(likelihood[:,1])[FO_initial] > cutoff]

	# If there is no detected events
	if len(FO_initial) == 0 and len(FS_initial) == 0:
		return out

	# Case where there is a FS at the very end of the recording with nothing before it.
	if len(FO_initial) == 0 and len(FS_initial == 1):
		# If the foot strike occurs within the final 350 ms, then accept the foot strike
		if FS_initial[0] > len(likelihood) - 350:
			out[FS_initial[0],0] = 1

			return out
		
		else:
			return [-1, -1]

	if len(FS_initial) == 0 and len(FO_initial == 1):
		# If the foot off occurs within the first 200 ms, then accept the foot off
		if FO_initial[0] < 200:
			out[FO_initial[0],1] = 1

			return out
		
		else:
			return [-1, -1]


	# There has been a mistake
	if len(FO_initial) > 1 and len(FS_initial) == 0:
		return [-1, -1]
	
	if len(FS_initial) > 1 and len(FO_initial) == 0:
		return [-1, -1]

	# Run through each of the FS and FO arrays to see if there are any peaks that should be combined (less than 35 ms apart)
	FS_initial2 = []
	do_not_append = []
	for i in range(1, len(FS_initial)):
		if FS_initial[i] - FS_initial[i-1] < 35: # If they are very close

			FS_temp_min = FS_initial[i-1] + argrelextrema(likelihood[FS_initial[i-1]:FS_initial[i],0]+1, np.less)[0]
			if len(FS_temp_min) > 0:
				if FS_temp_min[0] > 0.1:

					# Remove the previous first event and replace with one 3/4s between the two same events (towards the greatest point)
					max_point = max([FS_initial[i], FS_initial[i-1]])

					if max_point == FS_initial[i]: # If it is the second one
						temp = int((FS_initial[i] - FS_initial[i-1]) * 3/4) + FS_initial[i-1]
					else:
						temp = int((FS_initial[i] - FS_initial[i-1]) * 1/4) + FS_initial[i-1]
					
					# Add both to the do not append list
					do_not_append.append(FS_initial[i])
					do_not_append.append(FS_initial[i-1])

				else:
					temp = max([FS_initial[i], FS_initial[i-1]])

					if temp == FS_initial[i]: # If it is the second one
						# Add the previous point to the do not append list
						do_not_append.append(FS_initial[i-1])
					else:
						# Add current point to the do not append list
						do_not_append.append(FS_initial[i])
			else:
				temp = max([FS_initial[i], FS_initial[i-1]])

				if temp == FS_initial[i]: # If it is the second one
					# Add the previous point to the do not append list
					do_not_append.append(FS_initial[i-1])
				else:
					# Add current point to the do not append list
					do_not_append.append(FS_initial[i])

			FS_initial2.append(temp)


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
		if FO_initial[i] - FO_initial[i-1] < 35: # If they are very close

			FO_temp_min = FO_initial[i-1] + argrelextrema(likelihood[FO_initial[i-1]:FO_initial[i]+1,1], np.less)[0]
			if len(FO_temp_min) > 0:
				if FO_temp_min[0] > 0.1:

					# Remove the previous first event and replace with one 3/4s between the two same events (towards the greatest point)
					max_point = max([FO_initial[i], FO_initial[i-1]])

					if max_point == FO_initial[i]: # If it is the current point
						temp = int((FO_initial[i] - FO_initial[i-1]) * 3/4) + FO_initial[i-1]
					else:
						temp = int((FO_initial[i] - FO_initial[i-1]) * 1/4) + FO_initial[i-1]
					
					# Add both to the do not append list
					do_not_append.append(FO_initial[i])
					do_not_append.append(FO_initial[i-1])
				else:
					temp = max([FO_initial[i], FO_initial[i-1]]) #fix TODO regarding do_not_append

					if temp == FO_initial[i]: # If it is the second one
						# Add the previous point to the do not append list
						do_not_append.append(FO_initial[i-1])
					else:
						# Add current point to the do not append list
						do_not_append.append(FO_initial[i])
			else:
				temp = max([FO_initial[i], FO_initial[i-1]]) #fix TODO regarding do_not_append

				if temp == FO_initial[i]: # If it is the second one
					# Add the previous point to the do not append list
					do_not_append.append(FO_initial[i-1])
				else:
					# Add current point to the do not append list
					do_not_append.append(FO_initial[i])

			FO_initial2.append(temp)

		else:
			if FO_initial[i-1] not in do_not_append:
				FO_initial2.append(FO_initial[i-1])
			else:
				pass

	if FO_initial[-1] not in do_not_append and FO_initial[-1] not in FO_initial2:
		FO_initial2.append(FO_initial[-1])

	FS_initial = np.array(FS_initial2)
	FO_initial = np.array(FO_initial2)

	# Check which comes first
	if FS_initial[0] < FO_initial[0]:
		FS = 1
		FO = 0
	else:
		FO = 1
		FS = 1
	
	reset = 1

	while reset == 1:
		reset = 0

		if FO == 1:
			FS_temp = []
			for i in range(len(FO_initial)):
				try:
					FS_for_FO = FS_initial[np.logical_and(FS_initial>FO_initial[i], FS_initial<FO_initial[i+1])]

					if len(FS_for_FO) > 0:
						# Should be taking the first value
						condition_met = 0
						j = 0
						while condition_met == 0:
							try:
								if (FS_for_FO[j] - FO_initial[i] < 200) and (FS_for_FO[j] - FO_initial[i] > 35):
									FS_temp.append(FS_for_FO[j])
									condition_met = 1
							except IndexError:
								# FO is likely to be bad
								FO_initial[i] = -1
								condition_met = 1

								if i == 0: # The fact that FS is first may be incorrect
									reset = 1

									# Check which comes first
									if FS_initial[0] < FO_initial[1]:
										FS = 1
										FO = 0
									
									else:
										FO = 1
										FS = 0

							j += 1
					else:
						# Remove the FO value (there is no FS for it and it is not the last one) - shouldn't be here
						FO_initial[i] = -1
					
				except IndexError:
					FS_for_FO = FS_initial[FS_initial>FO_initial[i]]

					if len(FS_for_FO) > 0:
					# Should be taking the first value
						condition_met = 0
						j = 0
						while condition_met == 0:
							try:
								if (FS_for_FO[j] - FO_initial[i] < 200) and (FS_for_FO[j] - FO_initial[i] > 35):
									FS_temp.append(FS_for_FO[j])
									condition_met = 1
							except IndexError:
								# FO is likely to be bad
								FO_initial[i] = -1
								condition_met = 1

								if i == 0: # The fact that FS is first may be incorrect
									reset = 1

									# Check which comes first
									if FS_initial[0] < FO_initial[1]:
										FS = 1
										FO = 0
									
									else:
										FO = 1
										FS = 0

							j += 1
			FO_temp = FO_initial[FO_initial != -1]
		
		elif FS == 1:
			FO_temp = []
			for i in range(len(FS_initial)):
				try:
					FO_for_FS = FO_initial[np.logical_and(FO_initial>FS_initial[i], FO_initial<FS_initial[i+1])]

					if len(FO_for_FS) > 0:
						# Should be taking the first value
						condition_met = 0
						j = 0
						while condition_met == 0:
							try:
								if (FO_for_FS[j] - FS_initial[i] < 350) and (FO_for_FS[j] - FS_initial[i] > 160):
									FO_temp.append(FO_for_FS[j])
									condition_met = 1
							except IndexError:
								# FS is likely to be bad
								FS_initial[i] = -1
								condition_met = 1

								if i == 0: # The fact that FS is first may be incorrect
									reset = 1

									# Check which comes first
									if FS_initial[1] < FO_initial[0]:
										FS = 1
										FO = 0

									else:
										FO = 1
										FS = 0

							j += 1
					else:
						# Remove the FS value (there is no FO for it and it is not the last one) - shouldn't be here
						FS_initial[i] = -1
					
				except IndexError:
					FO_for_FS = FO_initial[FO_initial>FS_initial[i]]

					if len(FO_for_FS) > 0:
						# Should be taking the first value
						condition_met = 0
						j = 0
						while condition_met == 0:
							try:
								if (FO_for_FS[j] - FS_initial[i] < 350) and (FO_for_FS[j] - FS_initial[i] > 160):
									FO_temp.append(FO_for_FS[j])
									condition_met = 1
							except IndexError:
								# FS is likely to be bad
								FS_initial[i] = -1
								condition_met = 1

								if i == 0: # The fact that FS is first may be incorrect
									reset = 1

									# Check which comes first
									if FS_initial[1] < FO_initial[0]:
										FS = 1
										FO = 0
									
									else:
										FO = 1
										FS = 0

							j += 1
			FS_temp = FS_initial[FS_initial != -1]

	difference = len(FS_temp) - len(FO_temp)
	# If difference < 0, more FO events predicted.
	# If difference > 0, more FS events predicted.

	if difference > 0: # There are more FS events than FO events, will have started with FS
		# See if there are any FS's which have another FS before the next FO. If there is, take the first FS
		FS_diff = []
		for i in range(len(FS_temp)):
			FS_diff.append(FO_temp - FS_temp[i])

			for i in range(len(FS_diff)):
				neg_count = len(list(filter(lambda x: (x < 0), FS_diff[i].tolist())))
			if neg_count != i:
				# Then the value of the current FS is wrong
				FS_temp = np.delete(FS_temp, i)

		# If FO = 1, the 1st row of FO_diff should contain 0 negatives, 2nd 1, 3rd 2 and etc...
	elif difference < 0: # There are more FO events than FS events, will have started with FO
		# See if there are any FO's which have another FO before the next FS. If there is, take the second FO
		FO_diff = []
		for i in range(len(FO_temp)):
			FO_diff.append(FS_temp - FO_temp[i])

		for i in range(len(FO_diff)):
			neg_count = len(list(filter(lambda x: (x < 0), FO_diff[i].tolist())))
			if neg_count != i:
				# Then the value of FO previous is wrong
				FO_temp = np.delete(FO_temp, i-1)

	# Add FS and FO locations to the output array
	out[FS_temp,0] = 1
	out[FO_temp,1] = 1

	return out

# Compare predicted to true
def peak_cmp(true, predicted):
	'''
	This function returns the difference between a true event and a predicted event

	'''
	# If an error occured
	if type(predicted) == list:
		if predicted == [-1,-1]:
			return -1

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

		if i == 17:
			stop = 1

		est_events = peak_det(likelihood[i,:,1:], 0.15) # Foot strike and foot off

		sdist.append(peak_cmp(true[i,:,1:], est_events))

		if plot:
			plt.plot(est_events) # continous likelihood process
			plt.plot(true[i,:,1:]) # spikes on events
			plt.title(patient)
			plt.show()

	return sdist


def show_results(sdist: list, name: str = None):
	'''
	This function uses the output of eval_predictions and presents the results in a statistical manner

	'''

	# Number of samples which failed
	fail = len(np.where(np.array(sdist) == -1)[0])
	print('Number of failed samples = {}/{}'.format(fail,len(sdist)))

	# Remove the failed samples from the list
	to_remove = np.where(np.array(sdist) == -1)[0]
	for i in range(fail):
		del sdist[to_remove[i]-i]
	
	# Convert list of tuples to a list containing all FS and one containing all FO
	sample_FS = []
	sample_FO = []

	for sample in sdist:
		sample_FS.append(sample[0])
		sample_FO.append(sample[1])

	FS = [item for sublist in sample_FS for item in sublist]
	FO = [item for sublist in sample_FO for item in sublist]

	events = [FS, FO]
	names = ['Foot Strike', 'Foot Off']

	# Produce boxplots
	boxplot(events=events, names=names, name=name, absolute=False)


def boxplot(events: list, names: list, name: str, absolute: str = False):

	filename = os.path.splitext(name)[0]+'.png'

	name = name.split('\\')[-1]
	model_name = name.split('.')[0]

	if absolute:
		events[0] = abs(np.array(events[0])).tolist()
		events[1] = abs(np.array(events[1])).tolist()

	# Create boxplots of differences between real and estimated events
	fig, axs = plt.subplots()

	bp = axs.boxplot(events, labels = names)
	
	plt.setp(bp['medians'], color='k')

	axs.set_ylabel('Time difference (ms)')
	axs.set_xlabel('')
	axs.set_title('Estimated Gait Events vs True Gait Events\n{}'.format(model_name))
	
	# Add points
	num_boxes = len(names)

	for i in range(num_boxes):
		y = events[i]
		x = np.random.normal(1+i, 0.04, size=len(y))
		plt.plot(x, y, 'r.', alpha=0.2)

	# Medians
	FS_median = np.quantile(events[0], 0.5)
	FO_median = np.quantile(events[1], 0.5)

	# Quantiles
	FS_UQ = np.quantile(events[0], 0.75)
	FS_LQ = np.quantile(events[0], 0.25)
	FO_UQ = np.quantile(events[1], 0.75)
	FO_LQ = np.quantile(events[1], 0.25)

	# Min and max
	FS_min = np.quantile(events[0], 0)
	FS_max = np.quantile(events[0], 1)

	FO_min = np.quantile(events[1], 0)
	FO_max = np.quantile(events[1], 1)

	# Number of samples
	FS_nsamples = len(events[0])
	FO_nsamples = len(events[1])

	plt.text(x=0.65, y=2, s='median = {}\nUQ = {}\nLQ = {}\nmax = {}\nmin = {}\nnsamples = {}'.format(FS_median, FS_UQ, FS_LQ, FS_max, FS_min, FS_nsamples), bbox=dict(edgecolor='k', facecolor='w'))
	plt.text(x=1.65, y=2, s='median = {}\nUQ = {}\nLQ = {}\nmax = {}\nmin = {}\nnsamples = {}'.format(FO_median, FO_UQ, FO_LQ, FO_max, FO_min, FO_nsamples), bbox=dict(edgecolor='k', facecolor='w'))
	
	fig.set_size_inches(15,8)
	plt.savefig(filename, dpi=400)
	plt.show()
