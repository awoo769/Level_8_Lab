from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneGroupOut

from sklearn.metrics import roc_auc_score # For classification
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score # For regression

import statistics
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import pickle
import os
from scipy import signal

from utils import get_directory, load_features, rezero_filter

def learn(X: (dict, pd.DataFrame), y: (dict, pd.Series), data_folder: str, groups: list, test_split: float = None, name: str = None):
	'''
	This function trains either a classification or regression random forest model. It is able to handle
	either a singular pandas DataFrame or a dictionary of pandas DataFrames. If the input is a singular
	pandas DataFrame, the rows will be split into a training and testing dataset using test_split (0 - 1).
	If the input is a dictionary of pandas DataFrames, a leave one out method will be used to verify the
	models accuracy.

	Inputs:

	X: a dictionary of pandas DataFrames or a singular pandas DataFrame

	y: a dictionary of pandas Series of a singular pandas Series

	data_folder: the location of where to save the output

	test_split: the decimal percentage to split the training and testing datasets
				NOTE: this is only required if the X/y input is not a dictionary

	none: the name of the trial
		  NOTE: this is only required if the X/y input is not a dictionary

	Alex Woodall

	Auckland Bioengineering Institute

	08/04/2020

	'''

	if 'force' in data_folder or 'time' in data_folder:
		mode = 'regression'
	
	elif 'binary' in data_folder:
		mode = 'classification'


	if type(X) is pd.DataFrame:
		# Learning using one trial (or a combination into a DataFrame rather than a dictionary of DataFrames)
		
		if mode == 'classification':
			# Split into training and testing
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split)

			# Create classifier and train
			cl = RandomForestClassifier(n_estimators=128, n_jobs=-1)
			cl.fit(X_train, y_train)

			# Predict on classifier and convert to a pandas series, save output
			y_predict = cl.predict(X_test)
			y_predict = pd.Series(y_predict, index=X_test.index)

			y_predict.to_csv("{}y_predict.csv".format(data_folder), index=True, header=True)
			y_test.to_csv("{}y_test.csv".format(data_folder), index=True, header=True)

			# Print score and confusion matrix
			score = roc_auc_score(y_test, y_predict)
			conf_mat = confusion_matrix(y_test, y_predict)

			print("Roc auc = {}\n".format(score))
			print(conf_mat)

		elif mode == 'regression':
			# Split into training and testing
			split_int = int(len(X) * (1 - test_split))

			X_train = X.head(split_int)
			y_train = y.head(split_int)
			X_test = X.tail(len(X) - split_int) 
			y_test = y.tail(len(X) - split_int)

			# Create regressor and train
			rg = RandomForestRegressor(n_estimators=20, n_jobs=-1)
			rg.fit(X_train, y_train)

			# Predict
			y_predict = rg.predict(X_test)

			# Filter force array
			''' Filter force plate data at 60 Hz '''
			analog_frequency = 1000
			cut_off = 60 # Derie (2017), Robberechts et al (2019)
			order = 2 # Weyand (2017), Robberechts et al (2019)
			b_f, a_f = signal.butter(N=order, Wn=cut_off/(analog_frequency/2), btype='low')

			new_F = signal.filtfilt(b_f, a_f, y_predict)

			''' Rezero filtered forces'''
			threshold = 20 # 20 N
			filter_plate = rezero_filter(original_fz=new_F, threshold=threshold)
			
			y_predict = filter_plate * new_F

			# Convert output into a pandas series and save
			y_predict = pd.Series(y_predict, index=X_test.index)
			y_predict.to_csv("{}y_predict.csv".format(data_folder), index=True, header=True)
			y_test.to_csv("{}y_test.csv".format(data_folder), index=True, header=True)
			
			# Calculate R2 score and print
			score = r2_score(y_test, y_predict)
			print("R2 = {}\n".format(score))

			# Plot result
			plt.plot(y_test.tail(1000),'k', label='True data')
			plt.plot(y_predict.tail(1000),'r', label='Estimate data')
			plt.legend()
			plt.ylabel('Force (N)')
			plt.xlabel('Time (ms)')
			plt.title('Estimated data for {}'.format(name))

			# Save figure
			score = round(score, 4)
			
			plt.savefig('{}{}_{}.png'.format(data_folder, name, '_'.join(str(score).split('.'))))
			plt.show()
	
	elif type(X) is dict:

		# Create leave one group out split
		group_num = np.arange(len(groups))
		logo = LeaveOneGroupOut()
		logo.get_n_splits(groups=group_num)

		if mode == 'classification':
			# Create results text file
			f = open("{}results.txt".format(data_folder), "w")
			f.write("Results for classification\n\n")
			f.close()

			roc = []
			# Train on n - 1 groups, test on 1. Repeat for all
			for train_index, test_index in logo.split(X=X, groups=group_num):
				cl = RandomForestClassifier(n_estimators=128, n_jobs=-1)

				# Training data
				print('Hold out trial: {}'.format(groups[test_index[0]]))

				for index in train_index:
					try:
						X_train = X_train.append(X[groups[index]], ignore_index = True)
						y_train = y_train.append(y[groups[index]], ignore_index = True)
					
					except NameError:
						X_train = X[groups[index]]
						y_train = y[groups[index]]

				cl.fit(X_train, y_train)
				
				# Testing data
				X_test = X[groups[test_index[0]]]
				y_test = y[groups[test_index[0]]]
				
				# Predict
				y_estimate_test = cl.predict(X_test)
				y_estimate_test = pd.Series(y_estimate_test, index=X_test.index)

				roc.append(roc_auc_score(y_test, y_estimate_test))

				conf = confusion_matrix(y_test, y_estimate_test)

				np.savetxt("{}y_estimate_conf_{}.txt".format(data_folder, groups[test_index[0]]), conf, delimiter='\t', fmt='%i')

				f = open("{}results.txt".format(data_folder), "a")
				f.write("Predicting on {}: {}\n".format(groups[test_index[0]], round(roc[-1], 4)))
				f.close()

				# Save estimate
				y_estimate_test.to_csv("{}y_estimate_test_{}.csv".format(data_folder, groups[test_index[0]]), index=True, header=True)

				# Remove datasets
				del X_train
				del X_test
				del y_train
				del y_test

				# Save model
				f = open("{}{}_cl.pkl".format(data_folder, groups[test_index[0]]), "wb")
				pickle.dump(cl, f)
				f.close()

			f = open("{}results.txt".format(data_folder), "a")
			f.write("\nAverage roc auc score: {}".format(round(statistics.mean(roc), 4)))
			f.close()

		elif mode == 'regression':

			# Allow for different number of estimators depending on task
			if 'force' in data_folder:
				n_estimators = 10
			
			else:
				n_estimators = 10

			# Create results text file
			f = open("{}results.txt".format(data_folder), "w")
			f.write("Results for regression\n\n")
			f.close()

			r2 = []

			for train_index, test_index in logo.split(X=X, groups=group_num):
				rg = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1)
				
				# Training data
				print('Hold out trial: {}'.format(groups[test_index[0]]))

				for index in train_index:
					try:
						X_train = X_train.append(X[groups[index]], ignore_index = True)
						y_train = y_train.append(y[groups[index]], ignore_index = True)
					
					except NameError:
						X_train = X[groups[index]]
						y_train = y[groups[index]]

				rg.fit(X_train, y_train)
				
				# Testing data
				X_test = X[groups[test_index[0]]]
				y_test = y[groups[test_index[0]]]
				
				# Predict
				y_estimate_test = rg.predict(X_test)
				
				# Round estimate to a whole number
				y_estimate_test = np.around(y_estimate_test)
				
				# Any negative number = -1
				y_estimate_test[y_estimate_test < 0] = -1
				
				y_estimate_test = pd.Series(y_estimate_test, index=X_test.index)

				r2.append(r2_score(y_test, y_estimate_test))

				f = open("{}results.txt".format(data_folder), "a")
				f.write("Predicting on {}: {}\n".format(groups[test_index[0]], round(r2[-1], 4)))
				f.close()

				# Save estimate
				y_estimate_test.to_csv("{}y_estimate_test_{}.csv".format(data_folder, groups[test_index[0]]), index=True, header=True)

				# Remove datasets
				del X_train
				del X_test
				del y_train
				del y_test

				# Save model
				f = open("{}{}_rg.pkl".format(data_folder, groups[test_index[0]]), "wb")
				pickle.dump(rg, f)
				f.close()

			f = open("{}results.txt".format(data_folder), "a")
			f.write("\nAverage R^2 score: {}".format(round(statistics.mean(r2), 4)))
			f.close()

	else:
		print("X should be of type dict or pd.DataFrame")

		return
	
	return


if __name__ == "__main__":
	
	data_folder = "C:\\Users\\alexw\\Desktop\\tsFRESH\\data\\"
	event = 'HS'

	# columns in X = ['id', 'time', 'ax_l', 'ay_l', 'az_l', 'ax_r', 'ay_r', 'az_r',
	# 				'ax_diff', 'ay_diff', 'az_diff', 'a_res_l', 'a_res_r', 'a_res_diff']
	columns = ['id', 'time', 'a_res_l', 'a_res_r']
	#columns = ['id', 'time', 'ax_diff', 'ay_diff', 'az_diff', 'a_res_diff'] # Columns that we want to use

	directory = get_directory(initial_directory=data_folder, columns=columns, est_events=True, event=event)

	# Load features (after extract data has been run)
	X_dictionary, y_dictionary, groups = load_features(data_folder, directory, est_events=True)

	i = 6
	X = X_dictionary[groups[i]]
	y = y_dictionary[groups[i]]

	test_split = 0.33
	learn(X_dictionary, y_dictionary, directory, groups)
