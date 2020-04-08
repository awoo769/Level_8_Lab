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

def learn(X: (dict, pd.DataFrame), y: (dict, pd.Series), data_folder: str, test_split: float = None):
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

	Alex Woodall

	Auckland Bioengineering Institute

	08/04/2020

	'''
	if type(X) is pd.DataFrame:
		# Learning using one trial (or a combination into a DataFrame rather than a dictionary of DataFrames)
		
		try:
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split)

			cl = RandomForestClassifier(n_estimators=128)

			cl.fit(X_train, y_train)

			y_predict = cl.predict(X_test)
			y_predict = pd.Series(y_predict, index=X_test.index)

			y_predict.to_csv("{}y_predict.csv".format(data_folder), index=True, header=True)
			y_test.to_csv("{}y_test.csv".format(data_folder), index=True, header=True)

			score = roc_auc_score(y_test, y_predict)
			print("Roc auc = {}\n".format(score))

			conf_mat = confusion_matrix(y_test, y_predict)
			print(conf_mat)

		except ValueError: # Should be a regression model

			split_int = int(len(X) * (1 - test_split))

			X_train = X.head(split_int)
			y_train = y.head(split_int)
			X_test = X.tail(len(X) - split_int) 
			y_test = y.tail(len(X) - split_int)

			rg = RandomForestRegressor(n_estimators=20)

			rg.fit(X_train, y_train)

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

			y_predict = pd.Series(y_predict, index=X_test.index)

			y_predict.to_csv("{}y_predict.csv".format(data_folder), index=True, header=True)
			y_test.to_csv("{}y_test.csv".format(data_folder), index=True, header=True)
			
			score = r2_score(y_test, y_predict)
			print("R2 = {}\n".format(score))

			# Plot result
			plt.plot(y_test.tail(1000),'k', label='True data')
			plt.plot(y_predict.tail(1000),'r', label='Estimate data')
			plt.legend()
			plt.ylabel('Force (N)')
			plt.xlabel('Time (ms)')
			plt.title('Estimated data')

			score = round(score, 4)
			
			plt.savefig('{}{}.png'.format(data_folder, '_'.join(str(score).split('.'))))
			plt.show()
	
	elif type(X) is dict:
		group_num = np.arange(len(groups))

		logo = LeaveOneGroupOut()
		logo.get_n_splits(groups=group_num)

		try:
			roc = []
			for train_index, test_index in logo.split(X=X_dictionary, groups=group_num):
				cl = RandomForestClassifier(n_estimators=64)

				# Training data
				print('Creating train datasets')
				for index in train_index:
					X_train = X_dictionary[groups[index]]
					y_train = y_dictionary[groups[index]]

					print('Fitting to model\nDataset: {}'.format(index))
					cl.fit(X_train, y_train)
				
				# Testing data
				print('Creating test datasets')
				for index in test_index:
					X_test = X_dictionary[groups[index]]
					y_test = y_dictionary[groups[index]]
				
				# Predict
				print('Predicting')
				y_estimate_test = cl.predict(X_test)
				y_estimate_test = pd.Series(y_estimate_test, index=X_test.index)

				roc.append(roc_auc_score(y_test, y_estimate_test))

				print(roc)

				# Save estimate
				y_estimate_test.to_csv("{}y_estimate_test_{}.csv".format(data_folder, index), index=True, header=True)

			print("Mean score (using roc auc score): {}".format(statistics.mean(roc)))

			f = open(data_folder + "cl.pkl", "wb")
			pickle.dump(cl, f)
			f.close()
		
		except ValueError:
			j = 0
			r2 = []

			for train_index, test_index in logo.split(X=X_dictionary, groups=group_num):
				rg = RandomForestRegressor(n_estimators=64)
				
				# Training data
				print('Creating train datasets')
				for index in train_index:
					X_train = X_dictionary[groups[index]]
					y_train = y_dictionary[groups[index]]

					print('Fitting to model\nDataset: {}'.format(index))
					rg.fit(X_train, y_train)
				
				# Testing data
				print('Creating test datasets')
				for index in test_index:
					X_test = X_dictionary[groups[index]]
					y_test = y_dictionary[groups[index]]

				# Fit to model
				#print('Fitting to model')
				#rg.fit(X_train, y_train)

				# Predict
				print('Predicting')
				y_estimate_test = rg.predict(X_test)
				y_estimate_test = pd.Series(y_estimate_test, index=X_test.index)

				# Print result
				plt.plot(y_test,'b.', label='True data')
				plt.plot(y_estimate_test,'r.', label='Estimate data')
				plt.legend()
				plt.ylabel('Force (N)')
				plt.xlabel('Time (ms)')
				plt.title('Estimated data')
				#plt.show()
				plt.savefig(data_folder + 'trial_{}.png'.format(j))
				j += 1
				r2.append(r2_score(y_test, y_estimate_test))

				print(r2)

			print("Mean score (using mean squared error): {}".format(statistics.mean(r2)))

			f = open(data_folder + "rg.pkl", "wb")
			pickle.dump(rg, f)
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
	columns = ['id', 'time', 'ax_l', 'ay_l', 'az_l', 'ax_r', 'ay_r', 'az_r']
	#columns = ['id', 'time', 'ax_diff', 'ay_diff', 'az_diff', 'a_res_diff'] # Columns that we want to use

	directory = get_directory(initial_directory=data_folder, columns=columns)#, est_events=True, event=event)

	# Load features (after extract data has been run)
	X_dictionary, y_dictionary, groups = load_features(data_folder, directory, est_events=False)

	# NaN being created. Try to use all datasets to get features
	X = X_dictionary[groups[0]]
	y = y_dictionary[groups[0]]

	test_split = 0.33
	learn(X, y, directory, test_split)
