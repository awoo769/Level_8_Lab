from tsfresh import extract_features, extract_relevant_features
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.feature_extraction.settings import from_columns

import pickle
import numpy as np
import pandas as pd
import os


def extract_data(data_folder: str, columns: list, est_events: bool = False, event: str = None):
	'''
	This function uses tsFRESH to extract relevant features for multiple machine learning tasks.
	If a csv file of features to use already exists (as features.csv), then those features will
	be used instead of finding relevant features from scratch (speeds up computing time).

	Inputs:

	data_folder: a string containing the location of the directory which the dataset.pkl is saved in.
				 This dataset it created using data_preperation.py.
	columns: a list of strings containing the columns from the dataset which the user wishes to extract 
			 features from. This includes: id, time, ax_l, ay_l, az_l, ax_r, ay_r, az_r,
		 	 ax_diff, ay_diff, az_diff, a_res_l, a_res_r, a_res_diff.
			 NOTE: if id or time are not included in this list, they will be automatically added as they
			 are necessary.
	est_events: a boolean (either True or False). If True, features will be extracted to estimate whether
				an event occured or not within a 100 ms time frame. If False, features will be extracted
				to estimate vertical GRF for the entire timeseries.
	event: A string containing either HS or TO. This will indicate which event the user wants to predict on.
		   NOTE: this is only necessary as an input if est_events is True.
	
	Outputs:
	This function does not return anything. However, it does save *.csv files in appropriate folders (based off
	the columns chosen) which can be used to fit either a classification or regression model (depending on what
	task is required) - see model_fitting.py

	Alex Woodall

	Auckland Bioengineering Institute

	08/04/2020

	'''

	# Load data
	dataset = pickle.load(open(data_folder + "dataset.pkl", "rb"))

	# Possible columns in dataset
	columns_in_X = ['id', 'time', 'ax_l', 'ay_l', 'az_l', 'ax_r', 'ay_r', 'az_r',
		 				'ax_diff', 'ay_diff', 'az_diff', 'a_res_l', 'a_res_r', 'a_res_diff']
	
	# Columns which are selected by the user input
	columns_num = []

	count = 0
	for col in columns_in_X:
		if col in columns:
			columns_num.append(count)
		
		count += 1

	# Add id or time columns if they were not chosen - needed
	
	if 'id' not in columns:
		columns_num.append(0)
	
	if 'time' not in columns:
		columns_num.append(1)

	# Sort columns in order (low to high)
	columns_num.sort()

	# Create directories for saving
	new_directory = "{}{}\\".format(data_folder, ("_".join(map(str,columns_num))))
	# Create folder for the columns used (if it doesn't already exist)
	if os.path.isdir(new_directory):
		# See if subfolder exists
		if est_events:
			if os.path.isdir("{}{}\\".format(new_directory, event)):
				user = input('Folder already exists for this extraction, do you wish to continue? (Y/N): ')

				if 'y' in user or 'Y' in user:
					pass
				else:
					return
			
			else:
				# Make the directory
				os.mkdir("{}{}\\".format(new_directory, event))
	
		else:
			if os.path.isdir("{}force\\".format(new_directory)):
				user = input('Folder already exists for this extraction, do you wish to continue? (Y/N): ')

				if 'y' in user or 'Y' in user:
					pass
				else:
					return
			
			else:
				os.mkdir("{}force\\".format(new_directory))

	else:
		os.mkdir(new_directory)

		if est_events:
			os.mkdir("{}{}\\".format(new_directory, event))
		else:
			os.mkdir("{}force\\".format(new_directory))
	
	if est_events:
		save_dir = "{}{}\\".format(new_directory, event)
	else:
		save_dir = "{}force\\".format(new_directory)

	# Attempt to load features from the save directory.
	try:
		X_features = pd.read_csv("{}features.csv".format(save_dir), index_col=0) # DataFrame containing the features we want
		features_string = X_features.columns
		extraction_settings = from_columns(features_string) # These are the features that we will be using

		pre_extracted = True

	except FileNotFoundError: # File does not exist
		extraction_settings = ComprehensiveFCParameters() # Use all the features and then find relevant one

		pre_extracted = False

	# Iterate through all the trials in the dataset
	for key in dataset.keys():
		
		# Create the timeseries based on the user input columns
		for col in columns_num:
			if col == 0:
				timeseries = (dataset[key]['X'])[:,col] # Only true accelerations
			else:
				timeseries = np.vstack((timeseries, dataset[key]['X'][:,col]))

		# dataset[key].keys() = ['X', 'force', 'y_HS_binary', 'y_TO_binary', 'y_HS_time_to', 'y_TO_time_to']
		
		# Create y (real data output)
		if est_events: # If estimating events
			if event == 'HS':
				y_temp = dataset[key]['y_HS_binary'] # For HS
				
			elif event == 'TO':
				y_temp = dataset[key]['y_HS_binary'] # For TO

			else:
				print("Event must equal either 'HS' or 'TO'.")
				
				return
			
			# Convert to boolean (will remain boolean if already)
			y = (y_temp == 1.0)

		else: # Estimating forces
			y = dataset[key]['force'][:,2] # possible force = ['Fx', 'Fy', 'Fz'] Assuming z direction is vertical
		
		# Convert to pandas DataFrame/Series
		if type(timeseries) is np.ndarray:
			# Needs to be a pandas dataframe
			timeseries = pd.DataFrame(timeseries.T, columns=columns)

			# Convert ID column into integers
			timeseries = timeseries.astype({'id': int})

			if est_events:
				y = pd.Series(data=y, dtype=bool, name='events')
			else:
				# Change ID column to fit for regression method
				ID = (np.arange(0,len(timeseries))).astype(int)

				timeseries['id'] = ID
			
				y = pd.Series(data=y, dtype=float, name='Fz')

		# Save X full dataset
		timeseries.to_csv("{}{}_timeseries.csv".format(save_dir, key), index=True, header=True)

		# Extract features using tsFRESH
		if not pre_extracted:
			print('Finding relevant features using {}'.format(key))
			X_filtered = extract_relevant_features(timeseries, y,
												column_id="id", column_sort="time",
												default_fc_parameters=extraction_settings)
			
			X_filtered.to_csv("features.csv".format(save_dir), header=True)

			features_string = X_filtered.columns
			extraction_settings = from_columns(features_string) # These are the features that we will be using

			pre_extracted = True

		if pre_extracted:
			print('Using pre-extracted features')
			print(str(key))
			X_filtered = extract_features(timeseries,
										column_id="id", column_sort="time",
										kind_to_fc_parameters=extraction_settings)

		# Save dataframes
		X_filtered.to_csv("{}{}_X.csv".format(save_dir, key), index=True, header=True)
		y.to_csv("{}{}_y.csv".format(save_dir, key), index=True, header=True)

	return


if __name__ == "__main__":

	# HS or TO
	event = 'HS'

	data_folder = "C:\\Users\\alexw\\Desktop\\tsFRESH\\data\\"

	# columns in X = ['id', 'time', 'ax_l', 'ay_l', 'az_l', 'ax_r', 'ay_r', 'az_r',
	# 				'ax_diff', 'ay_diff', 'az_diff', 'a_res_l', 'a_res_r', 'a_res_diff']
	columns = ['id', 'time', 'a_res_l', 'a_res_r'] # Columns that we want to use

	extract_data(data_folder=data_folder, columns=columns)#, est_events=True, event=event)
