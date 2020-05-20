
def selected_columns(columns: list) -> (list, list):
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
		columns.insert(0, 'id')
	
	if 'time' not in columns:
		columns_num.append(1)
		columns.insert(1, 'time')

	# Sort columns in order (low to high)
	columns_num.sort()

	return columns, columns_num


def create_directories(new_directory: str, event: str, event_type: str, est_events: bool):
	import os

	isdir = os.path.isdir
	mkdir = os.mkdir

	# Create folder for the columns used (if it doesn't already exist)
	if isdir(new_directory):
		# See if subfolder exists
		if est_events:
			if isdir("{}{}_{}\\".format(new_directory, event, event_type)):
				user = input('Folder already exists for this extraction, do you wish to continue? (Y/N): ')

				if 'y' in user or 'Y' in user:
					pass
				else:
					return
			
			else:
				# Make the directory
				mkdir("{}{}_{}\\".format(new_directory, event, event_type))
	
		else:
			if isdir("{}force\\".format(new_directory)):
				user = input('Folder already exists for this extraction, do you wish to continue? (Y/N): ')

				if 'y' in user or 'Y' in user:
					pass
				else:
					return
			
			else:
				mkdir("{}force\\".format(new_directory))

	else:
		mkdir(new_directory)

		if est_events:
			mkdir("{}{}_{}\\".format(new_directory, event, event_type))
		else:
			mkdir("{}force\\".format(new_directory))
	
	if est_events:
		save_dir = "{}{}_{}\\".format(new_directory, event, event_type)
	else:
		save_dir = "{}force\\".format(new_directory)

	return save_dir


def extract_data(data_folder: str, columns: list, overlap = False, all: bool = True, est_events: bool = False, event: str = None, event_type: str = None):
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

	all: a boolean (either True or False). If true, feature extraction will be run using all the data, if
		 False, feature extraction will be run using the first trial, and then that we be used on all the
		 data.

	est_events: a boolean (either True or False). If True, features will be extracted to estimate whether
				an event occured or not within a 100 ms time frame. If False, features will be extracted
				to estimate vertical GRF for the entire timeseries.

	event: A string containing either FS or FO. This will indicate which event the user wants to predict on.
		NOTE: this is only necessary as an input if est_events is True.

	event_type: A string containing either binary or time. This will indicate which type of output the user wants.
		NOTE: this is only necessary as an input if est_events is True.
	
	Outputs:
	This function does not return anything. However, it does save *.csv files in appropriate folders (based off
	the columns chosen) which can be used to fit either a classification or regression model (depending on what
	task is required) - see model_fitting.py

	Alex Woodall

	Auckland Bioengineering Institute

	08/04/2020

	'''

	from tsfresh import extract_features, extract_relevant_features, select_features
	from tsfresh.feature_extraction import ComprehensiveFCParameters, MinimalFCParameters
	from tsfresh.feature_extraction.settings import from_columns
	from tsfresh.utilities.dataframe_functions import impute

	import pickle
	import numpy as np
	import pandas as pd
	import os

	# Load data
	try:
		if overlap:
			dataset = pickle.load(open(data_folder + "dataset_overlap.pkl", "rb"))

		else:
			dataset = pickle.load(open(data_folder + "dataset_no_overlap.pkl", "rb"))

	except FileNotFoundError:
		dataset = pickle.load(open(data_folder + "dataset_200.pkl", "rb"))

	# Number selected columns which the user chose to use for feature extraction
	columns, columns_num = selected_columns(columns)	

	# Create directories for saving
	new_directory = "{}{}\\".format(data_folder, ("_".join(map(str,columns_num))))
	save_dir = create_directories(new_directory, event, event_type, est_events)

	# Attempt to load features from the save directory.
	try:
		X_features = pd.read_csv("{}features.csv".format(save_dir), index_col=0) # DataFrame containing the features we want
		features_string = X_features.columns
		extraction_settings = from_columns(features_string) # These are the features that we will be using

		pre_extracted = True

	except FileNotFoundError: # File does not exist
		pre_extracted = False

	# List to append last uid's from each key (used when using all trials to extract features)
	uid_last = []

	# Iterate through all the trials in the dataset
	for key in dataset.keys():
		
		# Create the timeseries based on the user input columns
		for col in columns_num:
			if col == 0:
				timeseries = (dataset[key]['X'])[:,col] # Only true accelerations
			else:
				timeseries = np.vstack((timeseries, dataset[key]['X'][:,col]))

		# dataset[key].keys() = ['X', 'force', 'y_FS_binary', 'y_FO_binary', 'y_FS_time_to', 'y_FO_time_to']
		
		# Create y (real data output)
		if est_events: # If estimating events

			try:
				if event_type == 'binary':
					y = dataset[key]['y_{}_binary'.format(event)]

					# Convert to boolean (will remain boolean if already)
					y = (y == 1.0)

				elif event_type == 'time':
					y = dataset[key]['y_{}_time_to_next'.format(event)]

				else:
					print('Event type must either be binary or time')

					return
			
			except KeyError:
				print("Event must equal either 'FS' or 'FO'.")
				
				return	

		else: # Estimating forces
			# possible force = ['Fx', 'Fy', 'Fz'] Assuming z direction is vertical
			y = dataset[key]['y'][:,2]
		
		# Convert to pandas DataFrame/Series
		if type(timeseries) is np.ndarray:
			# Needs to be a pandas dataframe
			timeseries = pd.DataFrame(timeseries.T, columns=columns)

			# Convert ID column into integers
			timeseries = timeseries.astype({'id': int})

			if est_events:
				if event_type == 'binary':
					y = pd.Series(data=y, dtype=bool, name='events')
				elif event_type == 'time':
					y = pd.Series(data=y, dtype=float, name='events')
			else:
				# Change ID column to fit for regression method
				ID = (np.arange(0,len(timeseries))).astype(int)

				timeseries['id'] = ID
			
				y = pd.Series(data=y, dtype=float, name='Fz')

		# Save X full dataset
		timeseries.to_csv("{}{}_timeseries.csv".format(save_dir, key), index=True, header=True)

		# Extract features from the first trial and use those for the rest if all == True
		if not all:
			# Extract features using tsFRESH
			if not pre_extracted:
				print('Finding relevant features using {}'.format(key))
				X_filtered = extract_relevant_features(timeseries, y,
													column_id="id", column_sort="time",
													default_fc_parameters=ComprehensiveFCParameters())

				# Save filtered features
				X_filtered.to_csv("{}features.csv".format(save_dir), header=True)

				features_string = X_filtered.columns
				extraction_settings = from_columns(features_string) # These are the features that we will be using

				pre_extracted = True

			if pre_extracted:
				print('Using pre-extracted features for event = {}'.format(event))
				print(str(key))
				X_filtered = extract_features(timeseries,
											column_id="id", column_sort="time",
											kind_to_fc_parameters=extraction_settings)

			# Add start_time and mass column to dataframe
			if est_events:
				start_time = dataset[key]['X_starting_time']
				mass = dataset[key]['X_mass_sample']

				X_filtered.insert(0, "start_time", start_time, True)
				X_filtered.insert(1, "mass", mass, True)
			
			else:
				mass = dataset[key]['X_mass_all']
				X_filtered.insert(0, "mass", mass, True)

			# Save dataframes
			X_filtered.to_csv("{}{}_X.csv".format(save_dir, key), index=True, header=True)
			y.to_csv("{}{}_y.csv".format(save_dir, key), index=True, header=True)

		else:
			try:
				uid_change = timeseries_temp['id'].iloc[-1]

				uid_last.append(uid_change)

				timeseries['id'] = timeseries['id'] + uid_change + 1
				timeseries_temp = timeseries_temp.append(timeseries)
				y_temp = y_temp.append(y, ignore_index = True)

			except NameError: # *_temp DataFrames do not exist yet
				timeseries_temp = timeseries
				y_temp = y

	if all:
		print('Using all data to extract relevant features')
		
		# First remove any NaN values in y, this should only be at the end
		print('Extracting all features')
		X = extract_features(timeseries_temp,
							column_id="id", column_sort="time",
							default_fc_parameters=ComprehensiveFCParameters(),
							impute_function=impute)
		
		y = y_temp

		# Remove NaN index's from X and y
		remove_idx = pd.isnull(y.to_numpy()).nonzero()[0]
		y = y.drop(remove_idx)
		X = X.drop(remove_idx)
		
		print('Selecting relevant features')
		X_filtered = select_features(X, y)
				
		X_filtered.to_csv("{}features.csv".format(save_dir), header=True)
		
		# Now save individual datasets
		# Reload DataFrame
		X_features = pd.read_csv("{}features.csv".format(save_dir), index_col=0)
		
		# Index values
		names = X_features.index.values

		# Saving individual trials
		print('Saving features for each trial')
		start = 0
		i = 0
		for key in dataset.keys():
			try:
				end_temp = uid_last[i] # Name of the row

			except IndexError:
				# Last key
				end_temp = X_features.iloc[-1].name

			end = end_temp

			# Find the new end index accounting for removed values
			removed = True

			while removed:
				if end in remove_idx:
					end -= 1
				else:
					removed = False
			
			# end = the name of the row (NOT index) which is the last in the trial
			end_idx = np.where(names == end)[0][0]

			X_save = X_features.iloc[start:end_idx+1]
			X_save = X_save.reset_index(drop=True)

			y_save = y.iloc[start:end_idx+1]
			y_save = y_save.reset_index(drop=True)

			start = end_idx + 1
			i += 1

			# Add start_time and mass column to dataframe
			if est_events:
				start_time = dataset[key]['X_starting_time']
				mass = dataset[key]['X_mass_sample']

				# Remove those due to NaN's
				start_time_new = start_time[:len(X_save)]
				mass_new = mass[:len(X_save)]

				X_save.insert(0, "start_time", start_time_new, True)
				X_save.insert(1, "mass", mass_new, True)
			
			else:
				mass = dataset[key]['X_mass_all']

				# Remove those due to NaN's (should be zero for GRF estimation)
				mass_new = mass[:len(X_save)]
				X_save.insert(0, "mass", mass_new, True)

			# Save
			X_save.to_csv("{}{}_X.csv".format(save_dir, key), index=True, header=True)
			y_save.to_csv("{}{}_y.csv".format(save_dir, key), index=True, header=True)

	return


if __name__ == "__main__":

	# HS or TO
	event = 'FS'
	event_type = 'binary'

	data_folder = "C:\\Users\\alexw\\Desktop\\Harvard_data\\"

	# columns in X = ['id', 'time', 'ax_l', 'ay_l', 'az_l', 'ax_r', 'ay_r', 'az_r',
	# 				'ax_diff', 'ay_diff', 'az_diff', 'a_res_l', 'a_res_r', 'a_res_diff']
	columns = ['id', 'time', 'a_res_l', 'a_res_r'] # Columns that we want to use

	extract_data(data_folder=data_folder, columns=columns, all=True, est_events=False)#, event=event, event_type=event_type)
