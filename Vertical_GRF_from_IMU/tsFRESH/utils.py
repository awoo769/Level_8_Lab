import numpy as np
import pandas as pd
import pickle
import os
import re
import csv

def read_csv(filename: str):
	'''
	This function opens and reads a csv file, returning a numpy array (data) of the contents.

	'''

	with open(filename, 'r') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')

		data = []
		i = 0
		
		for row in reader:
			i += 1

			# First data row on line 8
			if i >= 8:
				if len(row) != 0:
					data.append(row)

	return np.array(data)


def rezero_filter(original_fz: np.ndarray, threshold: int = 20):
	'''
	Resets all values which were originally zero to zero

	Inputs:	original_fy: an array of unfiltered y data

	Outputs:	filter_plate: an array corresponding to a mask. '1' if above a threshold to keep,
				'0' if below a threshold and will be set to 0

	Original version in MATLAB written by Duncan Bakke

	'''

	filter_plate = np.zeros(np.shape(original_fz))

	# Binary test for values greater than 20
	force_zero = (original_fz > threshold) * 1 # Convert to 1 or 0 rather than True or False

	# We do not want to accept values which are over 20 but considered 'noise'.
	# Must be over 20 for more than 35 frames in a row. (Robberechts, 2019) Therefore, need to search for
	# [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] x 3.5 and get all the indices that meet this condition

	# Convert to string to test this condition
	force_str = ''.join(list(map(str, force_zero)))

	# Find all occurrences where the pattern occurs
	true_inds = [m.start() for m in re.finditer('(?=11111111111111111111111111111111111)', force_str)]

	# true_inds will not include the ends (e.g., 11...11100000) - will not include the final 3 1's
	extra_inds = [i + 35 for i in true_inds[0:-1]] # So make another array with 35 added on to all but the last value
	
	# Return the 'filtered' rezeroing array
	filter_plate[true_inds] = 1
	filter_plate[extra_inds] = 1

	# For values at the beginning of the array that should be there but are not counted.
	# For the values that are 1, make the next 35 values also 1.
	for i in true_inds:
		if i > 35:
			break
		else:
			filter_plate[i:i+35] = 1
	
	# Search string to see if there are any situations where the gap between consecutive 1's is less than 35.
	# If there is, make all values 1 within this section.
	previous_ind = -np.Inf
	for i in range(len(filter_plate)):
		if filter_plate[i] == 1:
			current_ind = i
		
			if current_ind - previous_ind < 35:
				filter_plate[previous_ind:current_ind+1] = 1
		
			previous_ind = current_ind

	return filter_plate


def load_features(base_data_folder: str, data_folder: str, est_events: bool =False, overlap: bool = False) -> (dict, dict, list): 

	if data_folder == -1:
		return -1, -1, -1

	# Dictionaries for data to be stored in
	X_dictionary = {}
	y_dictionary = {}

	# To use the keys in the dictionary
	if overlap:
		dataset = pickle.load(open("{}overlap_dataset.pkl".format(base_data_folder), "rb"))
	
	else:
		dataset = pickle.load(open("{}dataset_no_overlap.pkl".format(base_data_folder), "rb"))

	for key in dataset.keys():
		# pd.DataFrame
		X = pd.read_csv(data_folder + "{}_X.csv".format(key), index_col=0)

		# pd.Series
		if est_events:
			name = 'events'
		else:
			name = 'Fz'

		y = pd.read_csv("{}{}_y.csv".format(data_folder, key), index_col=0)[name]

		X_dictionary[key] = X
		y_dictionary[key] = y
	
		groups = list(dataset.keys())

	return X_dictionary, y_dictionary, groups


def get_directory(initial_directory: str, columns: list, est_events: str = False, event: str = None, event_type: str = None) -> str:

	columns_in_X = ['id', 'time', 'ax_l', 'ay_l', 'az_l', 'ax_r', 'ay_r', 'az_r',
		 				'ax_diff', 'ay_diff', 'az_diff', 'a_res_l', 'a_res_r', 'a_res_diff']
		
	columns_num = []

	count = 0
	for col in columns_in_X:
		if col in columns:
			columns_num.append(count)
		
		count += 1

	if 'time' not in columns:
		# Add a time column
		columns_num.append(1)

	if 'id' not in columns:
		columns_num.append(0)

	columns_num.sort()

	new_directory = "{}{}\\".format(initial_directory, ("_".join(map(str,columns_num))))
	
	final_dir = False
	while not final_dir:
		# Check that directory exists
		if os.path.isdir(new_directory):
			# See if subfolder exists
			if est_events:
				if os.path.isdir("{}{}_{}\\".format(new_directory, event, event_type)):
					final_dir = True
				
				else:
					print('Directory for this event for the chosen colums does not exist')
					print('Creating directory')

					os.mkdir("{}{}_{}\\".format(new_directory, event, event_type))

					final_dir = True

			else:
				if os.path.isdir("{}force\\".format(new_directory)):
					final_dir = True

				else:
					print('Directory for this event for the chosen colums does not exist')
					print('Creating directory')
					os.mkdir("{}force\\".format(new_directory))

					final_dir = True
		else:
			print('Directory for these columns does not exist')
			print('Creating directory')
			os.mkdir(new_directory)
	
	if est_events:
		save_dir = "{}{}_{}\\".format(new_directory, event, event_type)
	else:
		save_dir = "{}force\\".format(new_directory)

	return save_dir


def interpolate_data(time: np.ndarray, x: np.ndarray):
	from scipy import interpolate
	import numpy as np

	# Assumes the time array is in SI units
	ninterpolates_points = int((time[-1] - time[0]) * 1000) + 1

	# Create the new time array for interpolation
	new_t = np.linspace(time[0], time[-1], ninterpolates_points)
	reversed_axes = False
	# x should have shape (n, len(time))
	try:
		assert x.shape[1] == len(time)
	except Exception as e:
		
		if isinstance(e, IndexError):
			# 1 D
			x = x[np.newaxis,:]

		elif isinstance(e, AssertionError):
		
				try:
					assert x.shape[0] == len(time)

					x = np.swapaxes(x, 1, 0)
					reversed_axes = True

				except AssertionError:
					print('Dimensions in time array and data do not align')

	new_x = np.zeros(np.shape(x))
	i = 0
	for item in x:
		tck_x = interpolate.splrep(time, item, s=0)
		new_item = interpolate.splev(new_t, tck_x, der=0)

		new_x[i,:] = new_item

		i += 1
	
	if reversed_axes:
		new_x = np.swapaxes(new_x, 1, 0)
	
	return new_t, new_x



def resolve_overlap(y: pd.Series, start_times: list) -> pd.Series:


	# Ignoring any predicted times in y which are less than 10 and greater than 90 (edge cases)
	y = y.where(y >= 10, np.NaN)
	y = y.where(y <= 90, np.NaN)

	# List of values which correspond to
	# 0 - 99, 100 - 199, etc (no overlap)
	y_without_overlap = y.iloc[0::2]
	y_overlap_values = y.iloc[1::2]


	# A list which will have estimates for sample
	estimates = []

	for i in range(0, len(start_times), 2):
		# each sample contains 100 ms of data, 50 ms of overlap from the previous data.

		# First sample estimate
		est0 = y.iloc[i]

		if i == len(start_times) - 1:
			tu2=1

		
		elif i == len(start_times) - 2:
			if est0 < 50 and est0 is not np.NaN:
				if y.iloc[i-1] > 49:
					est1 = y.iloc[i-1] - 50
				else:
					est1 = np.NaN


		elif est0 < 50 and est0 is not np.NaN and i == 0:
			mean_est = est0
		
		elif est0 < 50 and est0 is not np.NaN:
			if y.iloc[i-1] > 49:
				est1 = y.iloc[i-1] - 50
			else:
				est1 = np.NaN

		elif est0 > 50:
			# Can only look forward
			if y.iloc[i+1] < 50:
				est1 = y.iloc[i+1] + 50
			else:
				est1 = np.NaN
		
			# Take mean
			mean_est = np.nanmean(np.array([est0, est1]))

		
			
			

		estimates.append(mean_est)

