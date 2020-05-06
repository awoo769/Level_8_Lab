import numpy as np

def read_csv(filename: str):
	'''
	This function opens and reads a csv file, returning a numpy array (data) of the contents.

	'''

	import csv
	import numpy as np

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
	
	import re
	import numpy as np

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

	import pickle
	import pandas as pd

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

	import os

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


def prepare_data(data: np.ndarray, sample_length: int, f: str, overlap: bool = False) -> (np.ndarray, np.ndarray, np.ndarray):
	'''
	This function creates the dataset of events in which features will be extracted from

	data: the data which will be split into samples of length sample_length
	dataset: the array which is being build of the samples from each trial
	HS_TO: a list of the truth values of the FS and FO events
	f: the name of the trial
	overlap: whether to overlap each sample by half

	'''

	import os
	from scipy import signal
	import numpy as np

	time = data[:,0].astype(np.float) # 1st column

	# Left foot
	a_l = (data[:,4:6+1].T).astype(np.float) # [ax, ay, az]

	# Right foot
	a_r = (data[:,7:9+1].T).astype(np.float) # [ax, ay, az]

	# Flip the x acceleration on the right foot. This will make the coordinate frames mirrored along the sagittal plane
	a_r[0] = -a_r[0]

	# Engineered timeseries
	a_diff = abs(a_l - a_r) # Difference between left and right
	a_res_l = np.linalg.norm(a_l, axis=0) # Left resultant
	a_res_r = np.linalg.norm(a_r, axis=0) # Right resultant
	a_res_diff = abs(a_res_l - a_res_r) # Difference between left and right resultant

	# Get force plate data for comparison
	F = (data[:,1:3+1].T).astype(np.float) #[Fx, Fy, Fz]; Fz = vertical

	# Rotate 180 deg around y axis (inverse Fx and Fz)
	F[0] = -F[0] # Fx
	F[2] = -F[2] # Fz

	''' Filter force plate data at 60 Hz '''
	analog_frequency = 1000
	cut_off = 60 # Derie (2017), Robberechts et al (2019)
	order = 2 # Weyand (2017), Robberechts et al (2019)
	b_f, a_f = signal.butter(N=order, Wn=cut_off/(analog_frequency/2), btype='low')

	new_F = np.zeros(np.shape(F))

	for i in range(len(F)):
		new_F[i,:] = signal.filtfilt(b_f, a_f, F[i,:])

	''' Rezero filtered forces'''
	threshold = 20 # 20 N
	filter_plate = rezero_filter(original_fz=new_F[2], threshold=threshold)
	
	for i in range(len(F)): # Fx, Fy, Fz
		new_F[i,:] = filter_plate * new_F[i,:]
	
	''' Ground truth event timings '''
	# Get the points where there is force applied to the force plate (stance phase). Beginning = heel strike, end = toe off
	heel_strike = []
	toe_off = []

	for i in range(1, len(new_F[2])-1):
		if (new_F[2])[i-1] == 0 and (new_F[2])[i] != 0:
			heel_strike.append(i-1)
		
		if (new_F[2])[i+1] == 0 and (new_F[2])[i] != 0:
			toe_off.append(i+1)

	# We now need to split each into individual samples of length sample_length ms (needs to be the
	# same size for ML). For the final timesteps that remain, add 0's to the end of HS_TO and 
	# continue with what the acceleration values are.

	# Initial time
	t = -np.Inf

	# Initial start and end periodss
	start_period = 0
	end_period = start_period + sample_length

	# Convert heel strike and toe off lists to arrays
	HS = np.array(heel_strike)
	TO = np.array(toe_off)

	# Unique id
	uid = 0
	
	# Temporary data lists
	acc_temp = []
	force_temp = []

	while t < time[-1]:

		# Append heel strikes and toe offs which are within the range
		sample_HS = np.intersect1d(HS[HS < end_period], HS[HS >= start_period])

		# Convert into new timeframe
		sample_HS = sample_HS - start_period

		sample_TO = np.intersect1d(TO[TO < end_period], TO[TO >= start_period])
		
		# Convert into new timeframe
		sample_TO = sample_TO - start_period

		# HS_TO will have 0's where there is no event and a 1 where there is.
		# Two columns per trial. 1st = HS, 2nd = TO
		temp_HS_TO = np.zeros((sample_length, 2))
		(temp_HS_TO[:,0])[sample_HS] = 1.0
		(temp_HS_TO[:,1])[sample_TO] = 1.0

		try:
			HS_TO = np.vstack((HS_TO, temp_HS_TO))
		except NameError: # HS_TO does not exist
			HS_TO = temp_HS_TO

		# Append accelerations which are within the range

		acc_temp.append([uid]*sample_length) # Unique id
		acc_temp.append(time[start_period:end_period]) # time

		for i in range(len(a_l)): # Left foot xyz
			acc_temp.append((a_l[i])[start_period:end_period])
		
		for i in range(len(a_r)): # Right foot xyz
			acc_temp.append((a_r[i])[start_period:end_period])
		
		for i in range(len(a_diff)): # Difference between left and right foot
			acc_temp.append((a_diff[i])[start_period:end_period])
		
		# Resultant accelerations
		acc_temp.append(a_res_l[start_period:end_period]) # Left foot
		acc_temp.append(a_res_r[start_period:end_period]) # Right foot
		acc_temp.append(a_res_diff[start_period:end_period]) # Difference between left and right foot

		for i in range(len(new_F)): # Filtered force plate data
			force_temp.append((new_F[i])[start_period:end_period])
		
		try:
			force = np.vstack((force, np.array(force_temp).T))
		except NameError: # force does not exist
			force = np.array(force_temp).T

		try:
			accelerations = np.vstack((accelerations, np.array(acc_temp).T))
		except NameError: # accelerations does not exist
			accelerations = np.array(acc_temp).T

		# New start and end period for next iteration
		if overlap:
			start_period = int(end_period - sample_length / 2)
		else:
			start_period = int(end_period)
		
		end_period = int(start_period + sample_length)
		
		# Update t for loop conditions
		try:
			t = time[end_period-1] # Time at the end of the sample (for loop exiting)
		except IndexError:
			t = time[-1]

		uid += 1 # Increase uid

		acc_temp = []
		force_temp = []
	
	# For the part of the timeseries which is left over
	final_sample_length = len(a_diff[0]) - start_period

	# Append heel strikes and toe offs which are within the range
	sample_HS = HS[HS >= start_period]

	# Convert into new timeframe
	sample_HS = sample_HS - start_period

	sample_TO = TO[TO >= start_period]
	
	# Convert into new timeframe
	sample_TO = sample_TO - start_period

	# HS_TO will have 0's where there is no event and a 1 where there is.
	# Two columns per trial. 1st = HS, 2nd = TO
	temp_HS_TO = np.zeros((final_sample_length, 2))
	(temp_HS_TO[:,0])[sample_HS] = 1.0
	(temp_HS_TO[:,1])[sample_TO] = 1.0

	HS_TO = np.vstack((HS_TO, temp_HS_TO))

	# Append accelerations which are within the range
	acc_temp.append([uid]*final_sample_length)
	acc_temp.append(time[start_period:])

	for i in range(len(a_l)): # Left foot xyz
		acc_temp.append((a_l[i])[start_period:end_period])
		
	for i in range(len(a_r)): # Right foot xyz
		acc_temp.append((a_r[i])[start_period:end_period])
	
	for i in range(len(a_diff)): # Difference between left and right foot
		acc_temp.append((a_diff[i])[start_period:end_period])
	
	# Resultant accelerations
	acc_temp.append(a_res_l[start_period:end_period]) # Left foot
	acc_temp.append(a_res_r[start_period:end_period]) # Right foot
	acc_temp.append(a_res_diff[start_period:end_period]) # Difference between left and right foot

	accelerations = np.vstack((accelerations, np.array(acc_temp).T))

	for i in range(len(new_F)):
		force_temp.append((new_F[i])[start_period:end_period])

	force = np.vstack((force, np.array(force_temp).T))
	
	HS_TO = np.array(HS_TO)

	return accelerations, HS_TO, force


def sort_events(FS: list, FO: list, first_event: str = 'FS', final_event: str = 'FO') -> (list, list):
	
	# We want the first event to be a 'first_event'
	if first_event == 'FS':
		if FO[0] < FS[0]:
			FO.pop(0)

	elif first_event == 'FO':
		if FS[0] < FO[0]:
			FS.pop(0)

	if first_event != final_event:
		# Make sure that the lengths of each list are equal

		assert_flag = False
		
		while not assert_flag:
			try:
				assert len(FS) == len(FO)
				assert_flag = True

			except AssertionError:
				if len(FS) > len(FO):
					FS.pop(-1)
				else:
					FO.pop(-1)
	
	else:

		if first_event == 'FS':
			# Create pointers to the lists
			first = FS
			second = FO
		
		elif first_event == 'FO':
			# Create pointers to the lists
			first = FO
			second = FS

		# Then the list of the first event string should have 1 more value than the opposite
		assert_flag = False

		while not assert_flag:
			try:
				assert(len(first) == len(second) + 1)
				assert_flag = True
			
			except AssertionError:
				# Remove the last value of second
				second.pop(-1)
	
	return FS, FO


def filter_acceleration(acc: np.ndarray) -> np.ndarray:
	from scipy import signal
	''' Filter acceleration data at 0.8 Hz and 45 Hz (band-pass) for general use '''
	analog_frequency = 1000
	cut_off_l = 0.8 # Derie (2017)
	cut_off_h = 45 # Derie (2017)
	order = 2 # Weyand (2017)
	b, a = signal.butter(N=order, Wn=[cut_off_l/(analog_frequency/2), cut_off_h/(analog_frequency/2)], btype='band')

	if acc.shape[-1] < len(a):
		acc = acc.T
	a_filt = signal.filtfilt(b, a, acc)

	return a_filt


def allocate_events(time: np.ndarray, a_left: np.ndarray, a_right: np.ndarray, FS: list, FO: list):
	import numpy as np
	# Make sure that the first and last event is a FS
	FS, FO = sort_events(FS, FO, first_event='FS', final_event='FS')

	FS_1 = FS[::2]
	FS_2 = FS[1::2]

	length = a_left.shape[-1]

	time_1 = np.split(time, FS_1)
	time_2 = np.split(time, FS_2)

	a_left_1 = np.vstack((np.split(a_left[0], FS_1), np.split(a_left[1], FS_1), np.split(a_left[2], FS_1)))
	a_left_2 = np.vstack((np.split(a_left[0], FS_2), np.split(a_left[1], FS_2), np.split(a_left[2], FS_2)))

	a_right_1 = np.vstack((np.split(a_right[0], FS_1), np.split(a_right[1], FS_1), np.split(a_right[2], FS_1)))
	a_right_2 = np.vstack((np.split(a_right[0], FS_2), np.split(a_right[1], FS_2), np.split(a_right[2], FS_2)))

	# Remove first and last strides as they will not be complete (assuming FS[0] != 0 and FS[-1] != length)
	if FS_1[0] != 0:
		time_1.pop(0)
		a_left_1 = np.vstack((np.delete(a_left_1[0], 0), np.delete(a_left_1[1], 0), np.delete(a_left_1[2], 0)))
		a_right_1 = np.vstack((np.delete(a_right_1[0], 0), np.delete(a_right_1[1], 0), np.delete(a_right_1[2], 0)))

	if FS_1[-1] != length:
		time_1.pop(-1)
		a_left_1 = np.vstack((np.delete(a_left_1[0], -1), np.delete(a_left_1[1], -1), np.delete(a_left_1[2], -1)))
		a_right_1 = np.vstack((np.delete(a_right_1[0], -1), np.delete(a_right_1[1], -1), np.delete(a_right_1[2], -1)))
	
	if FS_2[0] != 0:
		time_2.pop(0)
		a_left_2 = np.vstack((np.delete(a_left_2[0], 0), np.delete(a_left_2[1], 0), np.delete(a_left_2[2], 0)))
		a_right_2 = np.vstack((np.delete(a_right_2[0], 0), np.delete(a_right_2[1], 0), np.delete(a_right_2[2], 0)))

	if FS_2[-1] != length:
		time_2.pop(-1)
		a_left_2 = np.vstack((np.delete(a_left_2[0], -1), np.delete(a_left_2[1], -1), np.delete(a_left_2[2], -1)))
		a_right_2 = np.vstack((np.delete(a_right_2[0], -1), np.delete(a_right_2[1], -1), np.delete(a_right_2[2], -1)))


	# Look at first stride and see if maximum occurs in left or right foot in the y direction
	max_left_1 = max((a_left_1[:,0][1])[:int(len(a_left_1[:,0][1])/4)])
	max_left_2 = max((a_left_2[:,0][1])[:int(len(a_left_2[:,0][1])/4)])

	if max_left_1 > max_left_2:
		time_left = time_1
		time_right = time_2

		a_left = a_left_1
		a_right_for_left = a_right_1

		a_right = a_right_2
		a_left_for_right = a_left_2
	
	else:
		time_left = time_2
		time_right = time_1

		a_left = a_left_2
		a_right_for_left = a_right_2

		a_right = a_right_1
		a_left_for_right = a_left_1
	
	time_left = list(time_left)
	time_right = list(time_right)

	a_x_left = list(a_left[0])
	a_y_left = list(a_left[1])
	a_z_left = list(a_left[2])

	a_x_right_for_left = list(a_right_for_left[0])
	a_y_right_for_left = list(a_right_for_left[1])
	a_z_right_for_left = list(a_right_for_left[2])

	a_x_right = list(a_right[0])
	a_y_right = list(a_right[1])
	a_z_right = list(a_right[2])

	a_x_left_for_right = list(a_left_for_right[0])
	a_y_left_for_right = list(a_left_for_right[1])
	a_z_left_for_right = list(a_left_for_right[2])

	
	# Interpolate each stride to 100 points
	ninterpolates_points = 1000

	time_left_new, a_x_left_new, a_y_left_new, a_z_left_new, R_left = interpolate_acceleration(ninterpolates_points, time_left, a_x_left, a_y_left, a_z_left)
	time_left_new, a_x_right_for_left_new, a_y_right_for_left_new, a_z_right_for_left_new, R_right_for_left = interpolate_acceleration(ninterpolates_points, time_left, a_x_right_for_left, a_y_right_for_left, a_z_right_for_left)

	time_right_new, a_x_right_new, a_y_right_new, a_z_right_new, R_right = interpolate_acceleration(ninterpolates_points, time_right, a_x_right, a_y_right, a_z_right)
	time_left_new, a_x_left_for_right_new, a_y_left_for_right_new, a_z_left_for_right_new, R_left_for_right = interpolate_acceleration(ninterpolates_points, time_right, a_x_left_for_right, a_y_left_for_right, a_z_left_for_right)

	left = {}
	left['time'] = time_left_new
	left['a_x'] = a_x_left_new
	left['a_y'] = a_y_left_new
	left['a_z'] = a_z_left_new
	left['a_x_opposite_foot'] = a_x_right_for_left_new
	left['a_y_opposite_foot'] = a_y_right_for_left_new
	left['a_z_opposite_foot'] = a_z_right_for_left_new
	left['R'] = R_left
	left['R_opposite'] = R_right_for_left

	right = {}
	right['time'] = time_right_new
	right['a_x'] = a_x_right_new
	right['a_y'] = a_y_right_new
	right['a_z'] = a_z_right_new
	right['a_x_opposite_foot'] = a_x_left_for_right_new
	right['a_y_opposite_foot'] = a_y_left_for_right_new
	right['a_z_opposite_foot'] = a_z_left_for_right_new
	right['R'] = R_right
	right['R_opposite'] = R_left_for_right

	return left, right
	

def interpolate_acceleration(ninterpolates_points: int = 1000, time: np.ndarray = None, a_x: np.ndarray = None, a_y: np.ndarray = None, a_z: np.ndarray = None):
	
	from scipy import interpolate
	import numpy as np
	normal = np.linalg.norm

	for i in range(len(time)):

		t = time[i]
		
		# Create the new time array for interpolation
		new_t = np.linspace(t[0], t[-1], ninterpolates_points)

		# Interpolate using the cubic spline
		if a_x is not None:
			x = a_x[i]
			tck_x = interpolate.splrep(t, x, s=0)
			a_x_temp = interpolate.splev(new_t, tck_x, der=0)

			try:
				x_a = np.vstack((x_a, a_x_temp))
			except NameError:
				x_a = a_x_temp
		
		if a_y is not None:
			y = a_y[i]
			tck_y = interpolate.splrep(t, y, s=0)
			a_y_temp = interpolate.splev(new_t, tck_y, der=0)

			try:
				y_a = np.vstack((y_a, a_y_temp))
			except NameError:
				y_a = a_y_temp
		
		if a_z is not None:
			z = a_z[i]
			tck_z = interpolate.splrep(t, z, s=0)
			a_z_temp = interpolate.splev(new_t, tck_z, der=0)

			try:
				z_a = np.vstack((z_a, a_z_temp))
			except NameError:
				z_a = a_z_temp

		try:
			times = np.vstack((times, new_t))

		except NameError:
			times = new_t
		
		# Calculate resultant
		R = normal(np.vstack((a_x_temp, a_y_temp, a_z_temp)), axis=0)

		try:
			res = np.vstack((res, R))

		except NameError:
			res = R

	return times, x_a, y_a, z_a, res