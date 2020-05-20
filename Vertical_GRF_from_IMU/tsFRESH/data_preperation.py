'''
This script prepares acceleration data from ankle worn IMU's to find HS and TO events using a machine
learning process.

The IMU's should be placed on the medial aspect of the tibia (on each leg).

Left coordinate system: y = up, z = towards midline, x = forward direction
Right coordinate system: y = up, z = towards midline, x = backward direction

- assuming you are using a unit from IMeasureU, the little man should be facing upwards and
be on the medial side of each ankle

During this function, acceleration and force plate data will be interpolated to be at 1000 Hz

05/03/2020
Alex Woodall

'''

# For input type purposes
import numpy as np

def prepare_data(GRF_data: np.ndarray, IMU_data: np.ndarray, sample_length: int, overlap: bool = False) -> (np.ndarray, np.ndarray, np.ndarray):
	'''
	This function creates the dataset of events in which features will be extracted from

	data: the data which will be split into samples of length sample_length
	dataset: the array which is being build of the samples from each trial
	HS_TO: a list of the truth values of the FS and FO events
	overlap: whether to overlap each sample by half

	returns three numpy arrays: acceleration, force and event

	'''

	from scipy import signal
	import numpy as np

	from utils import interpolate_data, rezero_filter


	# Localise functions for speed improvements
	zeros = np.zeros
	normal = np.linalg.norm
	array = np.array
	Inf = np.Inf
	intersect1d = np.intersect1d
	vstack = np.vstack

	butter = signal.butter
	filtfilt = signal.filtfilt

	# Frequency to interpolate data to
	analog_frequency = 1000

	# Convert data to the correct shape
	if int(GRF_data.shape[1] > GRF_data.shape[0]) == 0:
		GRF_data = GRF_data.T
		IMU_data = IMU_data.T

	F = GRF_data[1:,:]
	GRF_time = GRF_data[0,:]

	# Rotate 180 deg around y axis (inverse Fx and Fz) - assuming that z is facing down
	F[0] = -F[0] # Fx
	F[2] = -F[2] # Fz

	''' Filter force plate data at 60 Hz '''
	cut_off = 60 # Derie (2017), Robberechts et al (2019)
	order = 2 # Weyand (2017), Robberechts et al (2019)
	b, a = butter(N=order, Wn=cut_off/(analog_frequency/2), btype='low')

	new_F = filtfilt(b, a, F, axis=1)

	''' Rezero filtered forces '''
	threshold = 50 # 60 N
	filter_plate = rezero_filter(original_fz=new_F[2], threshold=threshold)

	# Re-zero the filtered GRFs
	new_F = new_F * filter_plate

	# Interpolate GRF
	time, new_F = interpolate_data(GRF_time, new_F, analog_frequency)

	# Re-zero after interpolating
	filter_plate = rezero_filter(original_fz=new_F[2], threshold=threshold)
	new_F = new_F * filter_plate

	''' Ground truth event timings '''
	# Get the points where there is force applied to the force plate (stance phase). Beginning = foot strike, end = foot off
	foot_strike = []
	foot_off = []

	# Use the vertical GRF, but any GRF would produce the same result
	for i in range(1, len(new_F[2])-1):
		# FS
		if (new_F[2])[i-1] == 0 and (new_F[2])[i] != 0:
			foot_strike.append(i-1)
		
		# FO
		if (new_F[2])[i+1] == 0 and (new_F[2])[i] != 0:
			foot_off.append(i+1)

	# Filter and interpolate acceleration data
	''' Filter acceleration data at 0.8 Hz and 45 Hz (band-pass) '''
	cut_off_l = 0.8 # Derie (2017), Robberechts et al (2019)
	cut_off_h = 45 # Derie (2017), Robberechts et al (2019)
	order = 2 # Weyand (2017), Robberechts et al (2019)
	b, a = signal.butter(N=order, Wn=[cut_off_l/(analog_frequency/2), cut_off_h/(analog_frequency/2)], btype='band')

	IMU = IMU_data[1:,:]
	IMU_time = IMU_data[0,:]

	new_IMU = filtfilt(b, a, IMU, axis=1)
	time, new_IMU = interpolate_data(IMU_time, new_IMU, analog_frequency)

	a_r = new_IMU[:3]
	a_l = new_IMU[3:] 

	# Engineered timeseries
	a_diff = abs(a_l - a_r) # Difference between left and right
	a_res_l = normal(a_l, axis=0) # Left resultant
	a_res_r = normal(a_r, axis=0) # Right resultant
	a_res_diff = abs(a_res_l - a_res_r) # Difference between left and right resultant


	# We now need to split each into individual samples of length sample_length.
	# Because the data is sampled at 1000 Hz, 1 indice = 1 ms.

	# Initial time
	t = -Inf

	# Initial start and end periodss
	start_period = 0
	end_period = start_period + sample_length

	# Convert heel strike and toe off lists to arrays
	foot_strike = array(foot_strike)
	foot_off = array(foot_off)

	# Unique id
	uid = 0
	
	# Temporary data lists
	acc_temp = []
	force_temp = []

	# Run until t is equal to the final time iteration
	while t < time[-1]:

		# Append heel strikes and toe offs which are within the range
		sample_FS = intersect1d(foot_strike[foot_strike < end_period], foot_strike[foot_strike >= start_period])

		# Convert into new timeframe
		sample_FS = sample_FS - start_period

		sample_FO = intersect1d(foot_off[foot_off < end_period], foot_off[foot_off >= start_period])
		
		# Convert into new timeframe
		sample_FO = sample_FO - start_period

		# HS_TO will have 0's where there is no event and a 1 where there is.
		# Two columns per trial. 1st = HS, 2nd = TO
		temp_event = zeros((sample_length, 2))
		(temp_event[:,0])[sample_FS] = 1.0
		(temp_event[:,1])[sample_FO] = 1.0

		try:
			event = vstack((event, temp_event))
		except NameError: # HS_TO does not exist
			event = temp_event

		# Append accelerations which are within the range
		acc_temp.append([uid]*sample_length) # Unique id
		acc_temp.append(time[start_period:end_period]) # time

	 	# Left foot xyz
		for i in range(len(a_l)):
			acc_temp.append((a_l[i])[start_period:end_period])
		
		# Right foot xyz
		for i in range(len(a_r)):
			acc_temp.append((a_r[i])[start_period:end_period])
		
		# Difference between left and right foot
		for i in range(len(a_diff)):
			acc_temp.append((a_diff[i])[start_period:end_period])
		
		# Resultant accelerations
		acc_temp.append(a_res_l[start_period:end_period]) # Left foot
		acc_temp.append(a_res_r[start_period:end_period]) # Right foot
		acc_temp.append(a_res_diff[start_period:end_period]) # Difference between left and right foot

		for i in range(len(new_F)): # Filtered force plate data
			force_temp.append((new_F[i])[start_period:end_period])
		
		try:
			force = vstack((force, np.array(force_temp).T))
		except NameError: # force does not exist
			force = array(force_temp).T

		try:
			accelerations = vstack((accelerations, np.array(acc_temp).T))
		except NameError: # accelerations does not exist
			accelerations = array(acc_temp).T

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

		# Reset acceleration and force lists
		acc_temp = []
		force_temp = []
	
	# For the part of the timeseries which is left over
	final_sample_length = len(a_diff[0]) - start_period

	# Append foot strikes and foot offs which are within the range
	sample_FS = foot_strike[foot_strike >= start_period]
	sample_FS = sample_FS - start_period

	sample_FO = foot_off[foot_off >= start_period]
	sample_FO = sample_FO - start_period

	# HS_TO will have 0's where there is no event and a 1 where there is.
	# Two columns per trial. 1st = HS, 2nd = TO
	temp_event = zeros((final_sample_length, 2))
	(temp_event[:,0])[sample_FS] = 1.0
	(temp_event[:,1])[sample_FO] = 1.0

	event = vstack((event, temp_event))

	# Append accelerations which are within the range
	acc_temp.append([uid]*final_sample_length)
	acc_temp.append(time[start_period:])

	# Left foot xyz
	for i in range(len(a_l)):
		acc_temp.append((a_l[i])[start_period:end_period])
	
	# Right foot xyz
	for i in range(len(a_r)):
		acc_temp.append((a_r[i])[start_period:end_period])
	
	# Difference between left and right foot
	for i in range(len(a_diff)):
		acc_temp.append((a_diff[i])[start_period:end_period])
	
	# Resultant accelerations
	acc_temp.append(a_res_l[start_period:end_period]) # Left foot
	acc_temp.append(a_res_r[start_period:end_period]) # Right foot
	acc_temp.append(a_res_diff[start_period:end_period]) # Difference between left and right foot

	accelerations = vstack((accelerations, np.array(acc_temp).T))

	for i in range(len(new_F)):
		force_temp.append((new_F[i])[start_period:end_period])

	force = vstack((force, np.array(force_temp).T))

	return accelerations, event, force


def create_dataset(dataset_dict: dict, sample_length: int, f: str, overlap: bool = False) -> dict:
	'''
	This function is calls the prepare_data function. It will gather the output
	of the prepare_data and use it to create the truth values to be used to predict
	events and timeseries.

	08/05/2020
	Alex Woodall

	'''

	import numpy as np

	# Import required functions
	from utils import read_csv, get_runner_info

	# Localise functions for speed improvements
	zeros = np.zeros
	repeat = np.repeat
	where = np.where
	NaN = np.NaN
	isnan = np.isnan

	# Load the data
	GRF_data, IMU_data = read_csv(f)

	# Sort the data into samples and pre-process data for analysis
	X, y, force = prepare_data(GRF_data=GRF_data, IMU_data=IMU_data, sample_length=sample_length, overlap=overlap)

	# Get number of samples
	uids = list(set(X[:,0]))
	
	# Create binary output arrays
	HS_binary = zeros(len(uids))
	TO_binary = zeros(len(uids))

	# Create time to event output arrays
	HS_time_to = repeat(-1, len(uids))
	TO_time_to = repeat(-1, len(uids))

	# Create starting time output (of each sample)
	X_starting_time = zeros(len(HS_time_to))

	# For each sample
	for uid in uids:
		# Get the indices of each point in the sample
		uid_ind = where(X[:,0] == uid)[0]

		# The starting time (in ms)
		X_starting_time[int(uid)] = X[uid_ind[0],1] * 1000
		
		# Binary did an event happen in each sample
		if 1 in y[uid_ind[0]:uid_ind[-1]+1,0]: # If there is a HS event	
			HS_binary[int(uid)] = 1.0
		
		if 1 in y[uid_ind[0]:uid_ind[-1]+1,1]: # If there is a TO event
			TO_binary[int(uid)] = 1.0
		
		# Time to this event in each sample, will be -1 if no event
		if 1 in y[uid_ind[0]:uid_ind[-1]+1,0]: # If there is a HS event
			HS_time_to[int(uid)] = where(y[uid_ind[0]:uid_ind[-1]+1,0] == 1)[0] + X[uid_ind[0],1] * 1000

		if 1 in y[uid_ind[0]:uid_ind[-1]+1,1]: # If there is a TO event
			TO_time_to[int(uid)] = where(y[uid_ind[0]:uid_ind[-1]+1,1] == 1)[0] + X[uid_ind[0],1] * 1000

	# Time to next event (FS and FO) from beginning of sample
	HS_time_to_next = [0] * len(HS_time_to)
	HS_time_to = list(HS_time_to)

	TO_time_to_next = [0] * len(TO_time_to)
	TO_time_to = list(TO_time_to)

	for j in range(2):
		event_time = NaN

		if j == 0:
			time_to = HS_time_to
			time_to_next = HS_time_to_next
		
		else:
			time_to = TO_time_to
			time_to_next = TO_time_to_next

		for i in range(len(time_to)):
			# Go backwards through array
			if time_to[-1 - i] != -1:
				# Update event time
				event_time = time_to[-1 - i]

			if isnan(event_time): # If an event hasn't occured yet
				time_to_next[-1 - i] = NaN
			
			else:
				# The time to next event
				time_to_next[-1 - i] = event_time - X_starting_time[-1 - i]
	
	HS_time_to_next = np.array(HS_time_to_next)
	TO_time_to_next = np.array(TO_time_to_next)

	# Get the name of the trial and use it as the dictionary key
	ID = f.split('.')[0]
	ID = ID.split('\\')[-1]

	# Get mass of trial
	runner_ID = ID.split('ITL')[0] + 'a'
	info = get_runner_info('C:\\Users\\alexw\\Dropbox\\auckIMU\\demos.xlsx')

	iloc = np.where(info['Subject_ID'] == runner_ID)[0][0]
	mass = float(info['Mass'].iloc[iloc])

	mass_all = [mass] * len(X)
	mass_sample = [mass] * len(X_starting_time)

	if not isnan(mass):
		dataset_dict[ID] = {}

		# Save to the dataset dictionary
		dataset_dict[ID]['X'] = X
		dataset_dict[ID]['X_starting_time'] = X_starting_time
		dataset_dict[ID]['X_mass_all'] = mass_all
		dataset_dict[ID]['X_mass_sample'] = mass_sample

		dataset_dict[ID]['y_FS_binary'] = HS_binary
		dataset_dict[ID]['y_FO_binary'] = TO_binary
		dataset_dict[ID]['y_FS_time_to'] = HS_time_to
		dataset_dict[ID]['y_FO_time_to'] = TO_time_to
		dataset_dict[ID]['y_FS_time_to_next'] = HS_time_to_next
		dataset_dict[ID]['y_FO_time_to_next'] = TO_time_to_next

		dataset_dict[ID]['force'] = force

	return dataset_dict


def main():

	from glob import glob
	import numpy as np
	import pickle
	import os

	from utils import get_runner_info, sort_strike_pattern

	# Localise functions for speed improvements
	dump = pickle.dump

	# Select path and read all .csv files (these will be the trial data)
	file_path = 'C:\\Users\\alexw\\Dropbox\\auckIMU\\SNRCdat_default\\'
	info_path = 'C:\\Users\\alexw\\Dropbox\\auckIMU\\demos.xlsx'
	ext = '*.csv'

	all_csv_files = [file
					for path, subdir, files in os.walk(file_path)
					for file in glob(os.path.join(path, ext))]
	
	# If there is an 02, 03... etc, then don't use the 01 (or versions previous) (something may have gone wrong)
	prev = ''
	remove_index = []
	for i in range(len(all_csv_files)):
		cur = all_csv_files[i]

		if cur[:-6] == prev[:-6]:
			# Remove previous
			remove_index.append(i-1)
		prev = cur

	i = 0
	for element in remove_index:
		all_csv_files.pop(element - i)

		i += 1
		
	runner_info = get_runner_info(info_path)
	RFS, MFS, FFS, Mixed = sort_strike_pattern(runner_info)

	# Length of each sample = 100 ms
	length = 100

	# Dictionary to hold trial data and truth solutions
	dataset = {}

	overlap = True

	for f in all_csv_files:
		# Get runners ID
		ID = f.split('\\')[-1]
		ID = ID.split('ITL')[0] + 'a'

		if ID in RFS:
			print('Running file: ' + str(f))
			dataset = create_dataset(dataset, length, f, overlap)

	# Save dataset
	dataset_folder = "C:\\Users\\alexw\\Desktop\\Harvard_data\\"

	if overlap:
		f = open(dataset_folder + "dataset_overlap.pkl", "wb")
	else:
		f = open(dataset_folder + "dataset_no_overlap.pkl", "wb")
	dump(dataset, f)
	f.close()


if __name__ == '__main__':
	main()
	