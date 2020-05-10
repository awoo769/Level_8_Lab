'''
This script prepares acceleration data from ankle worn IMU's to find HS and TO events using a machine
learning process.

The IMU's should be placed on the medial aspect of the tibia (on each leg).

Left coordinate system: y = up, z = towards midline, x = forward direction
Right coordinate system: y = up, z = towards midline, x = backward direction

- assuming you are using a unit from IMeasureU, the little man should be facing upwards and be
visible.

During this function, acceleration and force plate data will be interpolated to be at 1000 Hz

05/03/2020
Alex Woodall

'''

def prepare_data(data: np.ndarray, sample_length: int, f: str, overlap: bool = False) -> (np.ndarray, np.ndarray, np.ndarray):
	'''
	This function creates the dataset of events in which features will be extracted from

	data: the data which will be split into samples of length sample_length
	dataset: the array which is being build of the samples from each trial
	HS_TO: a list of the truth values of the FS and FO events
	f: the name of the trial
	overlap: whether to overlap each sample by half

	'''

	from scipy import signal

	# Time array is the first value in the data
	time = data[:,0].astype(np.float) # 1st column

	# Left foot
	a_l = (data[:,4:6+1].T).astype(np.float) # [ax, ay, az]

	# Right foot
	a_r = (data[:,7:9+1].T).astype(np.float) # [ax, ay, az]

	# Flip the x acceleration on the right foot. This will make the coordinate frames mirrored along the sagittal plane
	a_r[0] = -a_r[0]

	# Interpolate acceleration data to ensure that is it at 1000 Hz
	_, a_l = interpolate_data(time, a_l)
	_, a_r = interpolate_data(time, a_r)

	# Engineered timeseries
	a_diff = abs(a_l - a_r) # Difference between left and right
	a_res_l = np.linalg.norm(a_l, axis=0) # Left resultant
	a_res_r = np.linalg.norm(a_r, axis=0) # Right resultant
	a_res_diff = abs(a_res_l - a_res_r) # Difference between left and right resultant

	# Get force plate data for comparison
	F = (data[:,1:3+1].T).astype(np.float) #[Fx, Fy, Fz]; Fz = vertical
	time, F = interpolate_data(time, F)

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
	
	# Re-zero the filtered GRFs
	new_F = new_F * filter_plate
	
	''' Ground truth event timings '''
	# Get the points where there is force applied to the force plate (stance phase). Beginning = heel strike, end = toe off
	heel_strike = []
	toe_off = []

	for i in range(1, len(new_F[2])-1):
		# FS
		if (new_F[2])[i-1] == 0 and (new_F[2])[i] != 0:
			heel_strike.append(i-1)
		
		# FO
		if (new_F[2])[i+1] == 0 and (new_F[2])[i] != 0:
			toe_off.append(i+1)

	# We now need to split each into individual samples of length sample_length ms.

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

	# Run until t is equal to the final time iteration
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

		# Reset acceleration and force lists
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


def create_dataset(dataset_dict: dict, sample_length: int, f: str) -> dict:
	'''
	This function is calls the prepare_data function. It will gather the output
	of the prepare_data and use it to create the truth values to be used to predict
	events and timeseries.

	08/05/2020
	Alex Woodall

	'''
	# Import required functions
	from utils import read_csv

	# Localise functions for speed improvements
	zeros = np.zeros
	repeat = np.repeat
	where = np.where
	NaN = np.NaN
	isnan = np.isnan

	# Load the data
	data = read_csv(f)

	# Get the name of the trial and use it as the dictionary key
	f = f.split('.')[0]
	f = f.split('\\')[-1]
	dataset_dict[f] = {}

	# Sort the data into samples and pre-process data for analysis
	X, y, force = prepare_data(data=data, sample_length=sample_length, f=f, overlap=True)

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

	# Save to the dataset dictionary
	dataset_dict[f]['X'] = X
	dataset_dict[f]['X_starting_time'] = X_starting_time

	dataset_dict[f]['y_HS_binary'] = HS_binary
	dataset_dict[f]['y_TO_binary'] = TO_binary
	dataset_dict[f]['y_HS_time_to'] = HS_time_to
	dataset_dict[f]['y_TO_time_to'] = TO_time_to
	dataset_dict[f]['y_HS_time_to_next'] = HS_time_to_next
	dataset_dict[f]['y_TO_time_to_next'] = TO_time_to_next

	dataset_dict[f]['force'] = force

	return dataset_dict


def get_subject_info(path: str) -> np.array:
	'''
	This function opens the subject infomation excel workbook and saves it to a numpy array

	08/05/2020
	Alex Woodall

	'''

	import openpyxl
	from pathlib import Path

	# Localise functions for speed improvements
	array = np.array

	# Open excel file
	xlsx_file = Path(path, 'Subject details.xlsx')
	wb_obj = openpyxl.load_workbook(xlsx_file)
	sheet = wb_obj.active

	subject_information = []

	# Extract subject infomation
	for row in sheet.iter_rows():
		temp_info = []

		for cell in row:
			temp_info.append(cell.value)
		
		# List of subject information: ID, Gender, Height, Weight
		subject_information.append(temp_info)

	subject_information = array(subject_information)

	return subject_information

import glob
import numpy as np
import pickle

from utils import read_csv, rezero_filter, interpolate_data

if __name__ == '__main__':

	# Localise functions for speed improvements
	save = np.save
	dump = pickle.dump

	# Select path and read all .csv files (these will be the trial data)
	path = 'C:\\Users\\alexw\\Desktop\\tsFRESH\\Raw Data\\'
	ext = 'csv'
	files = glob.glob('{}*.{}'.format(path, ext))
	
	# Length of each sample = 100 ms
	length = 100

	# Dictionary to hold trial data and truth solutions
	dataset = {}

	for f in files:
		print('Running file: ' + str(f))

		dataset = create_dataset(dataset, length, f)

	# Save dataset
	dataset_folder = "C:\\Users\\alexw\\Desktop\\tsFRESH\\data\\"
	f = open(dataset_folder + "dataset.pkl", "wb")
	dump(dataset, f)
	f.close()

	# Get subject information and save
	subject_information = get_subject_info(path)
	save(dataset_folder + "subject_information.npy", subject_information)