import os
import glob
import numpy as np
import csv
from scipy import signal
import re
from matplotlib import pylab as plt
import openpyxl
from pathlib import Path
import h5py
import pickle

from tensorflow import keras
from sklearn.model_selection import train_test_split

'''
This script prepares acceleration data from ankle worn IMU's to find HS and TO events using a machine
learning process.

The IMU's should be placed on the medial aspect of the tibia (on each leg).

Left coordinate system: y = up, z = towards midline, x = forward direction
Right coordinate system: y = up, z = towards midline, x = backward direction

- assuming you are using a unit from I Measure U, the little man should be facing upwards and be
visible.

05/03/2020
Alex Woodall

'''

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


	return filter_plate


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


def prepare_data(data: np.ndarray, dataset: np.array, HS_TO: list, dataset_type: str, sample_length: int, f: str):
	'''
	This function creates the dataset of events in which features will be extracted from

	data: the data which will be split into samples of length sample_length
	dataset: the array which is being build of the samples from each trial
	HS_TO: a list of the truth values of the FS and FO events
	f: the name of the trial

	'''

	uid = re.findall(r'\d+', f)
	uid = int(uid[0] + uid[1])

	time = data[:,0].astype(np.float) # 1st column

	# Left foot
	a_l = (data[:,4:6+1].T).astype(np.float) # [ax, ay, az]

	# Right foot
	a_r = (data[:,7:9+1].T).astype(np.float) # [ax, ay, az]

	# Flip the x acceleration on the right foot. This will make the coordinate frames mirrored along the sagittal plane
	a_r[0] = -a_r[0]

	''' Filter acceleration data at 0.8 Hz and 45 Hz (band-pass) '''
	analog_frequency = 1000
	cut_off_l = 0.8 # Derie (2017), Robberechts et al (2019)
	cut_off_h = 45 # Derie (2017), Robberechts et al (2019)
	order = 2 # Weyand (2017), Robberechts et al (2019)
	b_a, a_a = signal.butter(N=order, Wn=[cut_off_l/(analog_frequency/2), cut_off_h/(analog_frequency/2)], btype='band')

	new_a_l = np.zeros(np.shape(a_l))
	new_a_r = np.zeros(np.shape(a_r))

	for i in range(len(a_l)):
		new_a_l[i,:] = signal.filtfilt(b_a, a_a, a_l[i,:])
	
	for i in range(len(a_r)):
		new_a_r[i,:] = signal.filtfilt(b_a, a_a, a_r[i,:])

	if dataset_type == 'training' or dataset_type == 'testing':
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

		''' Rezero '''
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
	
		# Check that the heel strike and toe off events are in the correct order (HS then TO)
		while heel_strike[0] > toe_off[0]:
			# Remove first toe_off event
			toe_off = toe_off[1:]
		
		while len(heel_strike) > len(toe_off):
			# Remove last heel strike event
			heel_strike = heel_strike[:-1]
		
		assert len(heel_strike) == len(toe_off)

		# Check that the order of events is HS, TO, HS, TO, ... etc
		TO_prev = 0
		for i in range(len(heel_strike)):
			# The current HS should occur after the previous TO
			assert heel_strike[i] > TO_prev
			
			# The current TO should occur after the last HS
			assert toe_off[i] > heel_strike[i]

			TO_prev = toe_off[i]

		# We now need to split each into individual samples of length sample_length ms (needs to be the
		# same size for ML). For the final timesteps that remain, add 0's to the end of HS_TO and 
		# continue with what the acceleration values are.

		t = -np.Inf

		start_period = 0
		end_period = start_period + sample_length

		HS = np.array(heel_strike)
		TO = np.array(toe_off)
		while t < time[-1]:
			if t > 15.8:
				stop = 1
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

			HS_TO.append(temp_HS_TO)

			# Append accelerations which are within the range
			temp = []

			temp.append([uid]*sample_length)
			temp.append(time[start_period:end_period])

			for i in range(len(new_a_l)):
				temp.append((new_a_l[i])[start_period:end_period])
			
			for i in range(len(new_a_r)):
				temp.append((new_a_r[i])[start_period:end_period])
			
			if np.size(dataset) == 0:
				dataset = (np.array(temp))[np.newaxis, :]
			else:
				dataset = np.vstack((dataset, (np.array(temp))[np.newaxis, :]))

			start_period = int(end_period - sample_length/2)
			
			end_period = int(start_period + sample_length)
			
			try:
				t = time[end_period-1] # Time at the end of the sample (for loop exiting)
			except IndexError:
				t = time[-1]

		start_period = int(len(new_a_l[0]) - sample_length)

		# Append heel strikes and toe offs which are within the range
		sample_HS = HS[HS >= start_period]

		# Convert into new timeframe
		sample_HS = sample_HS - start_period

		sample_TO = TO[TO >= start_period]
		
		# Convert into new timeframe
		sample_TO = sample_TO - start_period

		# HS_TO will have 0's where there is no event and a 1 where there is.
		# Two columns per trial. 1st = HS, 2nd = TO
		temp_HS_TO = np.zeros((sample_length, 2))
		(temp_HS_TO[:,0])[sample_HS] = 1.0
		(temp_HS_TO[:,1])[sample_TO] = 1.0

		HS_TO.append(temp_HS_TO)

		# Append accelerations which are within the range
		temp = []
		
		temp.append([uid]*sample_length)
		temp.append(time[start_period:end_period])

		for i in range(len(new_a_l)):
			temp.append((new_a_l[i])[start_period:])
		
		for i in range(len(new_a_r)):
			temp.append((new_a_r[i])[start_period:])

		if np.size(dataset) == 0:
			dataset = (np.array(temp))[np.newaxis, :]
		else:
			dataset = np.vstack((dataset, (np.array(temp))[np.newaxis, :]))

		''' Testing if HS_TO events are good '''

		dataset = np.swapaxes(dataset, 1, -1)

		HS_TO = np.array(HS_TO_init)
		y = HS_TO[:,:,0] * 1 + HS_TO[:,:,1] * 2
		y_test = keras.utils.to_categorical(y)

		max_time = int(max(dataset[-1,:,1]) * 1000) # in ms
		y_true = np.zeros((max_time+1, 4))
		y_true[:,0] = np.arange(0,max_time+1)

		time = (dataset[:,:,1])[:, np.newaxis]
		time = np.swapaxes(time, 1, -1)

		for i in range(len(time)):

			if i == 54:
				stop = 1

			y_true[(np.squeeze(time[i]) * 1000).astype(int),1] = np.sqrt(y_true[(np.squeeze(time[i]) * 1000).astype(int),1]**2 + y_test[i,:,0]**2)
			y_true[(np.squeeze(time[i]) * 1000).astype(int),2] = np.sqrt(y_true[(np.squeeze(time[i]) * 1000).astype(int),2]**2 + y_test[i,:,1]**2)
			y_true[(np.squeeze(time[i]) * 1000).astype(int),3] = np.sqrt(y_true[(np.squeeze(time[i]) * 1000).astype(int),3]**2 + y_test[i,:,2]**2)

		events = np.zeros((max_time,2))
		events[HS,0] = 1
		events[TO,1] = 1
		return dataset, HS_TO

if __name__ == '__main__':
	''' Read in file '''
	path = 'C:\\Users\\alexw\\Desktop\\RunningData\\workingdir\\data\\Raw Data'
	ext = 'csv'
	os.chdir(path)
	files = glob.glob('*.{}'.format(ext))

	dataset_type = ['training', 'testing', 'validating']
	dataset_init = np.array([])
	HS_TO_init = []

	
	xlsx_file = Path(path, 'Subject details.xlsx')
	wb_obj = openpyxl.load_workbook(xlsx_file)
	sheet = wb_obj.active

	subject_information = []

	for row in sheet.iter_rows():
		temp_info = []

		for cell in row:
			temp_info.append(cell.value)
		
		# List of subject information: ID, Gender, Height, Weight
		subject_information.append(temp_info)
	subject_information = np.array(subject_information)

	# Length of each sample = 600 ms
	length = 600

	# Dictionary to hold trial data and true solutions
	dataset = {}

	for f in files:
		print('Running file: ' + str(f))

		data = read_csv(f)

		f = f.split('.')[0]

		dataset[f] = {}

		dataset_init, HS_TO_init = prepare_data(data, dataset_init, HS_TO_init, dataset_type[0], length, f)
		
		# Shape of the entire dataset
		# (n, 636, 6)
		dataset_init = np.swapaxes(dataset_init, 1, -1)

		# Shape of the correct HS and TO events for each event
		# (n, 636, 3) 1st column = no event, 2nd = FS, 3rd = FO
		HS_TO = np.array(HS_TO_init)
		y = HS_TO[:,:,0] * 1 + HS_TO[:,:,1] * 2
		y_true = keras.utils.to_categorical(y)

		dataset[f]['X'] = dataset_init
		dataset[f]['y'] = y_true

		dataset_init = []
		HS_TO_init = []


	# dataset will have n number of events. Each event has the structure:
	# left:
	#	ax
	#	ay
	#	az
	# right:
	#	ax
	#	ay
	#	az

	dataset_folder = "C:\\Users\\alexw\\Dropbox\\ABI\\Level_8_Lab\\Vertical_GRF_from_IMU\\data\\"

	'''
	X_train, X_test, y_train, y_test = train_test_split(dataset, y_true, test_size=0.2)
	# Save datasets as h5
	hf = h5py.File(dataset_folder + 'dataset.h5', 'w')

	hf.create_dataset('X_all', data=dataset, compression="gzip")
	hf.create_dataset('y_all', data=y_true, compression="gzip")

	hf.create_dataset('X_train', data=X_train, compression="gzip")
	hf.create_dataset('y_train', data=y_train, compression="gzip")
	hf.create_dataset('X_test', data=X_test, compression="gzip")
	hf.create_dataset('y_test', data=y_test, compression="gzip")
	'''
	f = open(dataset_folder + "dataset.pkl", "wb")
	pickle.dump(dataset, f)
	f.close()
	np.save(dataset_folder + "subject_information.npy", subject_information)