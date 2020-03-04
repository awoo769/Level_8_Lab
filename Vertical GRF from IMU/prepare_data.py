import os
import glob
import numpy as np
import csv
from scipy import signal
import re
from matplotlib import pylab as plt
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

def rezero_filter(original_fy: np.ndarray, threshold: int = 20):
	'''
	Resets all values which were originally zero to zero

	Inputs:	original_fy: an array of unfiltered y data

	Outputs:	filter_plate: an array corresponding to a mask. '1' if above a threshold to keep,
				'0' if below a threshold and will be set to 0

	Original version in MATLAB written by Duncan Bakke

	'''

	filter_plate = np.zeros(np.shape(original_fy))

	# Binary test for values greater than 20
	force_zero = (original_fy > threshold) * 1 # Convert to 1 or 0 rather than True or False

	# We do not want to accept values which are over 20 but considered 'noise'.
	# Must be over 20 for more than 35 frames in a row. (Robberechts, 2019) Therefore, need to search for
	# [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] x 3.5 and get all the indices that meet this condition

	# Convert to string to test this condition
	force_str = ''.join(list(map(str, force_zero)))

	# Find all occurrences where the pattern occurs
	true_inds = [m.start() for m in re.finditer('(?=11111111111111111111111111111111111)', force_str)]

	# true_inds will not include the ends (e.g., 11...11100000) - will not include the final 3 1's
	extra_inds = [i + 35 for i in true_inds[0:-1]] # So make another array with 10 added on to all but the last value
	
	# Return the 'filtered' rezeroing array
	filter_plate[true_inds] = 1
	filter_plate[extra_inds] = 1

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


def prepare_data(data: np.ndarray, dataset_type: str):
	'''
	This function is the main function in preparing the acceleration data.

	'''

	time = data[:,0].astype(np.float) # 1st column

	# Left foot
	a_l = (data[:,4:6+1].T).astype(np.float) # [ax, ay, az]

	# Right foot
	a_r = (data[:,7:9+1].T).astype(np.float) # [ax, ay, az]

	# Flip the x acceleration on the right foot. This will make the acceleration profiles on the left and right foot
	# 'look' the same
	a_r[0] = -a_r[0] # CHECK IF I CAN DO THIS TODO

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
		filter_plate = rezero_filter(original_fy=F[1], threshold=threshold)
		
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

	# We now need to split each into individual steps
	# Extract a period ranging from 200 ms before the HS to 200 ms after TO
	a = 1

	

if __name__ == '__main__':
	''' Read in file '''
	path = 'C:\\Users\\alexw\\Desktop\\RunningData\\'
	ext = 'csv'
	os.chdir(path)
	files = glob.glob('*.{}'.format(ext))

	dataset_type = ['training', 'testing', 'validating']

	for f in files:
		data = read_csv(f)
		prepare_data(data, dataset_type[0])
