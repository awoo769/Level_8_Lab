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

from write_mot import write_mot

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


def prepare_data(data: np.ndarray, acc_dataset: list, HS_TO: list, force_dataset: list, filepath: str):
	'''
	This function creates the dataset of events in which features will be extracted from

	'''

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

	# Write IMU motion file (input file)
	headers = ['time', 'L_accel.x', 'L_accel.y', 'L_accel.z', 'R_accel.x', 'R_accel.y', 'R_accel.z']

	data = np.vstack((time, new_a_l, new_a_r))

	new_path = filepath.rsplit("\\",1)[-1]
	new_path = new_path.rsplit(".",1)[0]

	name = filepath.rsplit("\\",1)[0] + '\\ProcessedIMU' + new_path + ".mot"

	write_mot(grf_complete=data, file_path=name, headers=headers, unit="mm/s^2")
	
	# Write Force motion file (output file)
	headers = ['time', 'Force.Fz1']
	data = np.vstack((time, new_F[-1]))

	name = filepath.rsplit("\\",1)[0] + '\\ProcessedvGRF' + new_path + ".mot"
	write_mot(grf_complete=data, file_path=name, headers=headers, unit="F")

if __name__ == '__main__':
	''' Read in file '''
	path = 'C:\\Users\\alexw\\Desktop\\RunningData\\'
	ext = 'csv'
	os.chdir(path)
	files = glob.glob('*.{}'.format(ext))

	acc_dataset = []
	force_dataset = []
	HS_TO = []

	
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

	for f in files:
		print('Running file: ' + str(f))
		data = read_csv(f)
		prepare_data(data, acc_dataset, HS_TO, force_dataset, filepath = path + f)
	