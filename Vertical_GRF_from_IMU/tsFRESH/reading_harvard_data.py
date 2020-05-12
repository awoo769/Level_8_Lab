'''
This script opens the dataset provided by Harvard Uni and sorts into force and acceleration data.

Within the acceleration data, there is a FORE (front) and AFT (back) force plate. Use the 
combined data which accounts for both.

Alex Woodall
12/05/2020

'''

def read_csv(filename: str):

	'''
	This function opens and reads a csv file, returning a numpy array (data) of the contents.

	'''

	import numpy as np
	array = np.array
	linspace = np.linspace

	with open(filename, 'r') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')

		GRF_time = []
		IMU_time = []

		GRF_data = []
		IMU_data = []
		i = 0
		
		freq = False
		meta = False
		create_time = False

		# Start with GRF data
		data = GRF_data
		time = GRF_time
		for row in reader:

			if len(row) == 0:
				if create_time == True:
					start_time = 0
					end_time = len(data) / frequency

					ninterpolates_points = len(data)

					# Create the new time array for interpolation
					time.append(list(linspace(start_time, end_time, ninterpolates_points, False)))


				data = IMU_data
				time = IMU_time

			elif meta:
				# Three rows of meta data
				i += 1

				if i == 3:
					meta = False
					create_time = True
					i = 0

			elif freq:
				frequency = int(row[0])

				freq = False
				meta = True

			elif len(row) != 0 and 'Devices' in row[0]:
				freq = True

			elif len(row) != 0 and freq == False and meta == False:
				if len(row) > 1:
					data.append(row)

	# Convert to numpy arrays
	GRF_data = array(GRF_data)
	GRF_time = array(GRF_time).T

	IMU_data = array(IMU_data)
	IMU_time = array(IMU_time).T

	# Only use the combined force arrays
	GRF_data = np.hstack((GRF_time, GRF_data[:,20:23].astype(float)))
	IMU_data = np.hstack((IMU_time, IMU_data[:,2:].astype(float)))

	return GRF_data, IMU_data

import csv
import pandas
from matplotlib import pyplot as plt
from scipy import signal

from utils import interpolate_data, rezero_filter

butter = signal.butter
filtfilt = signal.filtfilt

directory = 'C:\\Users\\alexw\\Desktop\\SNRCdat\\NDR0228\\NDR0228ITLaSaRl01.csv'

GRF_data, IMU_data = read_csv(directory)
analog_frequency = 1000

if int(GRF_data.shape[1] > GRF_data.shape[0]) == 0:
	GRF_data = GRF_data.T
	IMU_data = IMU_data.T

# Filter before interpolation
F = GRF_data[1:,:]
GRF_time = GRF_data[0,:]

# Rotate 180 deg around y axis (inverse Fx and Fz) - assuming that z is facing down
F[0] = -F[0] # Fx
F[2] = -F[2] # Fz

''' Filter force plate data at 60 Hz '''
cut_off = 60 # Derie (2017), Robberechts et al (2019)
order = 2 # Weyand (2017), Robberechts et al (2019)
b_f, a_f = butter(N=order, Wn=cut_off/(analog_frequency/2), btype='low')

new_F = filtfilt(b_f, a_f, F, axis=1)

''' Rezero filtered forces'''
threshold = 20 # 20 N
filter_plate = rezero_filter(original_fz=new_F[2], threshold=threshold)

# Re-zero the filtered GRFs
new_F = new_F * filter_plate

time, new_F = interpolate_data(GRF_time, new_F, analog_frequency)


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

ok=1
