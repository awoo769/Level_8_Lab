import numpy as np
from matplotlib import pyplot as plt
import os
from scipy import signal
from scipy.signal import argrelextrema
import csv
import glob
import openpyxl
from pathlib import Path
import pickle

'''
The purpose of this script is to estimate GRF's from IMU ankle acceleration data from running trials.

Alex Woodall
Auckland Bioengineering Institution

'''

from utils import prepare_data, read_csv, sort_events, filter_acceleration


def estimate_vGRF(time: np.array, FO: list, FS: list, ay: np.array, m: float):

	# Need to split FS and FO into left and right
	



	# Contact time = toe off - heel strike
	contact_time = time[FO[:-1]] - time[FS[:-1]]

	# Aerial time = heel strike[i+1] - toe off[i]
	aerial_time = time[FS[1:]] - time[FO[:-1]]

	# Getting FT,avg
	g = 9.8 # Gravity
	FT_avg = m * g * (contact_time + aerial_time / contact_time) # Scalar

	# Impulse, J1 corresponds to the vertical deceleration of m1 during surface impact
	m1 = m * 0.08 # m1 is 8.0 % of the body's mass (Plagenhoef et al., 1983; Winter, 1990)

	# Time interval between touchdown and vertical velocity of m1 slowing to 0 = time interval between
	# touchdown and peak vertical acceleration
	maximum = argrelextrema(ay, np.greater)[0]
	max_ay = []

	for i in range(len(HS)):
		if i == len(HS) - 1: # Last HS
			maximum = max(ay[FS[i]:])
			max_ay.append(np.where(ay == maximum)[0][0])
		else:
			maximum = max(ay[FS[i]:FS[i+1]])
			max_ay.append(np.where(ay == maximum)[0][0])

	# We have to loose the final step because aerial time requires the next step's HS to be present

	# Each of these values will be a scalar (for each stide)

	# The time interval between touchdown and the vertical velocity of m1 slowing to 0
	assert max_ay[0] > FS[0]

	t1 = time[max_ay[:-1]] - time[FS[:-1]] 

	# The change in acceleration over this period
	a1 = ay[max_ay[:-1]] - ay[FS[:-1]]

	# The impulse for m1 over this period
	J1 = (m1 * a1 + m1 * g) * (2 * t1)

	# The total impulse over this period
	JT = FT_avg * contact_time

	# The impulse for m2 over this period
	J2 = JT - J1

	# The average force for m1 over this period
	F1_avg = J1 / (2 * t1)

	# The average force for m2 over this period
	F2_avg = J2 / contact_time

	''' Force shaped curve '''
	# A bell-shaped force curve F(t) for each impulse (J1, J2) can be accurately modelled using the RCB curve
	# (Clark et., al). The raised cosine function can be derived from the first two terms of the Fourier series

	# Term 1: F(t) = alpha_0 + alpha_1 * sin(2*pi*f_1*t + theta_1)
	# Term 2: F(t) = alpha_0 + alpha_1 * cos(2*pi*f_1*t)

	# The RCB function is defined over a finite time interval of one period:
	# F(t) = 0 for t < B - C
	# F(t) = A/2 [1 + cos((t - B) / C * pi)] for B - C < t < B + C
	# F(t) = 0 for t > B + C

	# A is the peak amplitude, B is the center time of the peak and C is the half-width time interval

	# For J1

	A1 = 2 * F1_avg
	B1 = t1
	C1 = t1

	# F1_l will contain the time series F1 values
	F1 = np.zeros(len(time))
	starting_points = FS[:-1]

	starting_point_previous = 0
	B_previous = 0
	C_previous = 0

	j = 0

	for i in range(len(time)):

		# Change j when time[i] becomes greater than starting_point_l[j + 1]
		if j == len(starting_points) - 1:
			j = j
		elif time[i] > time[starting_points[j+1]]:
			j += 1 # Increase j

			starting_point_previous = starting_point
			B_previous = B
			C_previous = C

		# F(t) = 0 for t < B - C
		# F(t) = A/2 [1 + cos((t - B) / C * pi)] for B - C < t < B + C
		# F(t) = 0 for t > B + C
		A = A1[j]
		B = B1[j]
		C = C1[j]
		t = time[i]
		starting_point = time[starting_points[j]] - B + C

		if t < (starting_point + B - C) and t > (starting_point_previous + B_previous + C_previous):
			F1[i] = 0
		elif t > (starting_point + B + C):
			F1[i] = 0
		elif t >= (starting_point + B - C) and t <= (starting_point + B + C):
			F1[i] = A / 2 * (1 + np.cos((t - B) / C * np.pi))

	# For J2, assuming symmetry
	#A2 = 2 * F2_avg

	#B2_l = 0.5 * contact_time_left
	#B2_r = 0.5 * contact_time_right

	#C2_l = 0.5 * contact_time_left
	#C2_r = 0.5 * contact_time_right

	# For J2, assuming asymmetry
	# F(t) = 0 for t < B2 - C2L
	# F(t) = A2/2 [1 + cos((t - B2) / C2L * pi)] for B2 - C2L < t < B2
	# F(t) = A2/2 [1 + cos((t - B2) / C2T * pi)] for B2 < t < B2 + C2T
	# F(t) = 0 for t > B2 + C2T

	# A2 is the peak amplitude, B2 is the center time of the peak, C2L is the leading half-width time interval, 
	# and C2T is the trailing half-width time interval. B2 is set to 0.47 tc as per the center of mass asymmetry
	# originally reported by Cavagna et al. (1997). With the symmetry control, C2L = B2 and C2T = tc - B2
	A2 = 2 * F2_avg
	
	B2 = 0.47 * contact_time

	C2L = B2

	C2T = contact_time - B2

	j = 0

	# F2_l will contain the time series F2 values
	F2 = np.zeros(len(time))
	starting_points = FS[:-1]

	starting_point_previous = 0

	for i in range(len(time)):
		
		# Change j when time[i] becomes greater than starting_point_l[j + 1]
		if j == len(starting_points) - 1:
			j = j
		elif time[i] > time[starting_points[j+1]]:
			j += 1 # Increase j
			starting_point_previous = starting_point
		
		# Assuming asymmetry in J2
		# F(t) = 0 for t < B2 - C2L
		# F(t) = A2/2 [1 + cos((t - B2) / C2L * pi)] for B2 - C2L < t < B2
		# F(t) = A2/2 [1 + cos((t - B2) / C2T * pi)] for B2 < t < B2 + C2T
		# F(t) = 0 for t > B2 + C2T

		C2T = contact_time[j] - B2
		A2 = A2[j]
		B2 = B1[j]
		C2L = C2L[j]
		C2T = C2T[j]
		t = time[i]
		starting_point = time[starting_points[j]]

		if t < (starting_point + B2 - C2L) and t > (starting_point_previous + B2 + C2T):
			F2[i] = 0
		elif t > (starting_point + B2 + C2T):
			F2[i] = 0
		elif t >= (starting_point + B2 - C2L) and t <= (starting_point + B2):
			F2[i] = A2 / 2 * (1 + np.cos((t - B2) / C2L * np.pi))
		elif t > (starting_point + B2) and t <= (starting_point + B2 + C2T):
			F2[i] = A2 / 2 * (1 + np.cos((t - B2) / C2T * np.pi))

	FT = F1 + F2

	return FT

''' Read in file '''
path = 'C:\\Users\\alexw\\Desktop\\tsFRESH\\Raw Data'
ext = 'csv'
os.chdir(path)
files = glob.glob('*.{}'.format(ext))

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
length = 100

counter = 0

for f in files:
	counter += 1

	print('Running file: ' + str(f))

	data = read_csv(f)

	f = f.split('.')[0]

	X, y, force = prepare_data(data, length, f, False)

	# Get times/indices of FS and FO events
	FS = list(np.where(y[:,0] == 1)[0])
	FO = list(np.where(y[:,1] == 1)[0])
	
	FS_new, FO_new = sort_events(FS, FO)

	# Time array
	time = X[:,1]

	# Unfiltered acceleration array
	ay = np.vstack((X[:,3], X[:,6]))
	ay_filt = filter_acceleration(ay)

	m = float(subject_information[counter, -1])

	estimate_vGRF(time, FO, FS, ay, m)
