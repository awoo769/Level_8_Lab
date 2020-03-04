import numpy as np
from matplotlib import pyplot as plt
import os
from scipy import signal
from scipy.signal import argrelextrema
import csv
import glob

'''
The purpose of this script is to estimate GRF's from IMU ankle acceleration data from running trials.

Alex Woodall
Auckland Bioengineering Institution

'''

def estimate_vGRF(time: np.array, TO: list, HS: list, ay: np.array, m: float):

	# Contact time = toe off - heel strike
	contact_time = time[TO[:-1]] - time[HS_l[:-1]]

	# Aerial time = heel strike[i+1] - toe off[i]
	aerial_time = time[HS[1:]] - time[TO[:-1]]

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
			maximum = max(ay[HS[i]:])
			max_ay.append(np.where(ay == maximum)[0][0])
		else:
			maximum = max(ay[HS[i]:HS[i+1]])
			max_ay.append(np.where(ay == maximum)[0][0])

	# We have to loose the final step because aerial time requires the next step's HS to be present

	# Each of these values will be a scalar (for each stide)

	# The time interval between touchdown and the vertical velocity of m1 slowing to 0
	assert max_ay[0] > HS[0]

	t1 = time[max_ay[:-1]] - time[HS[:-1]] 

	# The change in acceleration over this period
	a1 = ay[max_ay[:-1]] - ay[HS[:-1]]

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
	starting_points = HS[:-1]

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
	starting_points = HS[:-1]

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

def get_heel_strike_event(time: np.array, az: np.array, foot: str):
		
	'''
	This function estimates the heel strike event for the ankle using IMU data. The IMU should be placed on the medial
	aspect of the tibia.

	The z acceleration data should be filtered at 2 Hz (low pass butter worth filter, n = 4)

	'''

	if 'l' in foot or 'r' in foot:
		# All minimas
		HS = argrelextrema(az, np.less)[0]

	else:
		print('Please enter appropriate label for foot type')

	return HS.tolist()

def get_toe_off_event(time: np.array, ay: np.array, foot: str):
	
	'''
	This function estimates the toe off event for the ankle using IMU data. The IMU should be placed on the medial
	aspect of the tibia.

	The y acceleration data should be filtered at 2 Hz (low pass butter worth filter, n = 4)

	'''

	if 'l' in foot or 'r' in foot:
		# Find all maximas acceleration in the y direction

		# All maximas
		TO = argrelextrema(ay, np.greater)[0]
		
	else:
		print('Please enter appropriate label for foot type')

	return TO.tolist()

''' Read in file '''
path = 'C:\\Users\\alexw\\Desktop\\RunningData\\'
ext = 'csv'
os.chdir(path)
files = glob.glob('*.{}'.format(ext))

contact_time_l_diff_all = []
contact_time_r_diff_all = []

left_differences_per_HS_all = []
right_differences_per_HS_all = []
left_differences_per_TO_all = []
right_differences_per_TO_all = []

left_differences_HS_all = []
right_differences_HS_all = []
left_differences_TO_all = []
right_differences_TO_all = []

for data_directory in files:
	with open(data_directory, 'r') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')

		read_file = []
		i = 0
		
		for row in reader:
			i += 1

			# First data row on line 8
			if i >= 8:
				if len(row) != 0:
					read_file.append(row)

	# Participant running with both legs on force plate 1.

	time = (np.array(read_file)[:,0]).astype(np.float) # 1st column

	# Left foot
	a_x_ankle_l = (np.array(read_file)[:,4]).astype(np.float) # 5th column
	a_y_ankle_l = (np.array(read_file)[:,5]).astype(np.float) # 6th column
	a_z_ankle_l = (np.array(read_file)[:,6]).astype(np.float) # 7th column

	# Right foot
	a_x_ankle_r = (np.array(read_file)[:,7]).astype(np.float) # 8th column
	a_y_ankle_r = (np.array(read_file)[:,8]).astype(np.float) # 9th column
	a_z_ankle_r = (np.array(read_file)[:,9]).astype(np.float) # 10th column

	# Also take force plate data for comparison
	grf_x = (np.array(read_file)[:,1]).astype(np.float) # 2nd column
	grf_y = (np.array(read_file)[:,2]).astype(np.float) # 3rd column
	grf_z = (np.array(read_file)[:,3]).astype(np.float) # 4th column

	''' Filter force plate data at 60 Hz for general use '''
	analog_frequency = 1000
	cut_off = 60 # Derie (2017)
	order = 2 # Weyand (2017)
	b_f, a_f = signal.butter(N=order, Wn=cut_off/(analog_frequency/2), btype='low')

	''' Filter acceleration data at 0.8 Hz and 45 Hz (band-pass) for general use '''
	analog_frequency = 1000
	cut_off_l = 0.8 # Derie (2017)
	cut_off_h = 45 # Derie (2017)
	order = 2 # Weyand (2017)
	b_a, a_a = signal.butter(N=order, Wn=[cut_off_l/(analog_frequency/2), cut_off_h/(analog_frequency/2)], btype='band')

	ax_filt_l = signal.filtfilt(b_a, a_a, a_x_ankle_l)
	ay_filt_l = signal.filtfilt(b_a, a_a, a_y_ankle_l)
	az_filt_l = signal.filtfilt(b_a, a_a, a_z_ankle_l)
	R_ankle_l = np.sqrt(np.power(ax_filt_l, 2) + np.power(ay_filt_l, 2) + np.power(az_filt_l, 2))

	ax_filt_r = signal.filtfilt(b_a, a_a, a_x_ankle_r)
	ay_filt_r = signal.filtfilt(b_a, a_a, a_y_ankle_r)
	az_filt_r = signal.filtfilt(b_a, a_a, a_z_ankle_r)
	R_ankle_r = np.sqrt(np.power(ax_filt_r, 2) + np.power(ay_filt_r, 2) + np.power(az_filt_r, 2))

	''' Filter data at 2 Hz to remove any high frequency components for event detection '''
	analog_frequency = 1000
	cut_off = 2
	order = 4 # Weyand (2017)
	b2, a2 = signal.butter(N=order, Wn=cut_off/(analog_frequency/2), btype='low')

	ax_filt_l_low = signal.filtfilt(b2, a2, a_x_ankle_l)
	ay_filt_l_low = signal.filtfilt(b2, a2, a_y_ankle_l)
	az_filt_l_low = signal.filtfilt(b2, a2, a_z_ankle_l)

	ax_filt_r_low = signal.filtfilt(b2, a2, a_x_ankle_r)
	ay_filt_r_low = signal.filtfilt(b2, a2, a_y_ankle_r)
	az_filt_r_low = signal.filtfilt(b2, a2, a_z_ankle_r)

	''' Figure out which direction IMU is facing and put in the correct direction if not what is expected '''
	# IMU should be on the medial aspect of the tibia. 
	# Left coordinate system: y = up, z = towards midline, x = forward direction
	# Right coordinate system: y = up, z = towards midline, x = backward direction

	'''
	if abs(max(ax_filt_l)) == max(abs(max(ax_filt_l)), abs(min(ax_filt_l))): # x should be positive
		pass
	else:
		# Rotate 180 deg around the y axis
		ax_filt_l = -ax_filt_l
		az_filt_l = -az_filt_l

	if abs(min(ax_filt_r)) == max(abs(max(ax_filt_r)), abs(min(ax_filt_r))): # x should be negative
		pass
	else:
		# Rotate 180 deg around the y axis
		ax_filt_r = -ax_filt_r
		az_filt_r = -az_filt_r
	'''

	# Because the IMU's are in different orientations for the left and right leg, we will treat these separately

	Fx_filt = signal.filtfilt(b_f, a_f, grf_x)
	Fy_filt = signal.filtfilt(b_f, a_f, grf_y)
	Fz_filt = signal.filtfilt(b_f, a_f, grf_z)

	# Flip vertical forces as z (vertical axis in force plate coordinate system) is negative - this is just to get contact points.
	# Will not be used in final version.
	Fz_filt = -Fz_filt
	grf_z = -grf_z

	# Find when the vertical forces drop below threshold, and then make all of the forces 0.0 at these points.
	force_threshold = 20

	# Find the indices where the vertical force is below the threshold
	force_zero = np.where(Fz_filt < force_threshold)

	# Set these values to 0
	Fx_filt[force_zero] = 0.0
	Fy_filt[force_zero] = 0.0
	Fz_filt[force_zero] = 0.0

	R_force = np.sqrt(np.power(Fx_filt, 2) + np.power(Fy_filt, 2) + np.power(Fz_filt, 2))
	# Get the points where there is force applied to the force plate (stance phase). Beginning = heel strike, end = toe off
	heel_strike = []
	toe_off = []

	for i in range(1, len(Fz_filt)-1):
		if Fz_filt[i-1] == 0 and Fz_filt[i] != 0:
			heel_strike.append(i-1)
		
		if Fz_filt[i+1] == 0 and Fz_filt[i] != 0:
			toe_off.append(i+1)
	

	# Check that the heel strike and toe off events are in the correct order
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


	# Low-pass filter at 2 Hz, 5 Hz, 10 Hz, 20 Hz, 40 Hz, 50 Hz
	analog_frequency = 1000
	order = 4 # Weyand (2017)

	cut_off = [2, 5, 10, 20, 40, 50]
	c = ['r', 'g', 'b', 'm', 'c', 'y']
	'''
	fig, axs = plt.subplots(3)
	axs[0].set_title('ax')
	axs[1].set_title('ay')
	axs[2].set_title('az')

	for i in range(len(cut_off)):
		b, a = signal.butter(N=order, Wn=cut_off[i]/(analog_frequency/2), btype='low')

		ax_filt_l = signal.filtfilt(b, a, a_x_ankle_l)
		ay_filt_l = signal.filtfilt(b, a, a_y_ankle_l)
		az_filt_l = signal.filtfilt(b, a, a_z_ankle_l)

		axs[0].plot(time[:4000], ax_filt_l[:4000], c[i])
		axs[1].plot(time[:4000], ay_filt_l[:4000], c[i], label=(str(cut_off[i]) + ' Hz'))
		axs[2].plot(time[:4000], az_filt_l[:4000], c[i])

		axs[0].plot((time[heel_strike])[:11], (ax_filt_l[heel_strike])[:11], 'ok')
		axs[0].plot((time[toe_off])[:11], (ax_filt_l[toe_off])[:11], 'ok')

		axs[1].plot((time[heel_strike])[:11], (ay_filt_l[heel_strike])[:11], 'ok')
		axs[1].plot((time[toe_off])[:11], (ay_filt_l[toe_off])[:11], 'ok')

		axs[2].plot((time[heel_strike])[:11], (az_filt_l[heel_strike])[:11], 'ok')
		axs[2].plot((time[toe_off])[:11], (az_filt_l[toe_off])[:11], 'ok')
		
	axs[1].legend()
	plt.show()

	'''
	# Plot 2 seconds of data (2000 points) for observation
	fig, axs = plt.subplots(2)

	ax = ax_filt_r_low.copy()
	ay = ay_filt_r_low.copy()
	az = az_filt_r_low.copy()
	"""
	nsamples = ax.size

	# regularize datasets by subtracting mean and dividing by s.d.
	ax -= ax.mean(); ax /= ax.std()
	az -= az.mean(); az /= az.std()

	from scipy.signal import correlate
	import scipy
	import pylab as pyl

	# Find cross-correlation
	xcorr = correlate(ax, az)
	dt = np.arange(1-nsamples, nsamples)

	recovered_time_shift = dt[xcorr.argmax()]

	print(recovered_time_shift)

	''' TRY: shift HS depending on how in/out of phase ax and az are '''
	# Get frequency of oscillations
	mins = argrelextrema(az, np.less)[0]

	difference = []
	for i in range(len(mins) - 1):
		difference.append(mins[i+1] - mins[i])
	
	period_av = np.mean(np.array(difference))

	# az and ax would be completely out of phase if the recovered_time_shift was period_av / 2
	out_of_phase = period_av / 2
	
	# So split into quaters.
	cut_off_1 = period_av / 4
	cut_off_2 = out_of_phase

	# If the shift is between cut_off 1 and cut_off 3, 
	if abs(recovered_time_shift) < cut_off_1:
		shift = int(recovered_time_shift)

	# If the shift is greater than cut_off 2
	elif recovered_time_shift > cut_off_2:
		shift = int(recovered_time_shift - out_of_phase)

	elif recovered_time_shift < -cut_off_2:
		shift = int(recovered_time_shift + out_of_phase)

	print(shift)
	"""

	# Acceleration data
	axs[1].plot(time[:4000], ax[:4000],'r', label='x ankle')
	axs[1].plot(time[:4000], ay[:4000],'g', label='y ankle')
	axs[1].plot(time[:4000], az[:4000],'b', label='z ankle')
	#axs[1].plot((time[toe_off])[:11], (ay[toe_off])[:11], 'ok', label='toe off')
	axs[1].plot((time[heel_strike])[:11], (az[heel_strike])[:11], 'ok', label='heel strike')
	axs[1].set_xlabel('Time (s)')
	axs[1].set_ylabel('Acceleration (mm/s^2)')
	axs[1].legend()

	# Scale force so viewing is easier (arbitrary y axis)
	max_R = max(R_ankle_r)
	max_F = max(Fz_filt)

	scale_factor = max_F / max_R

	Fz_filt = Fz_filt / scale_factor

	# Vertical GRF & resultant acceleration
	axs[0].plot(time[:4000], Fz_filt[:4000],'k', label='Vertical GRF')
	axs[0].plot(time[:4000], ay_filt_r[:4000],'m', label='ay ankle')
	axs[0].set_ylabel('Force/Acceleration (arbitrary)')
	axs[0].set_xlabel('Time (s)')
	axs[0].legend()

	plt.show()

	''' Find the HS and TO events of the left foot '''

	HS_l = get_heel_strike_event(time, az_filt_l_low, 'left')
	TO_l = get_toe_off_event(time, ay_filt_l_low, 'left')

	HS_r = get_heel_strike_event(time, az_filt_r_low, 'right')
	TO_r = get_toe_off_event(time, ay_filt_r_low, 'right')

	for i in range(2):
		if i == 0:
			# Pointer to left list
			HS = HS_l
			TO = TO_l
			foot = 'left'
		elif i == 1:
			# Pointer to right list
			HS = HS_r
			TO = TO_r
			foot = 'right'

		# HS event must be first
		while HS[0] > TO[0]:
			# Remove the first TO event
			del TO[0]

		# Length of heel_strike and toe_off should be the same
		while len(HS) != len(TO):
			# Remove last heel strike event (we know that HS is first)
			del HS[-1]
	
		assert len(HS) == len(TO)

		# Check that the order of events is HS, TO, HS, TO, ... etc
		TO_prev = 0
		for i in range(len(HS)):
			# The current HS should occur after the previous TO
			assert HS[i] > TO_prev
			
			# The current TO should occur after the last HS
			assert TO[i] > HS[i]

			TO_prev = TO[i]

	'''
	plt.plot(time, R_ankle_r,'k',label='Resultant Acceleration')
	#plt.plot(time[toe_off], R_ankle_l[toe_off], 'ro', label='Toe-off actual')
	#plt.plot(time[TO], R_ankle_l[TO],'bo', label='Toe-off estimate')

	#plt.plot(time[heel_strike], R_ankle_l[heel_strike], 'go', label='Heel-strike actual')
	#plt.plot(time[HS], R_ankle_l[HS],'mo', label='Heel-strike estimate')

	plt.vlines(x=time[HS_r], ymin=min(R_ankle_r), ymax=max(R_ankle_r), colors='r', label='Heel Strike')

	plt.vlines(x=time[TO_r], ymin=min(R_ankle_r), ymax=max(R_ankle_r), colors='g', label='Toe Off')

	for i in range(len(HS_r)):
		plt.axvspan(time[HS_r[i]], time[TO_r[i]], alpha=0.5, color='grey')

	plt.legend()

	plt.xlabel('time (s)')
	plt.ylabel('acceleration (mm/s^2)')
	plt.show()
	'''
	''' Look at accuracy '''

	# Split heel_strike and toe_off into left and right according to the first indice and comparing to the estimate
	heel_strike_1 = heel_strike[0::2]
	heel_strike_2 = heel_strike[1::2]

	toe_off_1 = toe_off[0::2]
	toe_off_2 = toe_off[1::2]

	# Flags to indicate if the left foot is using heel_strike_1 or heel_strike_2
	flag_1 = 0
	flag_2 = 0

	HS_0 = HS_l[0]
	differences_1 = []
	differences_2 = []

	for i in range(5): # Run through first 5 to find the starting point for the left foot
		differences_1.append(abs(heel_strike_1[i] - HS_0))
		differences_2.append(abs(heel_strike_2[i] - HS_0))

	smallest_diff_1 = differences_1.index(min(differences_1))
	smallest_diff_2 = differences_2.index(min(differences_2))

	if differences_1[smallest_diff_1] < differences_2[smallest_diff_2]:
		# heel strike 1 is for the left foot, starting at smallest_diff_1
		heel_strike_l = heel_strike_1[smallest_diff_1:]
		toe_off_l = toe_off_1[smallest_diff_1:]

		flag_1 = 1

	else:
		# heel strike 2 is for the left foot, starting at smallest_diff_2
		heel_strike_l = heel_strike_2[smallest_diff_2:]
		toe_off_l = toe_off_2[smallest_diff_2:]

		flag_2 = 1

	HS_0 = HS_r[0]
	differences = []

	for i in range(5): # Run through first 5 to find the starting point for the right foot
		if flag_1 == 0:
			differences.append(abs(heel_strike_1[i] - HS_0))
		elif flag_2 == 0:
			differences.append(abs(heel_strike_2[i] - HS_0))

	smallest_diff = differences.index(min(differences))

	if flag_1 == 0:
		heel_strike_r = heel_strike_1[smallest_diff:]
		toe_off_r = toe_off_1[smallest_diff:]
	elif flag_2 == 0:
		heel_strike_r = heel_strike_2[smallest_diff:]
		toe_off_r = toe_off_2[smallest_diff:]

	# Length of estimated events should be the same as the length of the actual events
	if len(HS_l) != len(heel_strike_l):
		print('Number of estimated left foot events is different to number of actual events for the left foot.')

	diff_l = (len(HS_l) - len(heel_strike_l))

	if len(HS_r) != len(heel_strike_r):
		print('Number of estimated right foot events is different to number of actual events for the right foot.')

	diff_r = (len(HS_r) - len(heel_strike_r))

	left_differences_HS = [] # In time units
	left_differences_TO = [] # In time units

	if diff_l <= 0: # More actual events
		for i in range(len(HS_l)):
			left_differences_HS.append((time[HS_l[i]] - time[heel_strike_l[i]]))
			left_differences_TO.append((time[TO_l[i]] - time[toe_off_l[i]]))

			left_differences_HS_all.append((time[HS_l[i]] - time[heel_strike_l[i]]))
			left_differences_TO_all.append((time[TO_l[i]] - time[toe_off_l[i]]))

	elif diff_l > 0: # More estimated events
		for i in range(len(heel_strike_l)):
			left_differences_HS.append((time[HS_l[i]] - time[heel_strike_l[i]]))
			left_differences_TO.append((time[TO_l[i]] - time[toe_off_l[i]]))

			left_differences_HS_all.append((time[HS_l[i]] - time[heel_strike_l[i]]))
			left_differences_TO_all.append((time[TO_l[i]] - time[toe_off_l[i]]))

	right_differences_HS = [] # In time units
	right_differences_TO = [] # In time units

	if diff_r <= 0: # More actual events
		for i in range(len(HS_r)):
			right_differences_HS.append((time[HS_r[i]] - time[heel_strike_r[i]]))
			right_differences_TO.append((time[TO_r[i]] - time[toe_off_r[i]]))

			right_differences_HS_all.append((time[HS_r[i]] - time[heel_strike_r[i]]))
			right_differences_TO_all.append((time[TO_r[i]] - time[toe_off_r[i]]))

	elif diff_r > 0: # More estimated events
		for i in range(len(heel_strike_r)):
			right_differences_HS.append((time[HS_r[i]] - time[heel_strike_r[i]]))
			right_differences_TO.append((time[TO_r[i]] - time[toe_off_r[i]]))

			right_differences_HS_all.append((time[HS_r[i]] - time[heel_strike_r[i]]))
			right_differences_TO_all.append((time[TO_r[i]] - time[toe_off_r[i]]))
	'''
	fig, axs = plt.subplots(2, 2)

	bp1 = axs[0, 0].boxplot(left_differences_HS)
	plt.setp(bp1['medians'], color='k')

	axs[0, 0].set_ylabel('Time difference (s)')
	axs[0, 0].set_xlabel('')
	axs[0, 0].set_title('Left Heel-strike')

	axs[0, 0].tick_params(
		axis='x',          # changes apply to the x-axis
		which='both',      # both major and minor ticks are affected
		bottom=False,      # ticks along the bottom edge are off
		top=False,         # ticks along the top edge are off
		labelbottom=False)

	bp2 = axs[0, 1].boxplot(left_differences_TO)
	plt.setp(bp2['medians'], color='k')
	axs[0, 1].set_ylabel('Time difference (s)')
	axs[0, 1].set_xlabel('')
	axs[0, 1].set_title('Left Toe-off')

	axs[0, 1].tick_params(
		axis='x',          # changes apply to the x-axis
		which='both',      # both major and minor ticks are affected
		bottom=False,      # ticks along the bottom edge are off
		top=False,         # ticks along the top edge are off
		labelbottom=False)

	bp3 = axs[1, 0].boxplot(right_differences_HS)
	plt.setp(bp3['medians'], color='k')
	axs[1, 0].set_ylabel('Time difference (s)')
	axs[1, 0].set_xlabel('')
	axs[1, 0].set_title('Right Heel-strike')

	axs[1, 0].tick_params(
		axis='x',          # changes apply to the x-axis
		which='both',      # both major and minor ticks are affected
		bottom=False,      # ticks along the bottom edge are off
		top=False,         # ticks along the top edge are off
		labelbottom=False)

	bp4 = axs[1, 1].boxplot(right_differences_TO)
	plt.setp(bp4['medians'], color='k')
	axs[1, 1].set_ylabel('Time difference (s)')
	axs[1, 1].set_xlabel('')
	axs[1, 1].set_title('Right Toe-off')

	axs[1, 1].tick_params(
		axis='x',          # changes apply to the x-axis
		which='both',      # both major and minor ticks are affected
		bottom=False,      # ticks along the bottom edge are off
		top=False,         # ticks along the top edge are off
		labelbottom=False)

	plt.show()
	'''
	# Error as a percentage of the event time (from force plate data)
	event_time_l = (time[toe_off_l] - time[heel_strike_l]).tolist()
	event_time_r = (time[toe_off_r] - time[heel_strike_r]).tolist()

	left_differences_per_HS = [] # In time units
	left_differences_per_TO = [] # In time units

	if diff_l <= 0: # More actual events
		for i in range(len(HS_l)):
			left_differences_per_HS.append(left_differences_HS[i] / event_time_l[i] * 100)
			left_differences_per_TO.append(left_differences_TO[i] / event_time_l[i] * 100)

			left_differences_per_HS_all.append(left_differences_HS[i] / event_time_l[i] * 100)
			left_differences_per_TO_all.append(left_differences_TO[i] / event_time_l[i] * 100)

	elif diff_l > 0: # More estimated events
		for i in range(len(heel_strike_l)):
			left_differences_per_HS.append(left_differences_HS[i] / event_time_l[i] * 100)
			left_differences_per_TO.append(left_differences_TO[i] / event_time_l[i] * 100)

			left_differences_per_HS_all.append(left_differences_HS[i] / event_time_l[i] * 100)
			left_differences_per_TO_all.append(left_differences_TO[i] / event_time_l[i] * 100)

	right_differences_per_HS = [] # In time units
	right_differences_per_TO = [] # In time units

	if diff_r <= 0: # More actual events
		for i in range(len(HS_r)):
			right_differences_per_HS.append(right_differences_HS[i] / event_time_r[i] * 100)
			right_differences_per_TO.append(right_differences_TO[i] / event_time_r[i] * 100)

			right_differences_per_HS_all.append(right_differences_HS[i] / event_time_r[i] * 100)
			right_differences_per_TO_all.append(right_differences_TO[i] / event_time_r[i] * 100)

	elif diff_r > 0: # More estimated events
		for i in range(len(heel_strike_r)):
			right_differences_per_HS.append(right_differences_HS[i] / event_time_r[i] * 100)
			right_differences_per_TO.append(right_differences_TO[i] / event_time_r[i] * 100)

			right_differences_per_HS_all.append(right_differences_HS[i] / event_time_r[i] * 100)
			right_differences_per_TO_all.append(right_differences_TO[i] / event_time_r[i] * 100)

	'''
	fig, axs = plt.subplots(2, 2)

	bp1 = axs[0, 0].boxplot(left_differences_per_HS)
	plt.setp(bp1['medians'], color='k')

	axs[0, 0].set_ylabel('Time difference (% event time)')
	axs[0, 0].set_xlabel('')
	axs[0, 0].set_title('Left Heel-strike')

	axs[0, 0].tick_params(
		axis='x',          # changes apply to the x-axis
		which='both',      # both major and minor ticks are affected
		bottom=False,      # ticks along the bottom edge are off
		top=False,         # ticks along the top edge are off
		labelbottom=False)

	bp2 = axs[0, 1].boxplot(left_differences_per_TO)
	plt.setp(bp2['medians'], color='k')
	axs[0, 1].set_ylabel('Time difference (% event time)')
	axs[0, 1].set_xlabel('')
	axs[0, 1].set_title('Left Toe-off')

	axs[0, 1].tick_params(
		axis='x',          # changes apply to the x-axis
		which='both',      # both major and minor ticks are affected
		bottom=False,      # ticks along the bottom edge are off
		top=False,         # ticks along the top edge are off
		labelbottom=False)

	bp3 = axs[1, 0].boxplot(right_differences_per_HS)
	plt.setp(bp3['medians'], color='k')
	axs[1, 0].set_ylabel('Time difference (% event time)')
	axs[1, 0].set_xlabel('')
	axs[1, 0].set_title('Right Heel-strike')

	axs[1, 0].tick_params(
		axis='x',          # changes apply to the x-axis
		which='both',      # both major and minor ticks are affected
		bottom=False,      # ticks along the bottom edge are off
		top=False,         # ticks along the top edge are off
		labelbottom=False)

	bp4 = axs[1, 1].boxplot(right_differences_per_TO)
	plt.setp(bp4['medians'], color='k')
	axs[1, 1].set_ylabel('Time difference (% event time)')
	axs[1, 1].set_xlabel('')
	axs[1, 1].set_title('Right Toe-off')

	axs[1, 1].tick_params(
		axis='x',          # changes apply to the x-axis
		which='both',      # both major and minor ticks are affected
		bottom=False,      # ticks along the bottom edge are off
		top=False,         # ticks along the top edge are off
		labelbottom=False)

	plt.show()
	'''
	# See percentage difference between estimated contact time (HS to TO) and actual contact time
	contact_time_l_diff = []
	contact_time_r_diff = []

	if diff_l <= 0: # More actual events
		for i in range(len(HS_l)):
			# Contact time = toe off - heel strike
			contact_time_l_est = time[TO_l[i]] - time[HS_l[i]]
			contact_time_l = time[toe_off_l[i]] - time[heel_strike_l[i]]

			# Percentage difference
			contact_time_l_diff.append((contact_time_l_est - contact_time_l) / contact_time_l * 100)
			contact_time_l_diff_all.append((contact_time_l_est - contact_time_l) / contact_time_l * 100)

	elif diff_l > 0: # More estimated events
		for i in range(len(heel_strike_l)):
			# Contact time = toe off - heel strike
			contact_time_l_est = time[TO_l[i]] - time[HS_l[i]]
			contact_time_l = time[toe_off_l[i]] - time[heel_strike_l[i]]

			# Percentage difference
			contact_time_l_diff.append((contact_time_l_est - contact_time_l) / contact_time_l * 100)
			contact_time_l_diff_all.append((contact_time_l_est - contact_time_l) / contact_time_l * 100)

	if diff_r <= 0: # More actual events
		for i in range(len(HS_r)):
			# Contact time = toe off - heel strike
			contact_time_r_est = time[TO_r[i]] - time[HS_r[i]]
			contact_time_r = time[toe_off_r[i]] - time[heel_strike_r[i]]

			# Percentage difference
			contact_time_r_diff.append((contact_time_r_est - contact_time_r) / contact_time_r * 100)
			contact_time_r_diff_all.append((contact_time_r_est - contact_time_r) / contact_time_r * 100)
	elif diff_r > 0: # More estimated events
		for i in range(len(heel_strike_r)):
			# Contact time = toe off - heel strike
			contact_time_r_est = time[TO_r[i]] - time[HS_r[i]]
			contact_time_r = time[toe_off_r[i]] - time[heel_strike_r[i]]

			# Percentage difference
			contact_time_r_diff.append((contact_time_r_est - contact_time_r) / contact_time_r * 100)
			contact_time_r_diff_all.append((contact_time_r_est - contact_time_r) / contact_time_r * 100)
	'''
	fig, axs = plt.subplots(1, 2)

	bp1 = axs[0].boxplot(contact_time_l_diff)
	plt.setp(bp1['medians'], color='k')

	axs[0].set_ylabel('Contact time difference (%)')
	axs[0].set_xlabel('')
	axs[0].set_title('Left Contact time')

	axs[0].tick_params(
		axis='x',          # changes apply to the x-axis
		which='both',      # both major and minor ticks are affected
		bottom=False,      # ticks along the bottom edge are off
		top=False,         # ticks along the top edge are off
		labelbottom=False)

	bp2 = axs[1].boxplot(contact_time_r_diff)
	plt.setp(bp2['medians'], color='k')
	axs[1].set_ylabel('Contact time difference (%)')
	axs[1].set_xlabel('')
	axs[1].set_title('Right Contact time')

	axs[1].tick_params(
		axis='x',          # changes apply to the x-axis
		which='both',      # both major and minor ticks are affected
		bottom=False,      # ticks along the bottom edge are off
		top=False,         # ticks along the top edge are off
		labelbottom=False)

	plt.show()
	'''
	# Estimate vertical GRF's
	mass = 70.0
	#force_l = estimate_vGRF(time, TO_l, HS_l, ay_filt_l, mass)

	# Calculate RMSE
	left_differences_HS = np.array(left_differences_HS)
	HS_RMSE_l = np.sqrt(np.sum(np.power(left_differences_HS, 2)) / len(left_differences_HS))
	HS_RMSE_r = np.sqrt(np.sum(np.power(right_differences_HS, 2)) / len(right_differences_HS))

	TO_RMSE_l = np.sqrt(np.sum(np.power(left_differences_TO, 2)) / len(left_differences_TO))
	TO_RMSE_r = np.sqrt(np.sum(np.power(right_differences_TO, 2)) / len(right_differences_TO))

HS_RMSE_all_l = np.sqrt(np.sum(np.power(left_differences_HS_all, 2)) / len(left_differences_HS_all))
HS_RMSE_all_r = np.sqrt(np.sum(np.power(right_differences_HS_all, 2)) / len(right_differences_HS_all))

TO_RMSE_all_l = np.sqrt(np.sum(np.power(left_differences_TO_all, 2)) / len(left_differences_TO_all))
TO_RMSE_all_r = np.sqrt(np.sum(np.power(right_differences_TO_all, 2)) / len(right_differences_TO_all))

print('Left HS RMSE: ' + str(HS_RMSE_all_l) + ' s')
print('Right HS RMSE: ' + str(HS_RMSE_all_r) + ' s')
print('Left TO RMSE: ' + str(TO_RMSE_all_l) + ' s')
print('Right TO RMSE: ' + str(TO_RMSE_all_r) + ' s')

fig, axs = plt.subplots(1, 2)

bp1 = axs[0].boxplot(contact_time_l_diff_all)
plt.setp(bp1['medians'], color='k')

axs[0].set_ylabel('Contact time difference (%)')
axs[0].set_xlabel('')
axs[0].set_title('Left Contact time')

axs[0].tick_params(
	axis='x',          # changes apply to the x-axis
	which='both',      # both major and minor ticks are affected
	bottom=False,      # ticks along the bottom edge are off
	top=False,         # ticks along the top edge are off
	labelbottom=False)

bp2 = axs[1].boxplot(contact_time_r_diff_all)
plt.setp(bp2['medians'], color='k')
axs[1].set_ylabel('Contact time difference (%)')
axs[1].set_xlabel('')
axs[1].set_title('Right Contact time')

axs[1].tick_params(
	axis='x',          # changes apply to the x-axis
	which='both',      # both major and minor ticks are affected
	bottom=False,      # ticks along the bottom edge are off
	top=False,         # ticks along the top edge are off
	labelbottom=False)

plt.show()

#######
fig, axs = plt.subplots(1, 2)

bp1 = axs[0].boxplot(left_differences_per_HS_all)
plt.setp(bp1['medians'], color='k')

axs[0].set_ylabel('Contact time difference (%)')
axs[0].set_xlabel('')
axs[0].set_title('Left HS time')

axs[0].tick_params(
	axis='x',          # changes apply to the x-axis
	which='both',      # both major and minor ticks are affected
	bottom=False,      # ticks along the bottom edge are off
	top=False,         # ticks along the top edge are off
	labelbottom=False)

bp2 = axs[1].boxplot(right_differences_per_HS_all)
plt.setp(bp2['medians'], color='k')
axs[1].set_ylabel('HS difference (% contact time)')
axs[1].set_xlabel('')
axs[1].set_title('Right HS time')

axs[1].tick_params(
	axis='x',          # changes apply to the x-axis
	which='both',      # both major and minor ticks are affected
	bottom=False,      # ticks along the bottom edge are off
	top=False,         # ticks along the top edge are off
	labelbottom=False)

plt.show()

#######
fig, axs = plt.subplots(1, 2)

bp1 = axs[0].boxplot(left_differences_per_TO_all)
plt.setp(bp1['medians'], color='k')

axs[0].set_ylabel('TO difference (% contact time)')
axs[0].set_xlabel('')
axs[0].set_title('Left TO time')

axs[0].tick_params(
	axis='x',          # changes apply to the x-axis
	which='both',      # both major and minor ticks are affected
	bottom=False,      # ticks along the bottom edge are off
	top=False,         # ticks along the top edge are off
	labelbottom=False)

bp2 = axs[1].boxplot(right_differences_per_TO_all)
plt.setp(bp2['medians'], color='k')
axs[1].set_ylabel('TO difference (% contact time)')
axs[1].set_xlabel('')
axs[1].set_title('Right TO time')

axs[1].tick_params(
	axis='x',          # changes apply to the x-axis
	which='both',      # both major and minor ticks are affected
	bottom=False,      # ticks along the bottom edge are off
	top=False,         # ticks along the top edge are off
	labelbottom=False)

plt.show()