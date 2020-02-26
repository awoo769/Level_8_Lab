import numpy as np
from matplotlib import pyplot as plt
import os
from scipy import signal
import csv

'''
The purpose of this script is to visualise IMU acceleration data from running trials.

Alex Woodall
Auckland Bioengineering Institution

'''

''' Read in file '''
data_directory = 'C:\\Users\\alexw\\Desktop\\RunningData\\0130run2.csv'

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
a_y_ankle_r = (np.array(read_file)[:,9]).astype(np.float) # 9th column
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

''' Figure out which direction IMU is facing and put in the correct direction if not what is expected '''
# IMU should be on the medial aspect of the tibia. 
# Left coordinate system: y = up, z = towards midline, x = forward direction
# Right coordinate system: y = down, z = towards midline, x = forward direction

if abs(max(ay_filt_l)) == max(abs(max(ay_filt_l)), abs(min(ay_filt_l))): # y should be positive
	pass
else:
	# Rotate 180 deg around the z axis
	ay_filt_l = -ay_filt_l
	ax_filt_l = -ax_filt_l

if abs(min(ay_filt_r)) == max(abs(max(ay_filt_r)), abs(min(ay_filt_r))): # y should be negative
	pass
else:
	# Rotate 180 deg around the z axis
	ay_filt_r = -ay_filt_r
	ax_filt_r = -ax_filt_r

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
Fz_filt[force_zero] = 0.0

# Get the points where there is force applied to the force plate (stance phase). Beginning = heel strike, end = toe off
heel_strike = []
toe_off = []

for i in range(1, len(Fz_filt)-1):
	if Fz_filt[i-1] == 0 and Fz_filt[i] != 0:
		heel_strike.append(i-1)
	
	if Fz_filt[i+1] == 0 and Fz_filt[i] != 0:
		toe_off.append(i+1)

# Plot 2 seconds of data (2000 points) for observation
fig, axs = plt.subplots(2)

# Acceleration data
axs[1].plot(time[:4000], ax_filt_l_low[:4000],'r', label='x ankle')
axs[1].plot(time[:4000], ay_filt_l_low[:4000],'g', label='y ankle')
axs[1].plot(time[:4000], az_filt_l_low[:4000],'b', label='z ankle')
axs[1].plot((time[heel_strike])[:11], (ax_filt_l_low[heel_strike])[:11], 'or', label='toe off')
axs[1].plot((time[heel_strike])[:11], (az_filt_l_low[heel_strike])[:11], 'or', label='toe off')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Acceleration (mm/s^2)')
axs[1].legend()

# Scale force so viewing is easier (arbitrary y axis)
max_R = max(R_ankle_l)
max_F = max(Fz_filt)

scale_factor = max_F / max_R

Fz_filt = Fz_filt / scale_factor

# Vertical GRF & resultant acceleration
axs[0].plot(time[:4000], Fz_filt[:4000],'k', label='Vertical GRF')
axs[0].plot(time[:4000], R_ankle_l[:4000],'m', label='R ankle')
axs[0].set_ylabel('Force/Acceleration (arbitrary)')
axs[0].set_xlabel('Time (s)')
axs[0].legend()

plt.show()

''' Find the HS and TO events of the left foot '''

def get_heel_strike_left(time: np.array, az_filt_l_low: np.array):
	
	'''
	This function estimates the heel strike event for the left foot using IMU data. The IMU should be placed on the medial
	aspect of the tibia.

	The z acceleration data should be filtered at 2 Hz (low pass butter worth filter, n = 4)

	'''

	# Find all maximas and minimas of acceleration in the y direction

	# All minimas
	HS = np.where(np.r_[True, az_filt_l_low[1:] < az_filt_l_low[:-1]] & np.r_[az_filt_l_low[:-1] < az_filt_l_low[1:], True] == True)[0]

	return HS.tolist()

def get_toe_off_left(time: np.array, ay_filt_l_low: np.array):
	
	'''
	This function estimates the toe off event for the left foot using IMU data. The IMU should be placed on the medial
	aspect of the tibia.

	'''

	# Find all maximas acceleration in the z direction

	# All maximas
	TO = np.where(np.r_[True, ay_filt_l_low[1:] > ay_filt_l_low[:-1]] & np.r_[ay_filt_l_low[:-1] > ay_filt_l_low[1:], True] == True)[0]
	return TO.tolist()

HS = get_heel_strike_left(time, az_filt_l_low)
TO = get_toe_off_left(time, ay_filt_l_low)

# HS and TO should have the same length
if len(HS) != len(TO):

	# Get difference between lengths
	diff = len(HS) - len(TO)

	# See which event occurs first.
	HS_1 = HS[0]
	TO_1 = TO[0]

	if abs(diff) == 1:
		if HS_1 < TO_1:
			HS.pop()
		else:
			del TO[0]
	
	else:
		print('Check data set')


plt.plot(time, R_ankle_l,'k',label='Resultant Acceleration')
#plt.plot(time[toe_off], R_ankle_l[toe_off], 'ro', label='Toe-off actual')
#plt.plot(time[TO], R_ankle_l[TO],'bo', label='Toe-off estimate')

#plt.plot(time[heel_strike], R_ankle_l[heel_strike], 'go', label='Heel-strike actual')
#plt.plot(time[HS], R_ankle_l[HS],'mo', label='Heel-strike estimate')

plt.vlines(x=time[HS], ymin=min(R_ankle_l), ymax=max(R_ankle_l), colors='r', label='Heel Strike')

plt.vlines(x=time[TO], ymin=min(R_ankle_l), ymax=max(R_ankle_l), colors='g', label='Toe Off')

for i in range(len(HS)):
	plt.axvspan(time[HS[i]], time[TO[i]], alpha=0.5, color='grey')

plt.legend()

plt.xlabel('time (s)')
plt.ylabel('acceleration (mm/s^2)')
plt.show()