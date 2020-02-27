import numpy as np
from matplotlib import pyplot as plt
import os
from scipy import signal
from scipy.signal import argrelextrema
import csv

'''
The purpose of this script is to visualise IMU acceleration data from running trials.

Alex Woodall
Auckland Bioengineering Institution

'''

''' Read in file '''
data_directory = 'C:\\Users\\alexw\\Desktop\\RunningData\\0126run2.csv'

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

if abs(max(ax_filt_l)) == max(abs(max(ax_filt_l)), abs(min(ax_filt_l))): # x should be positive
	pass
else:
	# Rotate 180 deg around the y axis
	ax_filt_l = -ax_filt_l
	az_filt_l = -az_filt_l

if abs(min(ay_filt_r)) == max(abs(max(ay_filt_r)), abs(min(ay_filt_r))): # x should be negative
	pass
else:
	# Rotate 180 deg around the y axis
	ax_filt_r = -ax_filt_r
	az_filt_r = -az_filt_r

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
'''
# Plot 2 seconds of data (2000 points) for observation
fig, axs = plt.subplots(2)

# Acceleration data
axs[1].plot(time[:4000], ax_filt_r_low[:4000],'r', label='x ankle')
axs[1].plot(time[:4000], ay_filt_r_low[:4000],'g', label='y ankle')
axs[1].plot(time[:4000], az_filt_r_low[:4000],'b', label='z ankle')
axs[1].plot((time[toe_off])[:11], (ay_filt_r_low[toe_off])[:11], 'ok', label='toe off')
axs[1].plot((time[heel_strike])[:11], (az_filt_r_low[heel_strike])[:11], 'om', label='heel strike')
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
axs[0].plot(time[:4000], R_ankle_r[:4000],'m', label='R ankle')
axs[0].set_ylabel('Force/Acceleration (arbitrary)')
axs[0].set_xlabel('Time (s)')
axs[0].legend()

plt.show()
'''
''' Find the HS and TO events of the left foot '''

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
	if HS[0] > TO[0]:
		del TO[0]

	diff = len(HS) - len(TO)

	# Length of heel_strike and toe_off should be the same
	while diff != 0:
		# HS event must be first
		if HS[0] > TO[0]:
			del TO[0]

		# Should be the same number of HS and TO events
		if len(HS) > len(TO):
			HS.pop() # Remove last heel strike event

		elif len(TO) > len(HS):
			TO.pop() # Remove last toe off event

		diff = len(HS) - len(TO)
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

# HS event must be first
if heel_strike[0] > toe_off[0]:
	del toe_off[0]

diff = len(heel_strike) - len(toe_off)

# Length of heel_strike and toe_off should be the same
while diff != 0:
	# HS event must be first
	if heel_strike[0] > toe_off[0]:
		del toe_off[0]

	# Should be the same number of HS and TO events
	if len(heel_strike) > len(toe_off):
		heel_strike.pop() # Remove last heel strike event

	elif len(toe_off) > len(heel_strike):
		toe_off.pop() # Remove last toe off event

	diff = len(heel_strike) - len(toe_off)

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

if diff_l < 0: # More actual events
	for i in range(len(HS_l)):
		left_differences_HS.append(abs(time[HS_l[i]] - time[heel_strike_l[i]]))
		left_differences_TO.append(abs(time[TO_l[i]] - time[toe_off_l[i]]))

elif diff_l > 0: # More estimated events
	for i in range(len(heel_strike_l)):
		left_differences_HS.append(abs(time[HS_l[i]] - time[heel_strike_l[i]]))
		left_differences_TO.append(abs(time[TO_l[i]] - time[toe_off_l[i]]))

else: # Same number of estimated and actual events
	for i in range(len(HS_l)):
		left_differences_HS.append(abs(time[HS_l[i]] - time[heel_strike_l[i]]))
		left_differences_TO.append(abs(time[TO_l[i]] - time[toe_off_l[i]]))

right_differences_HS = [] # In time units
right_differences_TO = [] # In time units

if diff_r < 0: # More actual events
	for i in range(len(HS_r)):
		right_differences_HS.append(abs(time[HS_r[i]] - time[heel_strike_r[i]]))
		right_differences_TO.append(abs(time[TO_r[i]] - time[toe_off_r[i]]))

elif diff_l > 0: # More estimated events
	for i in range(len(heel_strike_l)):
		right_differences_HS.append(abs(time[HS_r[i]] - time[heel_strike_r[i]]))
		right_differences_TO.append(abs(time[TO_r[i]] - time[toe_off_r[i]]))

else: # Same number of estimated and actual events
	for i in range(len(HS_l)):
		right_differences_HS.append(abs(time[HS_r[i]] - time[heel_strike_r[i]]))
		right_differences_TO.append(abs(time[TO_r[i]] - time[toe_off_r[i]]))

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

# Error as a percentage of the event time (from force plate data)
event_time_l = (time[toe_off_l] - time[heel_strike_l]).tolist()
event_time_r = (time[toe_off_r] - time[heel_strike_r]).tolist()

left_differences_per_HS = [] # In time units
left_differences_per_TO = [] # In time units

if diff_l < 0: # More actual events
	for i in range(len(HS_l)):
		left_differences_per_HS.append(left_differences_HS[i] / event_time_l[i] * 100)
		left_differences_per_TO.append(left_differences_TO[i] / event_time_l[i] * 100)

elif diff_l > 0: # More estimated events
	for i in range(len(heel_strike_l)):
		left_differences_per_HS.append(left_differences_HS[i] / event_time_l[i] * 100)
		left_differences_per_TO.append(left_differences_TO[i] / event_time_l[i] * 100)

else: # Same number of estimated and actual events
	for i in range(len(HS_l)):
		left_differences_per_HS.append(left_differences_HS[i] / event_time_l[i] * 100)
		left_differences_per_TO.append(left_differences_TO[i] / event_time_l[i] * 100)

right_differences_per_HS = [] # In time units
right_differences_per_TO = [] # In time units

if diff_r < 0: # More actual events
	for i in range(len(HS_r)):
		right_differences_per_HS.append(right_differences_HS[i] / event_time_r[i] * 100)
		right_differences_per_TO.append(right_differences_TO[i] / event_time_r[i] * 100)

elif diff_l > 0: # More estimated events
	for i in range(len(heel_strike_l)):
		right_differences_per_HS.append(right_differences_HS[i] / event_time_r[i] * 100)
		right_differences_per_TO.append(right_differences_TO[i] / event_time_r[i] * 100)

else: # Same number of estimated and actual events
	for i in range(len(HS_l)):
		right_differences_per_HS.append(right_differences_HS[i] / event_time_r[i] * 100)
		right_differences_per_TO.append(right_differences_TO[i] / event_time_r[i] * 100)


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