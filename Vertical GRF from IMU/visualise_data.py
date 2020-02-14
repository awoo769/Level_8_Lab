import numpy as np
from matplotlib import pyplot as plt
import os
from scipy import signal
import csv

# Read in file
data_directory = 'C:\\Users\\alexw\\Dropbox\\ABI\\Level_8_Lab\\Vertical GRF from IMU\\DataforSonia\\0102run2.csv'

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

# Filter data at 25 Hz
analog_frequency = 1000
cut_off = 25 # Weyand (2017)
order = 4 # Weyand (2017)
b, a = signal.butter(N=order, Wn=cut_off/(analog_frequency/2), btype='low')

ax_filt_l = signal.filtfilt(b, a, a_x_ankle_l)
ay_filt_l = signal.filtfilt(b, a, a_y_ankle_l)
az_filt_l = signal.filtfilt(b, a, a_z_ankle_l)
R_ankle_l = np.sqrt(np.power(ax_filt_l, 2) + np.power(ay_filt_l, 2) + np.power(az_filt_l, 2))

Rxz_ankle_l = np.sqrt(np.power(ax_filt_l, 2) + np.power(az_filt_l, 2))

ax_filt_r = signal.filtfilt(b, a, a_x_ankle_r)
ay_filt_r = signal.filtfilt(b, a, a_y_ankle_r)
az_filt_r = signal.filtfilt(b, a, a_z_ankle_r)
R_ankle_r = np.sqrt(np.power(ax_filt_r, 2) + np.power(ay_filt_r, 2) + np.power(az_filt_r, 2))

Rxz_ankle_r = np.sqrt(np.power(ax_filt_r, 2) + np.power(az_filt_r, 2))

Fx_filt = signal.filtfilt(b, a, grf_x)
Fy_filt = signal.filtfilt(b, a, grf_y)
Fz_filt = signal.filtfilt(b, a, grf_z)
R_F = np.sqrt(np.power(Fx_filt, 2) + np.power(Fy_filt, 2) + np.power(Fz_filt, 2))

# Flip vertical forces
Fz_filt = -Fz_filt

# Find when the vertical forces drop below threshold, and then make all of the forces 0.0 at these points.
force_threshold = 20

# Find the indices where the vertical force is below our threshold
force_zero = np.where(Fz_filt < force_threshold)

# Set these values to 0
Fx_filt[force_zero] = 0.0
Fy_filt[force_zero] = 0.0
Fz_filt[force_zero] = 0.0
R_F[force_zero] = 0.0

'''
# Find out whether the first step is with the left or right foot.
plt.plot(time,ax_filt_l,'b', label='Left ankle')
plt.plot(time,-ax_filt_r,'r', label='Right ankle')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('x acceleration (mm/s^2)')
plt.show()
'''

# Assuming z is vertical direction
heel_strike = []
toe_off = []

# First step is left. - we don't know this yet

# Get the points where there is force applied to the force plate (stance phase). Beginning = HS, end = TO
for i in range(1, len(Fz_filt)-1):
	if Fz_filt[i-1] == 0 and Fz_filt[i] != 0: 
		heel_strike.append(i-1)
	
	if Fz_filt[i+1] == 0 and Fz_filt[i] != 0:
		toe_off.append(i+1)

# Find out whether the first step is with the left or right foot.


fig, ax1 = plt.subplots()
ax1.plot(time,(Rxz_ankle_l + R_ankle_l)/2,'b', label='Left ankle')
ax1.plot(time,(Rxz_ankle_r + R_ankle_r)/2,'r', label='Right ankle')

ax1.vlines(x=time[heel_strike], ymin=0, ymax=max(max((Rxz_ankle_l + R_ankle_l)/2), max((Rxz_ankle_r + R_ankle_r)/2)), colors='g', label='Heel strike')
ax1.vlines(x=time[toe_off], ymin=0, ymax=max(max((Rxz_ankle_l + R_ankle_l)/2), max((Rxz_ankle_r + R_ankle_r)/2)), colors='y', label='Toe off')

ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Resultant acceleration (mm/s^2)')

ax2 = ax1.twinx()
ax2.plot(time,Fz_filt,'k', label='Fz')
ax2.set_ylabel('Vertical force (N)')

fig.tight_layout()

plt.legend()
plt.show()

'''
# Left = every odd
heel_strike_l = heel_strike[0::2]
toe_off_l = toe_off[0::2]

# Right = every even
heel_strike_r = heel_strike[1::2]
toe_off_r = toe_off[1::2]

plt.plot(time, ax_filt_r,'r', label='x')
#plt.plot(time, ay_filt_r,'g', label='y')
#plt.plot(time, az_filt_r,'b', label='z')
#lt.plot(time, R_ankle_l,'k', label='resultant')
#plt.plot(time, az_filt_l,'k', label='z')
#plt.plot(time,Fz_filt,'b',label='Fz_filt')

#plt.plot(time,Fz_filt,'r')
plt.plot(time[heel_strike_r], ax_filt_r[heel_strike_r],'ob', label='heel strike')
plt.plot(time[toe_off_r], ax_filt_r[toe_off_r],'og', label='toe off')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (mm/s^2)')
plt.legend()
plt.show()
'''
'''
Initial working - only good for Run30.csv

# Left
# Find all maximas and minimas of x acceleration
maximas_ind = np.where(np.r_[True, ax_filt_l[1:] > ax_filt_l[:-1]] & np.r_[ax_filt_l[:-1] > ax_filt_l[1:], True] == True)[0]
maximas_acc = ax_filt_l[maximas_ind].tolist()

maximas_acc.sort(reverse=True)
temp = maximas_acc.copy()

max_maxima = max(maximas_acc)

# All accepted maximas should not be less than 1/2 of the highest
sig_maxs = []
sig_maxs.append(max_maxima)
temp.remove(max_maxima)

for i in range(len(temp)):
	delta = max_maxima / temp[i]
	if delta <= 2 and temp[i] > 0:
		sig_maxs.append(temp[i])

# Get indices
sig_maxs_ind = []
for i in range(len(sig_maxs)):
	sig_maxs_ind.append(np.where(ax_filt_l == sig_maxs[i])[0][0])

minimas_ind = np.where(np.r_[True, ax_filt_l[1:] < ax_filt_l[:-1]] & np.r_[ax_filt_l[:-1] < ax_filt_l[1:], True] == True)[0]
minimas_acc = ax_filt_l[minimas_ind].tolist()

minimas_acc.sort()
temp = minimas_acc.copy()

min_minima = min(minimas_acc)

# All accepted maximas should not be less than 1/2 of the highest
sig_mins = []
sig_mins.append(min_minima)
temp.remove(min_minima)

for i in range(len(temp)):
	delta = min_minima / temp[i]
	if delta <= 2 and temp[i] < 0:
		sig_mins.append(temp[i])

# Get indices
sig_mins_ind = []
for i in range(len(sig_mins)):
	sig_mins_ind.append(np.where(ax_filt_l == sig_mins[i])[0][0])

# For each minima, find the minima previous to it. This will be heel strike
ind_HS = []

for i in range(len(sig_mins_ind)):
	ind_temp = sig_mins_ind[i]

	# Get all mins previous to the current indice
	minimas_less = minimas_ind[minimas_ind < ind_temp].tolist()

	# Sort, the indice at the beginning will be the one we want (unless it is greater than 0)
	minimas_less.sort(reverse=True)

	flag = 0
	j = 0
	while flag == 0:
		if ax_filt_l[minimas_less[j]] < 0:
			ind_HS.append(minimas_less[j])
			flag = 1
		else:
			j += 1

# For each maxima, find the maxima after it. This will be toe off
ind_TO = []

for i in range(len(sig_maxs_ind)):
	ind_temp = sig_maxs_ind[i]

	# Get all maxs after the current indice
	maximas_greater = maximas_ind[maximas_ind > ind_temp].tolist()

	# Sort, the indice at the beginning will be the one we want (unless it is greater than 0)
	maximas_greater.sort()

	flag = 0
	j = 0
	while flag == 0:
		if ax_filt_l[maximas_greater[j]] > 0:
			ind_TO.append(maximas_greater[j])
			flag = 1
		else:
			j += 1
	
plt.plot(time, ax_filt_l,'k')
plt.plot(time[ind_HS], ax_filt_l[ind_HS], 'ob', label='heel strike')
plt.plot(time[ind_TO], ax_filt_l[ind_TO], 'og', label='toe off')

plt.xlabel('Time (s)')
plt.ylabel('Acceleration (mm/s^2)')
plt.legend()
plt.show()


# Right
# Find all maximas and minimas of x acceleration
maximas_ind = np.where(np.r_[True, ax_filt_r[1:] > ax_filt_r[:-1]] & np.r_[ax_filt_r[:-1] > ax_filt_r[1:], True] == True)[0]
maximas_acc = ax_filt_r[maximas_ind].tolist()

maximas_acc.sort(reverse=True)
temp = maximas_acc.copy()

max_maxima = max(maximas_acc)

# All accepted maximas should not be less than 1/2 of the highest
sig_maxs = []
sig_maxs.append(max_maxima)
temp.remove(max_maxima)

for i in range(len(temp)):
	delta = max_maxima / temp[i]
	if delta <= 2 and temp[i] > 0:
		sig_maxs.append(temp[i])

# Get indices
sig_maxs_ind = []
for i in range(len(sig_maxs)):
	sig_maxs_ind.append(np.where(ax_filt_r == sig_maxs[i])[0][0])

minimas_ind = np.where(np.r_[True, ax_filt_r[1:] < ax_filt_r[:-1]] & np.r_[ax_filt_r[:-1] < ax_filt_r[1:], True] == True)[0]
minimas_acc = ax_filt_r[minimas_ind].tolist()

minimas_acc.sort()
temp = minimas_acc.copy()

min_minima = min(minimas_acc)

# All accepted maximas should not be less than 1/2 of the highest
sig_mins = []
sig_mins.append(min_minima)
temp.remove(min_minima)

for i in range(len(temp)):
	delta = min_minima / temp[i]
	if delta <= 2 and temp[i] < 0:
		sig_mins.append(temp[i])

# Get indices
sig_mins_ind = []
for i in range(len(sig_mins)):
	sig_mins_ind.append(np.where(ax_filt_r == sig_mins[i])[0][0])

# For each minima, find the minima previous to it. This will be heel strike
ind_HS = []

for i in range(len(sig_mins_ind)):
	ind_temp = sig_mins_ind[i]

	# Get all mins previous to the current indice
	minimas_less = minimas_ind[minimas_ind < ind_temp].tolist()

	# Sort, the indice at the beginning will be the one we want (unless it is greater than 0)
	minimas_less.sort(reverse=True)

	flag = 0
	j = 0
	while flag == 0:
		if ax_filt_r[minimas_less[j]] < 0:
			ind_HS.append(minimas_less[j])
			flag = 1
		else:
			j += 1

# For each maxima, find the maxima after it. This will be toe off
ind_TO = []

for i in range(len(sig_maxs_ind)):
	ind_temp = sig_maxs_ind[i]

	# Get all maxs after the current indice
	maximas_greater = maximas_ind[maximas_ind > ind_temp].tolist()

	# Sort, the indice at the beginning will be the one we want (unless it is greater than 0)
	maximas_greater.sort()

	flag = 0
	j = 0
	while flag == 0:
		if ax_filt_r[maximas_greater[j]] > 0:
			ind_TO.append(maximas_greater[j])
			flag = 1
		else:
			j += 1
	
plt.plot(time, ax_filt_r,'k')
plt.plot(time[ind_HS], ax_filt_r[ind_HS], 'ob', label='heel strike')
plt.plot(time[ind_TO], ax_filt_r[ind_TO], 'og', label='toe off')

plt.xlabel('Time (s)')
plt.ylabel('Acceleration (mm/s^2)')
plt.legend()
plt.show()

'''