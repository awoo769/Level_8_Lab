import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import filedialog
import csv

import os

from scipy import integrate

'''
This script will calculate the vertical jump height from data gathered by an IMU.

There are multiple methods to calculate the vertical jump height without the use of a force plate.

1) Time of Flight
2) Motion capture
3) Double integration of acceleration data

Time of Flight

Filter acceleration data - 5 Hz Butterworth low pass filter
Using acceleration data, one can calculate the time of flight. This can then be used to calculate the maximum height:
h = gT^2/8, where h is maximum jump height, g is the acceleration due to gravity and T is the time of flight.
Grainger et al (2019) have shown this method to have an R^2 value of 0.94 when compared to optical motion capture for
a vertical jump from the ground. Only acceleration in the y-axis (vertical axis) was analysed

The take-off time is the time when the acceleration first crossed 0. Landing time is the second time acceleration 
crossed 0.

IMU measurements most often underestimate the actual maximum jump height.

When integrating acceleration data to displacement, care must be taken to filter the data to minimise numerical drift
(which will occur). We can use certain points - such as a known initial height of centre of mass (CoM), can't go below 
0 etc.

Author: Alex Woodall
Auckland Bioengineering Institute
Date: 20/01/2020

'''

# Import data
root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename(initialdir = os.getcwd(),title = "Select file",filetypes = (("csv files","*.csv"),("all files","*.*")))

acc = []
with open(file_path, newline='') as csvfile:
	reader = csv.reader(csvfile)
	for row in reader:
		acc.append(row)
		
acc = np.array(acc)

# Take the y component (vertical direction) and time (convert to float), assume movement only in the y direction
y_acc = acc[1:,2].astype(np.float)
time = acc[1:,0].astype(np.float)

# Trim the first second of recording (ASSUME STILL FOR OVER 1 SECOND)

y_acc = y_acc[500:]
time = time[500:]

# Filter data
frequency = 500
cut_off = 5 # Low-pass cut-off frequency set to 5 Hz, as used by Grainger et al (2019)
b, a = signal.butter(4, cut_off/(frequency/2), 'low')

a_filt = signal.filtfilt(b, a, y_acc)

# Rotate data and remove effect of gravity

# Subject will be still for first part of trial, first 250 data points should do (0.5 s)
g = np.mean(a_filt[:251]) # This value should be near 9.81 (assuming movement only in y direction)

# Depending on calibration of sensor/rotation, positive acceleration may be positive or negative
if g > 0:
	filt_acc = -(a_filt - g) # Remove effect of gravity and flip
else:
	filt_acc = a_filt - g

# Plot data for demonstration purposes
plt.plot(time, filt_acc, label = 'vertical acceleration')
plt.legend()
plt.xlabel('time (s)')
plt.ylabel('vertical acceleration (m/s^2)')
plt.title('Entire trial')

plt.show()

# Get minima points, for each jump there should be 2 siginficant peaks
minima_ind = np.where(np.r_[True, filt_acc[1:] < filt_acc[:-1]] & np.r_[filt_acc[:-1] < filt_acc[1:], True] == True)
minima_acc = filt_acc[minima_ind].tolist()

# Sort in decending order (largest first)
minima_acc.sort()

temp = minima_acc.copy()

min_peak = min(temp)

sig_mins = []
sig_mins.append(min_peak)
temp.remove(min_peak)

for i in range(len(temp)):
	delta = min_peak / temp[i]
	if delta <= 4 and temp[i] < 0:
		sig_mins.append(temp[i])

# Each min point should be separated by a positive value (a jump or a rise at the end of a jump)
final_min_points_ind = []

# Get indices of sig_mins
sig_mins_ind = []
for i in range(len(sig_mins)):
	sig_mins_ind.append(np.where(filt_acc == sig_mins[i])[0][0])

sig_mins_ind.sort()

tol = 0.2

for i in range(len(sig_mins)):
	if i == len(sig_mins) - 2:
		if np.size(np.where(filt_acc[sig_mins_ind[i]:] > -tol)) > 0:
			final_min_points_ind.append(sig_mins_ind[i])

	elif i != len(sig_mins) - 1: # Not on the last minimum
		# Test if there is a positive number between the two mins
		if np.size(np.where(filt_acc[sig_mins_ind[i]:sig_mins_ind[i+1] + 1] > -tol)) > 0:
			final_min_points_ind.append(sig_mins_ind[i])
	else:
		if np.size(np.where(filt_acc[sig_mins_ind[i-1]:sig_mins_ind[i] + 1] > -tol)) > 0:
			final_min_points_ind.append(sig_mins_ind[i])

n_jumps_est = int(np.floor(len(final_min_points_ind) / 2))

plt.plot(time, filt_acc)
plt.plot(time[final_min_points_ind], filt_acc[final_min_points_ind],'bo')
plt.show()

# Check if number of jumps estimate is correct with user
passed = 0

if passed == 0:
	if n_jumps_est == 1:
		query = input('Did the person jump ' + str(n_jumps_est) + ' time? [Y/N] ')
	else:
		query = input('Did the person jump ' + str(n_jumps_est) + ' times? [Y/N] ')

	if query in ['Y', 'y', 'yes', 'Yes']:
		n_jumps = n_jumps_est
		passed = 1
	elif query in ['N', 'n', 'no', 'No']:
		n_jumps = input('How many times did the person jump? ')
		try: 
			n_jumps = int(n_jumps)
		except:
			n_jumps = input('How many times did the person jump? Please enter a numerical digit. ')
			n_jumps = int(n_jumps)
		passed = 1
	else:
		query = input('Please enter Y or N. Did the person jump ' + str(n_jumps_est) + ' times? [Y/N] ')

n_peaks = 2 * n_jumps
# First n_peaks are the important peaks
minima_jumps = minima_acc[:n_peaks]


''' Time of flight method '''
print('Using time of flight method.')

# Get indicies of these peaks
ind = []
for i in range(n_peaks):
	ind.append(np.where(filt_acc == minima_jumps[i])[0][0])

# Overlay plot
# Plot data for demonstration purposes
plt.plot(time, filt_acc, label = 'vertical acceleration')
plt.plot(time[ind], filt_acc[ind], 'bo', label = 'minima')
plt.legend()
plt.xlabel('time (s)')
plt.ylabel('vertical acceleration (m/s^2)')
plt.title('Entire trial')

plt.show()

# Two peaks per jump, so get the first peak of each jump.
ind.sort()

# First peak is every even indice (0, 2, 4... etc)
first_peaks = ind[::2]
second_peaks = ind[1::2]

take_off_list = []
touch_down_list = []

for i in range(n_jumps):
	''' Get take off and landing time '''

	# Take-off occurs when acceleration crosses 0 line after the first peak (first positive value)
	acc_temp = filt_acc[first_peaks[i]:]
	first_pos = (np.where(acc_temp > 0)[0])[0]

	# Evaluate the value of this to see if the next number is closer to 0
	if abs(acc_temp[first_pos]) < abs(acc_temp[first_pos - 1]):
		take_off_ind = first_pos + first_peaks[i]
	else:
		take_off_ind = first_pos - 1 + first_peaks[i]
	
	# Touch-down occurs when acceleration crosses 0 line before the last peak (last positive value)
	acc_temp = filt_acc[:second_peaks[i]].tolist()
	first_pos = next(j for j in reversed(range(len(acc_temp))) if acc_temp[j] > 0)

	# Evaluate the value of this to see if the next number is closer to 0
	if abs(acc_temp[first_pos]) < abs(acc_temp[first_pos + 1]):
		touch_down_ind = first_pos
	else:
		touch_down_ind = first_pos + 1

	# Overlay plot
	# Plot data for demonstration purposes
	plt.plot(time, filt_acc, label = 'vertical acceleration')
	plt.plot(time[take_off_ind], filt_acc[take_off_ind], 'bo', label = 'Take-off')
	plt.plot(time[touch_down_ind], filt_acc[touch_down_ind], 'ro', label = 'Touch-down')
	plt.legend()
	plt.xlabel('time (s)')
	plt.ylabel('vertical acceleration (m/s^2)')
	plt.title('Entire trial')

	plt.show()

	flight_time = time[touch_down_ind] - time[take_off_ind]

	take_off_list.append(take_off_ind)
	touch_down_list.append(touch_down_ind)

	g = 9.81
	max_height = (g * flight_time * flight_time) / 8

	print('Jump ' + str(i+1) + ' height = ' + str(np.round(max_height * 100, 2)) + ' cm.')

''' Integration method '''
print('Using integration method.')

# This method involves double integrating the acceleration data to get displacement data.
# Double integration will cause drift/numerical error

# There are multiple methods to reduce drift.
# 1) Use implicit integration (such as the trapezium rule)
# 2) Use the "integration in two reference frames" concept - not implemented, yet.
# 3) High-pass filter the integrated outputs. Luo et al (2013) used a high-pass filter with a cut-off frequency of
# 0.5 Hz when removing drift from ECG signals. The acceleration data has already been low-pass filtered at 5 Hz

# Re-zeroed acceleration data/and filtered
a_filt = -(filt_acc.copy())

# Filter data to remove drift
frequency = 500
cut_off = 0.5 # Play around with this.
d, c = signal.butter(N=4, Wn=cut_off/(frequency/2), btype='high')

# Integrate to get velocity
v = integrate.cumtrapz(y=a_filt, x=time, initial=0)
v_filt = signal.filtfilt(b=d, a=c, x=v)

# Integrate to get displacement
s = integrate.cumtrapz(y=v_filt, x=time, initial=0)

s_filt = signal.filtfilt(b=d, a=c, x=s)

jump_height = []
rough_mid_ind = []

# Get maximum height. Since centre of mass is zeroed, the maximum height will be the jump height.
for i in range(n_jumps):
	jump_height.append(max(s_filt[take_off_list[i]:touch_down_list[i]]))

	print("Jump %d height = %0.2f cm" % (i+1, jump_height[i] * 100))

	rough_mid_ind.append(int((touch_down_list[i] - take_off_list[i]) / 2 + take_off_list[i]))

ax1 = plt.subplot(311)
plt.plot(time, a_filt, 'r', label = 'vertical acceleration')
plt.plot(time[take_off_list], a_filt[take_off_list],'o')
plt.plot(time[touch_down_list], a_filt[touch_down_list],'o')
plt.plot(time[rough_mid_ind], a_filt[rough_mid_ind],'o')
plt.ylabel('vertical acceleration (m/s^2)')
plt.setp(ax1.get_xticklabels(), visible=False)

ax2 = plt.subplot(312, sharex=ax1)
plt.plot(time, v_filt, 'g', label = 'vertical velocity')
plt.plot(time[take_off_list], v_filt[take_off_list],'o')
plt.plot(time[touch_down_list], v_filt[touch_down_list],'o')
plt.plot(time[rough_mid_ind], v_filt[rough_mid_ind],'o')

plt.ylabel('vertical velocity (m/s)')
plt.setp(ax2.get_xticklabels(), visible=False)

ax3 = plt.subplot(313, sharex=ax1)
plt.plot(time, s_filt, 'b', label = 'vertical displacement')
plt.plot(time[take_off_list], s_filt[take_off_list],'o')
plt.plot(time[touch_down_list], s_filt[touch_down_list],'o')
plt.plot(time[rough_mid_ind], s_filt[rough_mid_ind],'o')

plt.ylabel('vertical displacement (cm)')

plt.xlabel('time (s)')

plt.show()
