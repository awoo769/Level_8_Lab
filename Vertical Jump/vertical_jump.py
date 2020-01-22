import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import filedialog
import csv

import os

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

# Filter data
frequency = 500 # Arbitrary for now
cut_off = 5
b, a = signal.butter(4, cut_off/(frequency/2), 'low')
filt_acc = signal.filtfilt(b, a, y_acc)

# Rotate data and remove effect of gravity

# Subject will be still for first part of trial, first 250 data points should do (0.5 s)
g = np.mean(filt_acc[:251]) # This value should be near 9.81 (assuming movement only in y direction)

# Depending on calibration of sensor/rotation, positive acceleration may be positive or negative
if g > 0:
	filt_acc = -(filt_acc - g) # Remove effect of gravity and flip
else:
	filt_acc = filt_acc - g

# Plot data for demonstration purposes
plt.plot(time, filt_acc, label = 'vertical acceleration')
plt.legend()
plt.xlabel('time (s)')
plt.ylabel('vertical acceleration (m/s^2)')
plt.title('Entire trial')

plt.show()

# Get minima points, for each jump there should be 2 siginficant peaks per jump
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

n_jumps_est = int(np.floor(len(sig_mins) / 2))

# Get number of jumps from user - potentially find this out automatically
passed = 0

if passed == 0:
	query = input('Did the person jump ' + str(n_jumps_est) + ' times? [Y/N] ')

	if query == 'Y' or 'y' or 'yes' or 'Yes':
		n_jumps = n_jumps_est
		passed = 1
	elif query == 'N' or 'n' or 'no' or 'No':
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

	g = 9.81
	max_height = (g * flight_time * flight_time) / 8

	print('Jump ' + str(i+1) + ' height = ' + str(np.round(max_height * 100, 2)) + ' cm.')
