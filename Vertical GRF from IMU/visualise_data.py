import numpy as np
from matplotlib import pyplot as plt
import os
from scipy import signal

# Read in file
data_directory = 'C:\\Users\\alexw\\Dropbox\\ABI\\Level_8_Lab\\Vertical GRF from IMU\\MHEALTHDATASET'

data_files = [f for f in os.listdir(data_directory) if (os.path.isfile(os.path.join(data_directory, f)) and '.log' in f)]

for i in range(len(data_files)):
	fid = open(os.path.join(data_directory, data_files[i]), 'r')

	read_file = []
	for line in fid:
		line = line.rstrip("\r\n")
		read_file.append(line.split('\t'))

	a_x_ankle_l = (np.array(read_file)[:,5]).astype(np.float) # 6th column
	a_y_ankle_l = (np.array(read_file)[:,6]).astype(np.float) # 7th column
	a_z_ankle_l = (np.array(read_file)[:,7]).astype(np.float) # 8th column

	R_ankle_l = np.sqrt(np.power(a_x_ankle_l, 2) + np.power(a_y_ankle_l, 2) + np.power(a_z_ankle_l, 2))

	# Filter data at 5 Hz
	frequency = 50
	cut_off = 10 # Running occurs in the 10 Hz to 30 Hz range
	b, a = signal.butter(4, cut_off/(frequency/2), 'low')

	ay_filt = signal.filtfilt(b, a, a_y_ankle_l)
	ax_filt = signal.filtfilt(b, a, a_x_ankle_l)
	az_filt = signal.filtfilt(b, a, a_z_ankle_l)
	R_filt = signal.filtfilt(b, a, R_ankle_l)

	label = (np.array(read_file)[:,-1]).tolist()

	standing_still_start = label.index('1')
	standing_still_end = len(label) - 1 - label[::-1].index('1')

	# Shift by average acceleration during standing still phase
	gx = np.mean(ax_filt[standing_still_start:standing_still_end+1])
	gy = np.mean(ay_filt[standing_still_start:standing_still_end+1])
	gz = np.mean(az_filt[standing_still_start:standing_still_end+1])
	gR = np.mean(R_filt[standing_still_start:standing_still_end+1])

	ax_filt = ax_filt - gx
	ay_filt = ay_filt - gy
	az_filt = az_filt - gz
	R_filt = R_filt - gR

	# Each activity in the dataset has 3071 points
	# Jogging = L10, running = L11, data is recorded at 50 Hz. 1 both jogging and running were performed for 1 minute each
	jogging_start = label.index('10')
	jogging_end = len(label) - 1 - label[::-1].index('10')

	running_start = label.index('11')
	running_end = len(label) - 1 - label[::-1].index('11')

	#plt.plot(ax_filt[running_start+1500:running_start+1601],'r', label='x')
	plt.plot(ay_filt[running_start+1500:running_start+1601],'g', label='y')
	#plt.plot(az_filt[running_start+1500:running_start+1601],'b', label='z')
	#plt.plot(R_filt[running_start+1500:running_start+1601], 'k', label='resultant')
	
	plt.legend()
	plt.show()