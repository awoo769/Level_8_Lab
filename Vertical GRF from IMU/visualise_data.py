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

	# Filter data at 5 Hz
	frequency = 50
	cut_off = 5 # Low-pass cut-off frequency set to 5 Hz, as used by Grainger et al (2019)
	b, a = signal.butter(4, cut_off/(frequency/2), 'low')

	a_filt = signal.filtfilt(b, a, a_y_ankle_l)

	label = (np.array(read_file)[:,-1]).tolist()

	# Jogging = L10, running = L11, data is recorded at 50 Hz. 1 both jogging and running were performed for 1 minute each
	jogging_start = label.index('10')
	running_start = label.index('11')
	jogging_end = running_start - 1
	running_end = label.index('12') - 1

	standing_still_start = label.index('1')
	standing_still_end = label.index('2') - 1

	#plt.plot(a_x_ankle_l,'r', label='x')
	plt.plot(a_filt[standing_still_start:standing_still_end+1],'g', label='y')
	#plt.plot(a_z_ankle_l,'b', label='z')
	
	plt.legend()
	plt.show()