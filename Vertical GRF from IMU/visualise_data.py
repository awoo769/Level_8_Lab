import numpy as np
from matplotlib import pyplot as plt
import os
from scipy import signal
import csv

# Read in file
data_directory = 'C:\\Users\\alexw\\Dropbox\\ABI\\Level_8_Lab\\Vertical GRF from IMU\\Run30.csv'

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
# Take data from the right

time = (np.array(read_file)[:,0]).astype(np.float) # 1st column
a_x_ankle = (np.array(read_file)[:,4]).astype(np.float) # 5th column
a_y_ankle = (np.array(read_file)[:,5]).astype(np.float) # 6th column
a_z_ankle = (np.array(read_file)[:,6]).astype(np.float) # 7th column

# Also take force plate data for comparison
grf_x = (np.array(read_file)[:,1]).astype(np.float) # 2nd column
grf_y = (np.array(read_file)[:,2]).astype(np.float) # 3rd column
grf_z = (np.array(read_file)[:,3]).astype(np.float) # 4th column

# Filter data at 5 Hz
frequency = 1000
cut_off = 30 # Running occurs in the 10 Hz to 30 Hz range
b, a = signal.butter(4, cut_off/(frequency/2), 'low')

ax_filt = signal.filtfilt(b, a, a_x_ankle)
ay_filt = signal.filtfilt(b, a, a_y_ankle)
az_filt = signal.filtfilt(b, a, a_z_ankle)
R_ankle = np.sqrt(np.power(ax_filt, 2) + np.power(ay_filt, 2) + np.power(az_filt, 2))

Fx_filt = signal.filtfilt(b, a, grf_x)
Fy_filt = signal.filtfilt(b, a, grf_y)
Fz_filt = signal.filtfilt(b, a, grf_z)
R_F = np.sqrt(np.power(Fx_filt, 2) + np.power(Fy_filt, 2) + np.power(Fz_filt, 2))

# Flip vertical forces
Fz_filt = -Fz_filt

# Find when the vertical forces drop below threshold, and then make all of the forces and CoP values 0.0 at these points.
force_threshold = 20 # Set this to 20 N

# Find the indices where the vertical force is below our threshold
force_zero = np.where(Fz_filt < force_threshold)

# Set these values to 0
Fx_filt[force_zero] = 0.0
Fy_filt[force_zero] = 0.0
Fz_filt[force_zero] = 0.0
R_F[force_zero] = 0.0

plt.plot(time, ax_filt,'r', label='x')
plt.plot(time, ay_filt,'g', label='y')
plt.plot(time, az_filt,'b', label='z')
#plt.plot(time,R_F,'r',label='reactant')
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.legend()
plt.show()