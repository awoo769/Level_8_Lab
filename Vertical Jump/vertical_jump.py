import numpy
from scipy import signal
import matplotlib.pyplot as plt

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
acc = 

# Take the y component (vertical direction) and time
y_acc = 
time = 

# Filter data
frequency = 500 # Arbitrary for now
b, a = signal.butter(4, 5/(frequency/2), 'low')
filt_acc = signal.filtfilt(b, a, y_acc, axis=0)

# Plot data for demonstration purposes
plt.plot(t, filt_acc, label = 'vertical acceleration')
plt.legend()
plt.xlabel('time (s)')
plt.ylabel('vertical acceleration (m/s^2)')

plt.show()

# Get take off and landing time
# Create a mask, 1 if greater than 0 and 1 if less (or equal)
mask = filt_acc > 0

tol = 1e-6

# Get index when acceleration is 0
t0 = np.where(abs(filt_acc) < tol)

take_off = 0
landing = 0
for t in t0:
	if landing == 0:
		if take_off == 0:
			# Take off occurs the first instance where acceleration goes from negative to positive
			if mask[t - 1] == 0 and mask[t + 1] == 1:
				start = time[t]
				take_off = 1 # Set take off flag

		elif take_off == 1:
			# Landing occurs the next time acceleration crosses 0
			stop = t
			landing = 1

flight_time = stop - start

g = 9.81
max_height = (g * t^2) / 8
