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

# Read in file
data_directory = 'C:\\Users\\alexw\\Desktop\\RunningData\\0128run1.csv'

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

''' Figure out which direction IMU is facing and put in the correct direction if not what is expected'''
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

Fx_filt = signal.filtfilt(b, a, grf_x)
Fy_filt = signal.filtfilt(b, a, grf_y)
Fz_filt = signal.filtfilt(b, a, grf_z)
R_F = np.sqrt(np.power(Fx_filt, 2) + np.power(Fy_filt, 2) + np.power(Fz_filt, 2))

# Flip vertical forces as z (vertical axis in force plate coordinate system) is negative - this is just to get contact points.
# Will not be used in final version.
Fz_filt = -Fz_filt
grf_z = -grf_z

# Find when the vertical forces drop below threshold, and then make all of the forces 0.0 at these points.
force_threshold = 20

# Find the indices where the vertical force is below our threshold
force_zero = np.where(Fz_filt < force_threshold)

# Set these values to 0
Fz_filt[force_zero] = 0.0
R_F[force_zero] = 0.0


heel_strike = []
toe_off = []
# Get the points where there is force applied to the force plate (stance phase). Beginning = HS, end = TO
for i in range(1, len(Fz_filt)-1):
	if Fz_filt[i-1] == 0 and Fz_filt[i] != 0:
		heel_strike.append(i-1)
	
	if Fz_filt[i+1] == 0 and Fz_filt[i] != 0:
		toe_off.append(i+1)
'''
fig, ax1 = plt.subplots()
ax1.plot(time, ay_filt_l,'b', label='Left ankle')

#ax1.plot(time[heel_strike], ay_filt_l[heel_strike], 'og', label='heel strike')
#ax1.plot(time[toe_off], ay_filt_l[toe_off], 'or', label='toe off')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('y acceleration (mm/s^2)')
ax1.legend()

#ax2 = ax1.twinx()
#ax2.plot(time,Fz_filt,'k', label='Fz')
#ax2.set_ylabel('Vertical force (N)')

fig.tight_layout()
'''
# Find all maximas and minimas of y acceleration

# All minimas
minimas_ind = np.where(np.r_[True, ay_filt_l[1:] < ay_filt_l[:-1]] & np.r_[ay_filt_l[:-1] < ay_filt_l[1:], True] == True)[0]

#All maximas
maximas_ind = np.where(np.r_[True, ay_filt_l[1:] > ay_filt_l[:-1]] & np.r_[ay_filt_l[:-1] > ay_filt_l[1:], True] == True)[0]

# We only care about the maxima's after the HS event, so sort through and pull out the significant ones
sig_maxs_ind = []

# Initial sort by value of the maxima. Due to drift, find a maximum to compare to every 2 s (2000 data points)
previous_step = 0
maximas_ind = np.array(maximas_ind)
for step in range(2000,len(ay_filt_l), 2000):
	# Start of the section
	ind_low = maximas_ind[np.where(maximas_ind > previous_step)[0][0]]
	# End of the section
	ind_high = maximas_ind[np.where(maximas_ind < step)[0][-1]]
	# Maximum in the 2 s section
	max_maxima = max(ay_filt_l[ind_low:ind_high+1])

	# List holding the indicies of the maximums within the 2 s section
	temp = maximas_ind[np.where(maximas_ind == ind_low)[0][0]:np.where(maximas_ind == ind_high)[0][0] + 1].tolist()

	# Location of the maximum within the section
	max_location = np.where(ay_filt_l == max_maxima)[0][0]
	# Remove the max from the temp list and append it to the significant maxima's list
	temp.remove(max_location)
	sig_maxs_ind.append(max_location)

	# Sort through each maxima and accept only those which are high enough
	for i in range(len(temp)):
		maxima_value = ay_filt_l[temp[i]]
		delta = max_maxima / maxima_value
		# All accepted maximas should not be less than 1/3 of the highest within the 2 second section
		if delta <= (1/(1 - 1/3)) and maxima_value > 0:
			sig_maxs_ind.append(temp[i])

	# Loop again, with the previous step being changed to the current step from this iteration
	previous_step = step

# Check that all values are being encompased (including the end)
if previous_step < len(ay_filt_l):
	step = len(ay_filt_l)
	# Start of the final section
	ind_low = maximas_ind[np.where(maximas_ind > previous_step)[0][0]]
	# End of the final section
	ind_high = maximas_ind[np.where(maximas_ind < step)[0][-1]]
	# Maximum in the final section
	max_maxima = max(ay_filt_l[ind_low:ind_high+1])

	# List holding the indicies of the maximums within the final section
	temp = maximas_ind[np.where(maximas_ind == ind_low)[0][0]:np.where(maximas_ind == ind_high)[0][0] + 1].tolist()

	# Location of the maximum within the section
	max_location = np.where(ay_filt_l == max_maxima)[0][0]
	# Remove the max from the temp list and append it to the significant maxima's list
	temp.remove(max_location)
	sig_maxs_ind.append(max_location)

	# Sort through each maxima and accept only those which are high enough
	for i in range(len(temp)):
		maxima_value = ay_filt_l[temp[i]]
		delta = max_maxima / maxima_value
		# All accepted maximas should not be less than 1/3 of the highest within the final section
		if delta <= (1/(2/3)) and maxima_value > 0:
			sig_maxs_ind.append(temp[i])

# First sort is complete. Now we will sort through the remainder based on spacing apart, gradiant, and peak-to-trough distance

# Copy the signiciant maxima indices and wipe the list
temp_sig = sig_maxs_ind.copy()
sig_maxs_ind = []

# A list of maximas that will be excluded
no_add = []

# There should at least be a 0.55 s (100 indices) gap between HS. If this isn't the case, the code won't automatically rule 
# the maxima out

# Make sure that the list is sorted (smallest to largest)
temp_sig.sort()
for i in range(0, len(temp_sig)):
	# Get the current maxima indice
	maxima = temp_sig[i]

	# Don't add the maxima to the list
	if maxima in no_add:
		pass

	else:
		# If this is the last in the list, and the previous maxima was accepted, accept the maxima
		if (i == len(temp_sig) - 1) and (temp_sig[i-1] in sig_maxs_ind):
			sig_maxs_ind.append(maxima)
			break

		# If this is the last in the list, but the previous maxima was not accepted
		elif (i == len(temp_sig) - 1) and (temp_sig[i-1] not in sig_maxs_ind):
			# Sometimes there are maxima which we don't want, but they are high enough to be counted. 
			# Check the gradient near the top
			for maxima in temp_sig:
				# go back 0.01 s (10 indices) and take a linear gradient reading between the two
				t1, t2 = time[maxima - 10], time[maxima]
				a1, a2 = ay_filt_l[maxima - 10], ay_filt_l[maxima]

				m = (a2 - a1) / (t2 - t1)

				# If the gradient is acceptable, take the maxima
				if m > 500:
					sig_maxs_ind.append(maxima)

		# If we are at any point other than the last in the list
		elif i < len(temp_sig) - 1:
			# See if the distance between the current maxima and the next is greater than 0.55 s.
			# If it is, accept the maxima
			if temp_sig[i+1] - maxima > 550:
				sig_maxs_ind.append(maxima)
			
			# If there are two maxima's in close proximity to each other
			else:
				# Sometimes there are maxima which we don't want, but they are high enough to be counted. 
				# Check the gradient near the top
				
				# go back 0.01 s (10 indices) and take a linear gradient reading between the two
				t1, t2 = time[maxima - 10], time[maxima]
				a1, a2 = ay_filt_l[maxima - 10], ay_filt_l[maxima]

				m = (a2 - a1) / (t2 - t1)

				if m > 500:
					# Every so often maxima's which we don't want will pass this test. 
					# Check the distance between max and min of temp_sig[i+1] and temp_sig[i]. The greatest (should) be the HS.
					# Compare to the minima to the left of the maxima
					max_1 = maxima # temp_sig[i]
					max_2 = temp_sig[i+1] # Next maxima
					min_1 = minimas_ind[minimas_ind < max_1][-1] # minima to the left of the first maxima
					min_2 = minimas_ind[minimas_ind < max_2][-1] # minima to the left of the second maxima

					# Distance between the trough and peak
					dis_1 = ay_filt_l[max_1] - ay_filt_l[min_1]
					dis_2 = ay_filt_l[max_2] - ay_filt_l[min_2]

					# If the first distance is greater than the second, the 1st maxima is the one we want
					if dis_1 >= dis_2:					
						sig_maxs_ind.append(maxima)
						# Exclude the second maxima
						no_add.append(temp_sig[i+1])

# Sometimes a maxima that we don't want will slip through the code, but it will be caught in the no_add list
for maxima in sig_maxs_ind:
	if maxima in no_add:
		# Remove it. I don't know how it slipped through, but alas, this is the world we live in.
		sig_maxs_ind.remove(maxima)

# The HS will be at the minima just before a maxima.
HS = []

for maxima in sig_maxs_ind:
	# Check that the minima to the left of the maxima is true, and not just a small "blimp"
	temp_min = minimas_ind[minimas_ind < maxima][-1]
	check_ind1 = temp_min - 35 # Main check
	check_ind2 = temp_min - 20 # Fine detail

	# Make sure that the check indices are not negative
	if check_ind1 < 0:
		check_ind1 = 0

		if check_ind2 < 0:
			check_ind2 = 0

	# Flag for when HS has been found
	HS_flag = 0

	# Initialise counter
	i = 0
	# Find the minima just before the maxima (HS)
	while HS_flag == 0:

		if (ay_filt_l[check_ind1] > ay_filt_l[temp_min]) and not (ay_filt_l[check_ind2] < ay_filt_l[temp_min]):
			HS.append(temp_min)

			# HS has been found
			HS_flag = 1
		
		# It was a "blimp". Move to the next minima to the left and readjust checks
		else:
			temp_min = minimas_ind[minimas_ind < maxima][i-2]
			check_ind1 = temp_min - 35
			check_ind2 = temp_min - 20

			# HS has not been found
			HS_flag = 0

			# Increase counter
			i += 1

plt.plot(time, ay_filt_l)
plt.plot(time[HS], ay_filt_l[HS], 'or')
plt.show()


"""
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

ymax = max(max((Rxz_ankle_l + R_ankle_l)/2), max((Rxz_ankle_r + R_ankle_r)/2))
ax1.vlines(x=time[heel_strike], ymin=0, ymax=ymax, colors='g', label='Heel strike')
ax1.vlines(x=time[toe_off], ymin=0, ymax=ymax, colors='y', label='Toe off')

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

"""