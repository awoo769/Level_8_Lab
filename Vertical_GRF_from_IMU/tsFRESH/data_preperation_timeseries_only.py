'''
This script prepares acceleration data from ankle worn IMU's to find HS and TO events using a machine
learning process.

The IMU's should be placed on the medial aspect of the tibia (on each leg).

Left coordinate system: y = up, z = towards midline, x = forward direction
Right coordinate system: y = up, z = towards midline, x = backward direction

- assuming you are using a unit from IMeasureU, the little man should be facing upwards and
be on the medial side of each ankle

During this function, acceleration and force plate data will be interpolated to be at 1000 Hz

05/03/2020
Alex Woodall

'''

# For input type purposes
import numpy as np

def prepare_data(GRF_data: np.ndarray, IMU_data: np.ndarray, sample_length: int) -> (np.ndarray, np.ndarray, np.ndarray):
	'''
	This function creates the dataset of events in which features will be extracted from

	data: the data which will be split into samples of length sample_length
	dataset: the array which is being build of the samples from each trial
	HS_TO: a list of the truth values of the FS and FO events
	overlap: whether to overlap each sample by half

	returns three numpy arrays: acceleration, force and event

	'''

	from scipy import signal
	import numpy as np

	from utils import interpolate_data, rezero_filter


	# Localise functions for speed improvements
	normal = np.linalg.norm
	butter = signal.butter
	filtfilt = signal.filtfilt

	# Frequency to interpolate data to
	analog_frequency = 1000
	interpolate_frequency = 200

	# Convert data to the correct shape
	if int(GRF_data.shape[1] > GRF_data.shape[0]) == 0:
		GRF_data = GRF_data.T
		IMU_data = IMU_data.T

	F = GRF_data[1:,:]
	GRF_time = GRF_data[0,:]

	# Rotate 180 deg around y axis (inverse Fx and Fz) - assuming that z is facing down
	F[0] = -F[0] # Fx
	F[2] = -F[2] # Fz

	''' Filter force plate data at 60 Hz '''
	cut_off = 60 # Derie (2017), Robberechts et al (2019)
	order = 2 # Weyand (2017), Robberechts et al (2019)
	b, a = butter(N=order, Wn=cut_off/(analog_frequency/2), btype='low')

	new_F = filtfilt(b, a, F, axis=1)

	''' Rezero filtered forces '''
	threshold = 50 # 60 N
	filter_plate = rezero_filter(original_fz=new_F[2], threshold=threshold, frequency=analog_frequency)

	# Re-zero the filtered GRFs
	new_F = new_F * filter_plate
	new_F[new_F < 0] = 0

	from matplotlib import pyplot as plt
	plt.plot(new_F[2])
	plt.show()

	# Interpolate GRF
	time, new_F = interpolate_data(GRF_time, new_F, interpolate_frequency)

	# Re-zero after interpolating
	filter_plate = rezero_filter(original_fz=new_F[2], threshold=threshold, frequency=interpolate_frequency)
	new_F = new_F * filter_plate

	from matplotlib import pyplot as plt
	plt.plot(new_F[2])
	plt.show()

	# Filter and interpolate acceleration data
	''' Filter acceleration data at 0.8 Hz and 45 Hz (band-pass) '''
	cut_off_l = 0.8 # Derie (2017), Robberechts et al (2019)
	cut_off_h = 45 # Derie (2017), Robberechts et al (2019)
	order = 2 # Weyand (2017), Robberechts et al (2019)
	b, a = signal.butter(N=order, Wn=[cut_off_l/(analog_frequency/2), cut_off_h/(analog_frequency/2)], btype='band')

	IMU = IMU_data[1:,:]
	IMU_time = IMU_data[0,:]

	new_IMU = filtfilt(b, a, IMU, axis=1)
	time, new_IMU = interpolate_data(IMU_time, new_IMU, interpolate_frequency)

	a_r = new_IMU[:3]
	a_l = new_IMU[3:]

	from matplotlib import pyplot as plt
	plt.plot(a_r[1])
	plt.show()

	# Engineered timeseries
	a_diff = abs(a_l - a_r) # Difference between left and right
	a_res_l = normal(a_l, axis=0) # Left resultant
	a_res_r = normal(a_r, axis=0) # Right resultant
	a_res_diff = abs(a_res_l - a_res_r) # Difference between left and right resultant

	force = new_IMU.T

	uid = np.linspace(0, len(time) - 1, len(time))
	accelerations = np.vstack((uid, time, a_l, a_r, a_diff, a_res_l, a_res_r, a_res_diff)).T

	return accelerations, force


def create_dataset(dataset_dict: dict, sample_length: int, f: str) -> dict:
	'''
	This function is calls the prepare_data function. It will gather the output
	of the prepare_data and use it to create the truth values to be used to predict
	events and timeseries.

	08/05/2020
	Alex Woodall

	'''

	import numpy as np

	# Import required functions
	from utils import read_csv, get_runner_info

	# Localise functions for speed improvements
	isnan = np.isnan

	# Load the data
	GRF_data, IMU_data = read_csv(f)

	# Sort the data into samples and pre-process data for analysis
	X, y = prepare_data(GRF_data=GRF_data, IMU_data=IMU_data, sample_length=sample_length)
	
	# Get the name of the trial and use it as the dictionary key
	ID = f.split('.')[0]
	ID = ID.split('\\')[-1]

	# Get mass of trial
	if 'ITL' in ID:
		runner_ID = ID.split('ITL')[0] + 'a'
	else:
		runner_ID = ID[:8]
	info = get_runner_info('C:\\Users\\alexw\\Dropbox\\auckIMU\\demos.xlsx')

	iloc = np.where(info['Subject_ID'] == runner_ID)[0][0]
	mass = float(info['Mass'].iloc[iloc])

	mass_all = [mass] * len(X)

	if not isnan(mass):
		dataset_dict[ID] = {}

		# Save to the dataset dictionary
		dataset_dict[ID]['X'] = X
		dataset_dict[ID]['X_mass_all'] = mass_all

		dataset_dict[ID]['y'] = y

	return dataset_dict


def main():

	from glob import glob
	import numpy as np
	import pickle
	import os

	from utils import get_runner_info, sort_strike_pattern

	# Localise functions for speed improvements
	dump = pickle.dump

	# Select path and read all .csv files (these will be the trial data)
	#file_path = 'C:\\Users\\alexw\\Dropbox\\auckIMU\\SNRCdat_default\\'
	file_path = 'C:\\Users\\alexw\\Dropbox\\auckIMU\\SNRCrcp\\'
	info_path = 'C:\\Users\\alexw\\Dropbox\\auckIMU\\demos.xlsx'
	ext = '*.csv'

	all_csv_files = [file
					for path, subdir, files in os.walk(file_path)
					for file in glob(os.path.join(path, ext))]
	
	# If there is an 02, 03... etc, then don't use the 01 (or versions previous) (something may have gone wrong)
	prev = ''
	remove_index = []
	for i in range(len(all_csv_files)):
		cur = all_csv_files[i]

		if cur[:-6] == prev[:-6]:
			# Remove previous
			remove_index.append(i-1)
		prev = cur

	i = 0
	for element in remove_index:
		all_csv_files.pop(element - i)

		i += 1
		
	runner_info = get_runner_info(info_path)
	RFS, MFS, FFS, Mixed = sort_strike_pattern(runner_info)

	# Length of each sample = 100 ms
	length = 100

	# Dictionary to hold trial data and truth solutions
	dataset = {}

	for f in all_csv_files:
		# Get runners ID
		ID = f.split('\\')[-1]
		if 'ITL' in ID:
			ID = ID.split('ITL')[0] + 'a'
		else:
			ID = ID[:8]

		if ID in RFS:
			print('Running file: ' + str(f))
			dataset = create_dataset(dataset, length, f)

	# Save dataset
	dataset_folder = "C:\\Users\\alexw\\Desktop\\Harvard_data\\"

	f = open(dataset_folder + "dataset_200.pkl", "wb")

	dump(dataset, f)
	f.close()


if __name__ == '__main__':
	main()
	