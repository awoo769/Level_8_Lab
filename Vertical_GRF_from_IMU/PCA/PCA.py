import numpy as np
from matplotlib import pyplot as plt
import os
from glob import glob
from sklearn.decomposition import PCA

from utils import prepare_data, read_csv, filter_acceleration, phase_split, sort_strike_pattern, get_runner_info

''' Read in file '''
# Select path and read all .csv files (these will be the trial data)
file_path = 'C:\\Users\\alexw\\Dropbox\\auckIMU\\SNRCdat_default\\'
info_path = 'C:\\Users\\alexw\\Dropbox\\auckIMU\\demos.xlsx'
ext = '*.csv'

all_csv_files = [file
				for path, subdir, files in os.walk(file_path)
				for file in glob(os.path.join(path, ext))]


# If there is an 02, then don't use the 01 (something may have gone wrong)
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

explained_variance_ratio_RFS = []
explained_variance_ratio_FFS = []

for f in all_csv_files:

	print('Running file: ' + str(f))

	GRF_data, IMU_data = read_csv(f)

	# Get runners ID
	f = f.split('\\')[-1]
	f = f.split('ITL')[0] + 'a'

	GRF_data, IMU_data, FS, FO = prepare_data(GRF_data, IMU_data)

	# Time array
	time_array = IMU_data[0]

	a_left_filt = IMU_data[4:]
	a_right_filt = IMU_data[1:4]

	# Reshape data into stances
	left_stride, right_stride = phase_split(time_array, a_left_filt, a_right_filt, FS, FO, 'stance')

	res = np.vstack((left_stride['R'], right_stride['R']))

	pca = PCA(n_components=5)

	pca.fit(res)

	explained_variance_ratio_i = pca.explained_variance_

	#plt.plot(res[0])
	#plt.bar(x = [1, 2, 3, 4, 5], height=explained_variance_ratio_i)
	#plt.show()


	# Add to lists for an	
	try:
		if f in FFS:
			explained_variance_ratio_FFS.append(explained_variance_ratio_i) 
			res_FFS = np.vstack((res_FFS, res))
		elif f in RFS:
			explained_variance_ratio_RFS.append(explained_variance_ratio_i) 
			res_RFS = np.vstack((res_RFS, res))

			x_axis = np.linspace(0, 100, 1000)

			plt.plot(x_axis, pca.components_[-1])
			plt.xlabel('Time (% stride)')
			plt.show()

		res_all = np.vstack((res_all, res))
	
	except NameError:
		if f in FFS: 
			res_FFS = res
		elif f in RFS:
			res_RFS = res

		res_all = res

# All strides
pca = PCA(n_components=5)
pca.fit(res_all)

explained_variance_ratio = pca.explained_variance_ratio_

#plt.bar(x = [1, 2, 3, 4, 5], height=explained_variance_ratio)
#plt.show()

# Front foot stride
pca = PCA(n_components=5)
pca.fit(res_FFS)

explained_variance_ratio = pca.explained_variance_ratio_

#plt.bar(x = [1, 2, 3, 4, 5], height=explained_variance_ratio)
#plt.show()

# Rear foot stride
pca = PCA(n_components=5)
pca.fit(res_RFS)

explained_variance_ratio = pca.explained_variance_ratio_

#plt.bar(x = [1, 2, 3, 4, 5], height=explained_variance_ratio)
#plt.show()

for i in range(len(explained_variance_ratio_FFS)):
	plt.plot(explained_variance_ratio_FFS[i][0], explained_variance_ratio_FFS[i][1], 'og')

for i in range(len(explained_variance_ratio_RFS)):
	plt.plot(explained_variance_ratio_RFS[i][0], explained_variance_ratio_RFS[i][1], 'or')

plt.show()

okay = 1
