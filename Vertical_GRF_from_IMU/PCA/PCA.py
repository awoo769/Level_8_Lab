import numpy as np
from matplotlib import pyplot as plt
import os
from glob import glob
from sklearn.decomposition import PCA

from utils import prepare_data, read_csv, sort_events, filter_acceleration, allocate_events, sort_strike_pattern, get_runner_info

''' Read in file '''
# Select path and read all .csv files (these will be the trial data)
file_path = 'C:\\Users\\alexw\\Dropbox\\auckIMU\\SNRCdat_default\\'
info_path = 'C:\\Users\\alexw\\Dropbox\\auckIMU\\demos.xlsx'
ext = '*.csv'

all_csv_files = [file
				for path, subdir, files in os.walk(file_path)
				for file in glob(os.path.join(path, ext))]

runner_info = get_runner_info(info_path)
RFS, MFS, FFS, Mixed = sort_strike_pattern(runner_info)

# Length of each sample = 600 ms
length = 100

explained_variance = []
explained_variance_ratio = []

for f in all_csv_files:

	print('Running file: ' + str(f))

	GRF_data, IMU_data = read_csv(f)

	f = f.split('.')[0]

	GRF_data, IMU_data, FS, FO = prepare_data(GRF_data, IMU_data)
	
	FS_new, FO_new = sort_events(FS, FO, first_event='FS', final_event='FS')

	# Time array
	time_array = IMU_data[0]

	a_left_filt = IMU_data[4:]
	a_right_filt = IMU_data[1:4]

	# Reshape data into strides
	left_stride, right_stride = allocate_events(time_array, a_left_filt, a_right_filt, FS, FO)

	res = np.vstack((left_stride['R'], right_stride['R']))

	try:
		res_all = np.vstack((res_all, res))
	
	except NameError:
		res_all = res

	pca = PCA()

	pca.fit(res)
	
	explained_variance.append(pca.explained_variance_[0])
	explained_variance_ratio.append(pca.explained_variance_ratio_[0])

	print('Explained variance of first component: {}'.format(explained_variance[-1]))
	print('Explained variance ratio of first component:: {}'.format(explained_variance_ratio[-1]))

	plt.plot(res[0])
	plt.show()

print('Mean explained variance of first component:e: {}'.format(np.mean(explained_variance)))
print('Mean explained variance ratio of first component:: {}'.format(np.mean(explained_variance_ratio)))

pca = PCA()

pca.fit(res_all)

explained_variance_all = pca.explained_variance_[0]
explained_variance_ratio_all = pca.explained_variance_ratio_[0]

print('Explained variance: {}'.format(explained_variance_all))
print('Explained variance ratio: {}'.format(explained_variance_ratio_all))
