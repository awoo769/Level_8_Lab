import numpy as np
from matplotlib import pyplot as plt
import os
import glob
from sklearn.decomposition import PCA

from utils import prepare_data, read_csv, sort_events, filter_acceleration, allocate_events

''' Read in file '''
path = 'C:\\Users\\alexw\\Desktop\\tsFRESH\\Raw Data'
ext = 'csv'
os.chdir(path)
files = glob.glob('*.{}'.format(ext))

# Length of each sample = 600 ms
length = 100

explained_variance = []
explained_variance_ratio = []

for f in files:

	print('Running file: ' + str(f))

	data = read_csv(f)

	f = f.split('.')[0]

	X, y, force = prepare_data(data, length, f, False)

	# Get times/indices of FS and FO events
	FS = list(np.where(y[:,0] == 1)[0])
	FO = list(np.where(y[:,1] == 1)[0])
	
	FS_new, FO_new = sort_events(FS, FO, first_event='FS', final_event='FS')

	# Time array
	time_array = X[:,1]

	# Filter acceleration arrays
	a_left = X[:,2:5]
	a_right = X[:,5:8]
	a_left_filt = filter_acceleration(a_left)
	a_right_filt = filter_acceleration(a_right)

	# Reshape data into strides
	left_stride, right_stride = allocate_events(time_array, a_left_filt, a_right_filt, FS_new, FO_new)

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

print('Mean explained variance of first component:e: {}'.format(np.mean(explained_variance)))
print('Mean explained variance ratio of first component:: {}'.format(np.mean(explained_variance_ratio)))

pca = PCA()

pca.fit(res_all)

explained_variance_all = pca.explained_variance_[0]
explained_variance_ratio_all = pca.explained_variance_ratio_[0]

print('Explained variance: {}'.format(explained_variance_all))
print('Explained variance ratio: {}'.format(explained_variance_ratio_all))
