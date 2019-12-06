import numpy as np

def rezero_filter(original_fy: np.ndarray):
	'''
	Resets all values which were originally zero to zero

	'''

	filter_plate = np.zeros(np.shape(original_fy))

	for i in range(len(original_fy[0,:])):
		positive_fy = original_fy[:,i] > 20

		true_ind = np.where(positive_fy)[0]

		filter_plate[true_ind, i] = True

	a = 1