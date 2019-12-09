import numpy as np
import re

def rezero_filter(original_fy: np.ndarray):
	'''
	Resets all values which were originally zero to zero

	'''

	filter_plate = np.zeros(np.shape(original_fy))

	for i in range(len(original_fy[0,:])):
		# Binary test for values greater than 20
		force_zero = (original_fy[:,i] > 20) * 1 # Convert to 1 or 0 rather than True or False

		# We do not want to accept values which are over 20 but considered 'nosie'.
		# Must be over 20 for more than 10 frames in a row. Therefore, need to search for
		# [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] and get all the indices that meet this condition

		# Convert to string to test this condition
		force_str = ''.join(list(map(str, force_zero)))

		# Find all occurrences where the pattern occurs
		true_inds = [m.start() for m in re.finditer('(?=1111111111)', force_str)]

		# true_inds will not include the ends (e.g., 11...11100000) - will not include the final 3 1's
		extra_inds = [i + 10 for i in true_inds[0:-1]] # So make another array with 10 added on to all but the last value
		
		# Return the 'filtered' rezeroing array
		filter_plate[true_inds,i] = 1
		filter_plate[extra_inds,i] = 1

	return filter_plate