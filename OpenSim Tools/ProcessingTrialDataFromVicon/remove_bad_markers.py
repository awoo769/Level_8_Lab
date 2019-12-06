import numpy as np

def remove_bad_markers(trimmed_markers: dict, marker_names: list):
	'''
	This function checks for gaps and duplicate markers, then removes them.

	'''

	# Check for gaps
	bad_keys = []
	
	for key in trimmed_markers.keys():
		if np.size(np.where(abs(trimmed_markers[key]["X"]) == 0)) > 10:
			bad_keys.append(key)
	
	# Cannot get duplicate keys in a dictionary structure

	# Remove all bad markers, if there is a duplicate, remove both
	for key in bad_keys:
		trimmed_markers.pop(key, None)
		marker_names.remove(key)

	return trimmed_markers, marker_names, bad_keys