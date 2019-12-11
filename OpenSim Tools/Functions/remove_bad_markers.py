import numpy as np

def remove_bad_markers(trimmed_markers: dict, marker_names: list):
	'''
	This function checks for gaps and duplicate markers, then removes them.

	Inputs: trimmed_markers: a dictionary containing the markers and their data
			marker_names: a list of the marker names

	Outputs:	trimmed_markers: a dictionary of the good marker names and their data
				marker_names: a list of the good marker names
				bad_keys: a list of the bad marker names

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