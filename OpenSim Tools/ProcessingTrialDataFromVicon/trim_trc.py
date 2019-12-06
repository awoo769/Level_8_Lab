import numpy as np

def trim_trc(old_markers: dict, old_frames: np.ndarray, old_times: np.ndarray, inds: list):
	'''
	Trims markers, frames, and times from a trc to within the start and end indices within inds

	'''
	
	# Create new dictionary for markers
	new_markers = {}

	for key in old_markers.keys():
		new_markers[key] = {}
		new_markers[key]["X"] = old_markers[key]["X"][inds[0]:inds[-1]+1]
		new_markers[key]["Y"] = old_markers[key]["Y"][inds[0]:inds[-1]+1]
		new_markers[key]["Z"] = old_markers[key]["Z"][inds[0]:inds[-1]+1]
		new_markers[key]["All"] = old_markers[key]["All"][inds[0]:inds[-1]+1]

	new_frames = old_frames[inds[0]:inds[-1]+1]
	new_time = old_times[inds[0]:inds[-1]+1]

	return new_markers, new_frames, new_time