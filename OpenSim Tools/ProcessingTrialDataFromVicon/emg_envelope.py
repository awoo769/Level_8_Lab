import numpy as np
from scipy import signal
from sklearn.preprocessing import normalize

def emg_envelope(emg_data: np.array, emg_frequency: float):
	'''
	This function filters, rectifies, and normalises EMG data for use in muscle force analysis.
	NOTE: Based in part on code written by Ajay Seth for use with OpenSim
	http://simtk-confluence.stanford.edu:8080/display/OpenSim/Tools+for+Preparing+Motion+Data

	Inputs: emg_data: an array of all the EMG data from the trial
			emg_frequency: the frequency that the EMG data was recorded

	Output: emg_env: the EMG envelope

	'''

	# Use a fourth order Butterworth filter on the EMG data to filter
	bb, ab = signal.butter(4, [20/(emg_frequency/2), 400/(emg_frequency/2)], 'band')

	# Use a low pas filter to create EMG envelope
	bl, al = signal.butter(4, 10/(emg_frequency/2), 'low')

	emg_envelope = np.zeros(np.shape(emg_data))

	for i in range(np.shape(emg_data)[1]):
		filtered_emg = signal.filtfilt(bb, ab, emg_data[:,i], axis=0)
		rectified_emg = abs(filtered_emg)
		low_pass_emg = signal.filtfilt(bl, al, rectified_emg, axis=0)

		# Normalise 0-1 based on maximum and minimum value.
		# NOTE: normally, you would get an Fmax with a maximum force test. However, this is often difficult
		# with pathology patients
	
		emg_envelope[:,i] = (low_pass_emg - min(low_pass_emg)) / (max(low_pass_emg) - min(low_pass_emg))

	return emg_envelope
