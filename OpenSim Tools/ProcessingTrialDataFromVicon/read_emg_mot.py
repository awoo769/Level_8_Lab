from tkinter import filedialog
import numpy as np

def read_emg_mot(file_path: str):

	'''
	Reads an EMG motion file

	Inputs: file_path: containing the full path to the _EMG.mot file

	Outputs:	emg_headers: a list of the motion data file headers
				emg_data: the data from the motion file
				emg_frequency: the frequency of the recorded EMG data 

	'''

	# Open mot file with read only access
	mot_fid = open(file_path,'r')

	mot_lines = mot_fid.readlines()
	mot_fid.close()

	# EMG frequency stored on the second line
	emg_frequency = np.float(mot_lines[1])
	
	# Headers are stored on line 4
	emg_headers = mot_lines[3].split('\t')

	# Data stored from line 6 onwards
	mot_data = mot_lines[5:]

	data = []

	# Split into an array for easier manipulation
	for i in range(len(mot_data)):
		tmp = [float(i) for i in mot_data[i].split()]
		# Make sure that the line was not empty
		if tmp != []:
			data.append(tmp)

	# Convert to array of floats
	emg_data = np.array(data)

	return emg_headers, emg_data, emg_frequency