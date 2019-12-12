import numpy as np

def write_emg(emg_data: np.array, emg_headers: list, emg_filename: str):
	'''
	Writes EMG .mot files. Based on a script originally written by Ajay Seth.

	Inputs:	emg_data: an array containing all the emg data (after processing)
			emg_headers: a list of all the headers to write to file
			emg_filename: the full filename to write the new emg .mot file

	'''

	fid = open(emg_filename, 'w')

	# Write header
	fid.write("Normalized EMG Linear Envelopes\n")
	fid.write("nRows=%d\n" % (np.shape(emg_data)[0]))
	fid.write("nColumns=%d\n\n" % (np.shape(emg_data)[-1]))
	fid.write("endheader\n")

	# Add labels to header
	for header in emg_headers:
		if header != emg_headers[-1]:
			fid.write("%s\t" % (header))
		else:
			fid.write("%s" % (header))

	if '\n' not in header: # if there isn't already a new line at the end of the final header
		fid.write('\n') # Write one

	# Now append data
	for i in range(np.shape(emg_data)[0]):
		for j in range(np.shape(emg_data)[-1]):
			if j == 0:
				fid.write("%4.3f\t" % (emg_data[i,j])) # Time
			else:
				fid.write("%0.6f\t" % (emg_data[i,j])) # EMG data
		
		fid.write("\n")
	
	fid.close()

	print("New EMG MOT location: %s\n" % (emg_filename))