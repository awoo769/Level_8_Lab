"""
Create and write a TRC file

Note that you can also export a TRC file using Mocka
File > Export > Motion Analysis Corp. > TRC file 

"""

__author__ = "Nathan Brantly, Alex Woodall"
__version__ = "2.0"
__license__ = "ABI"

import numpy as np
from numpy import matlib
import os
import tkinter as tk
from tkinter import filedialog

def write_trc(marker_labels: list, header_information: dict, frame_numbers: np.ndarray, marker_data: np.ndarray, full_file_name: str):
	'''
	Creates a TRC file with (name, directory, and file extension (should always be '.trc') specified
	in full_file_name)

	Inputs:
			marker_labels: a list of all the marker labels
			header_information: a dictionary of the header information required for the trc file
			frame_numbers: an array of all the frame numbers
			marker_data: an array of all the marker data to write to the trc file
			full_file_name: a string of the full path and file name for the new trc file

	'''

	# Get header information from the information dictionary
	frame_rate = header_information["CameraRate"]
	n_markers = int(header_information["NumMarkers"])
	num_frames = int(header_information["NumFrames"])
	units = header_information["Units"]

	# Get the number of columns to write for the data section of the trc file
	_, num_col = np.shape(marker_data)

	file_extention = os.path.splitext(full_file_name)[-1] # Get the extension of the filename

	# Check that the new file name is a trc file
	if file_extention != '.trc':
		full_file_name = input("Please enter a full file name with the extension '.trc'") # Request a different filename from the user

	# Open the new file for writing
	fid = open(full_file_name, "w+")

	# Get just the file name from the full file name
	file_name = full_file_name.rsplit('/',1)[-1]
	
	# Write the header information
	fid.write("PathFileType\t4\t(X/Y/Z)\t%s\n" % (file_name))
	fid.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
	fid.write("%f\t%f\t%d\t%d\t%s\t%f\t%d\t%d\n" % (frame_rate, frame_rate, num_frames, n_markers, units, frame_rate, frame_numbers[0], frame_numbers[-1]))
	fid.write("Frame#\tTime\t")

	for i in range(n_markers):
		fid.write("%s\t\t\t" % (marker_labels[i]))

	fid.write("\n\t\t")

	for i in range(n_markers):
		fid.write("X%i\tY%i\tZ%i\t" % (i+1, i+1, i+1))

	fid.write("\n\n")

	# Close the file
	fid.close()

	# Now append the data to the file now that the header has been written out

	# Open the file to append to
	fid = open(full_file_name, "a+")

	# Save the marker array to the file
	for kk in range(num_frames):
		for ll in range(num_col): # If an entry in column 1 (i.e., frame#), write format as integer
			if ll == 0:
				fid.write("%d\t" % (marker_data[kk, ll]))

			elif ll == 1: # If an entry in column 2 (i.e., 'time'), write format as compact format
				fid.write("%g\t" % (marker_data[kk,ll]))

			elif ll == num_col - 1:
				fid.write("%0.5f\n" % (marker_data[kk,ll])) # Carriage return after writing each line

			else: # Otherwise, write format as fixed-point notation with 5 dp
				fid.write("%0.5f\t" % (marker_data[kk,ll]))

	# Close the file
	fid.close()

	# Print new file name for the user
	print(full_file_name)


