"""
Read marker data from a TRC file

"""

__author__ = "Nathan Brantly, Alex Woodall"
__version__ = "2.0"
__license__ = "ABI"

import numpy as np
from numpy import matlib
import os
import tkinter as tk
from tkinter import filedialog

def read_trc(*file_path: str):
	"""
	Create a class of a TRC marker data file 

	trc_read() returns a class, trcContents (format shown below), containing the contents of the TRC file selected 
	via a user interface dialog box.

	Input: file_path: optional argument containing the full file path to the .trc file

	Output Structure Format:

	trcContents (Dictionary): 
		Information (Dictionary): Field names and values populated from the 2nd and 3rd file header lines; Number of fields and
							field names may vary.
			FileName (1x? char): From filedialog.askopenfilename (e.g., 'Static.trc')
			PathName (1x? char): From filedialog.askopenfilename (e.g., 'C://Users//')
			DataRate (1x1 float32): Data collection frequency in Hz
			CameraRate (1x1 float32): Frame capture frequency in Hz
			NumFrames (1x1 uint32): Number of frames
			NumMarkers (1x1 uint16): Number of markers
			Units (1x? char): Units of the data (e.g. 'mm')
			OrigDataRate (1x1 float32): Original data collection frequency
			DataStartFrame (1x1 uint32): Starting frame number (sometimes seen as 'OrigDataStartFrame')
			OrigNumFrames (1x1 uint32): Original number of frames (sometimes not in TRC file)

		Data (Dictionary): Data from the remaining TRC file entries
			MarkerLabels (NumMarkers x 1 list): String list containing all unmodified marker labels
			ModifiedMarkerLabels (NumMarkers x 1 list): Modified marker labels (i.e., without '.' & ':' chars for use in 'Markers' Class)
			FrameNums (NumFrames x 1 uint32): 1d array of frame numbers
			Time (NumFrames x 1 float32): 1d array of time values (based on the DataRate)
			RawData (NumFrames x (3*NumMarkers) float64): ndarray of all raw data
			Markers (Dictionary): All modified marker labels and their corresponding 3D coordinate data
				ModifiedMarkerLabels{n} (Dictionary) nth marker containing X, Y, Z coordinate values
					X: (NumFrames x 1 float64): 1d array of X coordinate values for the marker
					Y: (NumFrames x 1 float64): 1d array of Y coordinate values for the marker
					Z: (NumFrames x 1 float64): 1d array of Z coordinate values for the marker
					All: (NumFrames x 3 float64): ndarray of X,Y,Z marker data, equivalent to [ X Y Z ]
	
	Example: trcContents = trcRead()

	"""	

	# If optional argument is given, don't find filepath, if not, then do
	if len(file_path) == 0:
		root = tk.Tk()
		root.withdraw()

		file_path = filedialog.askopenfilename(initialdir = "r",title = "Select file",filetypes = (("trc files","*.trc"),("all files","*.*")))

	else:
		# Convert file path tuple to string
		file_path = str(file_path[0])

	# Split into file_name and path_name

	file_name = file_path.rsplit('/',1)[-1]
	path_name = file_path.rsplit('/',1)[0]

	if not file_name.strip(): # If the user selects 'cancel'
		# Display a message and reprompt the user to select a file
		print('You have selected ''Cancel''. Please select a TRC file.')
		file_path = filedialog.askopenfilename(initialdir = "r",title = "Select file",filetypes = (("trc files","*.trc"),("all files","*.*")))

		# Split into file_name and path_name

		file_name = file_path.rsplit('/',1)[-1]
		path_name = file_path.rsplit('/',1)[0]

		if not file_name.strip(): # If the user selects 'cancel' again
			# Display a message
			print('Leaving readTRC')

			return
	
	# Open trc file with read only access
	trc_fid = open(os.path.join(path_name, file_name),'r')

	# Process first four header lines and store all entries as strings. The fifth (last) header line only 
	# contains X,Y,Z and marker numbers.

	trc_lines = trc_fid.readlines()
	trc_fid.close

	# No useful information from the first header line
	
	h_line2 = trc_lines[1] # Second line, contains information headings
	h_line2_entries = h_line2.split()
	
	h_line3 = trc_lines[2] # Third line, contains information values
	h_line3_entries = h_line3.split()
	
	h_line4 = trc_lines[3] # Fourth line, contains marker labels
	h_line4_entries = h_line4.split()

	# Create a dictionary to structure trc file
	trcContents = {}

	# The dictionary contains two parents
	trcContents["Information"] = {}
	trcContents["Data"] = {}

	# Add the 'file_name' and 'path_name' variables to trcContents.Information
	
	trcContents["Information"]["FileName"] = file_name
	trcContents["Information"]["PathName"] = path_name

	# Add header information to the trcContents.Information

	trcContents["Information"][h_line2_entries[0]] = np.float32(h_line3_entries[0])
	trcContents["Information"][h_line2_entries[1]] = np.float32(h_line3_entries[1])
	trcContents["Information"][h_line2_entries[2]] = np.float32(h_line3_entries[2])

	# The number of markers is a positive integer, store in a variable for convienient use
	num_markers = np.uint16(h_line3_entries[3])

	trcContents["Information"][h_line2_entries[3]] = num_markers
	trcContents["Information"][h_line2_entries[4]] = h_line3_entries[4]
	trcContents["Information"][h_line2_entries[5]] = np.float32(h_line3_entries[5])
	trcContents["Information"][h_line2_entries[6]] = np.uint32(h_line3_entries[6])

	
	if np.size(h_line3_entries) > 7: # Some TRC files contain additional entries
		trcContents["Information"][h_line2_entries[7]] = np.uint32(h_line3_entries[7])
		
	# Skipping all five header lines, organise the data into three matrices: 
	# a 32-bit unsigned int array containing the frame numbers
	# a single precision array containing times
	# a double precision matrix containing the marker location values for that frame 

	frame_numbers = []
	time = []
	marker_data = []

	# Line 6 in the trc file is often a '\n', but sometimes the new line is not there, so check
	trc_line_6 = trc_lines[5]

	if trc_line_6 == '\n':
		start = 6 # Start on line 7

	else:
		start = 5 # Start on line 6

	for line in trc_lines[start:]:
		frame_numbers.append(np.uint32(int(line.split('\t')[0]))) # Get frame number (integer)
		time.append(np.float32(line.split('\t')[1]))
		marker_data.append(np.float64(line.rsplit('\t')[2:]))
	
	# Convert lists of arrays to ndarray
	marker_data = np.stack(marker_data, axis=0)
	time = np.stack(time, axis=0)
	frame_numbers = np.stack(frame_numbers, axis=0)

	# Marker labels come after the 'Frame#' and 'Time' strings
	marker_labels = h_line4_entries[2:]

	trcContents["Data"]["MarkerLabels"] = marker_labels

	# Check that num_markers equals the number of marker labels, check is most likely unnecessary
	if np.size(marker_labels) != num_markers:
		print("Number of marker labels does not equal the number of markers, exiting")

		return

	# Replace periods with underscores in marker names due to class format
	marker_labels_WO_periods = []
	mod_marker_labels = []

	for i in range(num_markers):
		marker_labels_WO_periods.append(marker_labels[i].replace('.','_'))
		mod_marker_labels.append(marker_labels_WO_periods[i].replace(':','_')) # Replace colons with underscores

	# Store modified marker labels
	trcContents["Data"]["ModifiedMarkerLabels"] = mod_marker_labels

	# Output the array of frames and times for comprehensiveness
	trcContents["Data"]["FrameNums"] = frame_numbers
	trcContents["Data"]["Time"] = time
	trcContents["Data"]["RawData"] = marker_data

	# Create structure for each marker containing three column arrays of X, Y, and Z data and a num_frames x 3 all data
	# matrix (based on laboratory coordinate frame), easily tying each marker label to its respective data.

	trcContents["Data"]["Markers"] = {}

	for i in range(1,num_markers+1):
	
		trcContents["Data"]["Markers"][mod_marker_labels[i-1]] = {}
		trcContents["Data"]["Markers"][mod_marker_labels[i-1]]["X"] = marker_data[:,i*3 - 3]
		trcContents["Data"]["Markers"][mod_marker_labels[i-1]]["Y"] = marker_data[:,i*3 - 2]
		trcContents["Data"]["Markers"][mod_marker_labels[i-1]]["Z"] = marker_data[:,i*3 - 1]
		trcContents["Data"]["Markers"][mod_marker_labels[i-1]]["All"] = marker_data[:,(i*3 - 3):(i*3)]

	return trcContents, file_path
