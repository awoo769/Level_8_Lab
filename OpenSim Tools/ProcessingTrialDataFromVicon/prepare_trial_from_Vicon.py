import os
import sys
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from read_trc import read_trc
from write_trc import write_trc
from read_mot import read_mot
from trim_trc import trim_trc
from remove_bad_markers import remove_bad_markers
from rezero_filter import rezero_filter
from fix_grf_headers import fix_grf_headers
from write_mot import write_mot
from xml_shorten import xml_shorten
from read_emg_mot import read_emg_mot
from emg_envelope import emg_envelope
from write_emg import write_emg

from setup_muscle_force_direction_xml import setup_muscle_force_direction_xml

# OpenSim API
from setup_ID_xml import setup_ID_xml
from setup_IK_xml import setup_IK_xml
from setup_load_xml import setup_load_xml
from setup_scale_xml import setup_scale_xml
from setup_muscle_analysis_xml import setup_muscle_analysis_xml

def prepare_trial_from_Vicon(model: str, trial: str, output_directory: str, input_directory: str):
	'''
	prepare_trial_from_Vicon: A function to condition and collate trial data and setup all 
	necessary OpenSim analysis xmls.

	Assumes data has been pre-processed in Nexus.
	Assumes that in the input directory, there is (at least) a trc and mot file of the trial.

	Inputs:		model = Name of subject, assuming model file is "Subject.osim"
				trial = Name of motion capture trial
				output_directory = Location of output
				input_directory = Location of input files

	Example: prepare_trial_from_Vicon('AB08', '_12Mar_ss_12ms_01', 'Output', 'InputDirectory')
	This will take the AB08_12Mar_ss_12ms_01.trc/mot file from the folder 'InputDirectory' and
	deposit the output in the folder 'Output'

	'''

	''' Initial Setup/Names '''

	# Change input and output directories to absolute file paths, if they are already absolute,
	# nothing will change
	input_directory = os.path.abspath(input_directory)
	output_directory = os.path.abspath(output_directory)

	# List which contains the n ames of the motion capture trial which didn't record EMG data
	bad_EMG_trials = ['SAFIST015_SS21_20Jun_ss_035ms_02','SAFIST015_SS21_20Jun_fast_075ms_02',
	'SAFIST015_SS42_20Jun_ss_035ms_01','SAFIST015_SS42_20Jun_fast_055ms_01','SAFIST015_SS52_ss_04ms_02',
	'SAFIST015_SS52_fast_07ms_01','SS77_SAFIST015_18Jun_fast_04ms_02','SAFIST015_19Jun_SS90_ss_035ms_01',
	'SAFIST015_19Jun_SS90_fast_055ms_01','_12Mar_ss_12ms_01']

	bad_EMG = 0 # Set bad_EMG flag to 0 (trial contains good EMG data)
	recalculate_COP = 1 # Do you want to recalcuate the COP (recommended)

	# Check if the trial you are running is in the list of bad EMG trials
	if any(trial == s in s for s in bad_EMG_trials):
		bad_EMG = 1 # Set bad_EMG flag to 1 (trial does not contain good EMG data)

	# Identify files from Vicon/Nexus export to read
	trc_filename = os.path.join(input_directory, trial + "." + "trc")
	mot_filename = os.path.join(input_directory, trial + "." + "mot")
	emg_filename = os.path.join(input_directory, trial + "_EMG." + "mot")

	# Check if the trc/mot/emg files do not exist
	if not os.path.exists(trc_filename):
		trc_filename = os.path.join(input_directory, model + trial + "." + "trc")

	if not os.path.exists(mot_filename):
		mot_filename = os.path.join(input_directory, model + trial + "." + "mot")

	if not os.path.exists(emg_filename):
		emg_filename = os.path.join(input_directory, model + trial + "_EMG." + "mot")
    
		if not os.path.exists(emg_filename):
			print('No EMG data for subject %s.\n' % (model))
			bad_EMG = 1

	# Make new folder for the output of this model if it doesn't already exist
	output_model_dir = os.path.join(output_directory, model)
	output_model_trial_dir = os.path.join(output_model_dir, trial)
	
	if not os.path.exists(output_model_dir):
		os.mkdir(output_model_dir)
		print("Making new ouput model directory")

	if not os.path.exists(output_model_trial_dir):
		os.mkdir(output_model_trial_dir)
		print("Making new output model trial directory")

	# XML directory - maybe let user select this
	xml_directory = output_directory.replace("Output", "xmlTemplates")

	# Generate filenames
	muscle_force_direction_filename = os.path.join(xml_directory, "MuscleForceDirectionSetup.xml")

	''' Pull in exported Vicon files, identify time range of interest '''
	# Note: this approach differs with regard to available event data

	# Read the trc file
	mkr_data, _ = read_trc(trc_filename)

	# Pull out the data of interest
	frames = mkr_data["Data"]["FrameNums"]
	time = mkr_data["Data"]["Time"]
	data_rate = mkr_data["Information"]["DataRate"]
	markers = mkr_data["Data"]["Markers"]
	marker_names = mkr_data["Data"]["ModifiedMarkerLabels"]

	print(model)
	print(trial)
	print('\n')
	
	if not bad_EMG:
		emg_headers, emg_data, emg_frequency = read_emg_mot(emg_filename)

	grf_headers, full_grf_data = read_mot(8, mot_filename)

	if ('SS' in model) or ('AB' in model): # If SS or AB, recorded at AUT Millenium
		steps = ['l', 'r']
		plates = [1, 2]

	# Create time range
	time_range = []

	time_range.append(round(max(time[0], 0) + 0.020, 3))
	time_range.append(np.float64(time[-1]))

	index_start = np.where(time == time_range[0])
	index_end = np.where(time == time_range[1])

	# Create frame range
	frame_range = []
	
	frame_range.append(frames[index_start]) 
	frame_range.append(frames[index_end])

	''' IK file '''
	
	trimmed_markers, trimmed_frames, trimmed_time = trim_trc(markers, frames, time, [int(index_start[0]), int(index_end[0])])

	# Remove markers may not be needed, as conditioning done in Nexus
	good_markers, good_marker_names, bad_marker_names = remove_bad_markers(trimmed_markers, marker_names)
	
	# Convert good_markers into an ndarray (nframes x nmarkers * 3)
	marker_data = []
	
	for keys in good_markers.keys():
		marker_data.append(good_markers[keys]["All"])
	
	marker_data = np.array(marker_data).transpose(1,0,2).reshape(len(trimmed_frames),-1)

	# concatenate marker data with frame numbers and times
	new_mkr_data = np.concatenate((trimmed_frames[:, np.newaxis], trimmed_time[:, np.newaxis], marker_data),axis=1)

	new_filename = os.path.join(output_model_trial_dir, trial + "." + "trc")

	# Edit mkr_data["Information"] for trimmed dataset
	mkr_data["Information"]["NumFrames"] = len(trimmed_frames)
	mkr_data["Information"]["NumMarkers"] = np.size(good_marker_names)

	write_trc(good_marker_names, mkr_data["Information"], trimmed_frames, new_mkr_data, new_filename)

	# Create the IK setup xml file using the OpenSim API
	setup_IK_xml(trial, model, output_directory, time_range, good_marker_names)
	filename = output_directory + "\\" + model + "\\" + trial + "\\" + trial + "IKSetup.xml"
	xml_shorten(filename)
	
	''' ID files '''

	# Define the grf capture rate
	grf_rate = (len(full_grf_data) - 1) / (full_grf_data[-1,0] - full_grf_data[0,0])

	# Get the original vertical forces
	indices = [i for i, s in enumerate(grf_headers) if 'vy' in s]
	original_fy = full_grf_data[:,indices]

	# Filter GRF data
	cut_off_frequency = 10
	Wn = cut_off_frequency/(grf_rate/2)

	# Describe filter characteristics using 4th order Butterworth filter
	b, a = signal.butter(4, Wn)

	new_grf_data = np.zeros(np.shape(full_grf_data))
	new_grf_data[:,0] = full_grf_data[:,0] # Not filtering time data

	for i in range(1,len(grf_headers)):
		new_grf_data[:,i] = signal.filtfilt(b, a, full_grf_data[:,i], axis=0)

	# Re-zero grfs
	filter_plate = rezero_filter(original_fy)

	# Re-zero all columns except those which refer to the centre of pressure
	# Assumes that the only headers which contain 'p' are CoP

	for i in range(1,len(grf_headers)): # Are not rezeroing time
		# If not centre of pressure AND force plate 1
		if ('p' not in grf_headers[i]) and ('1' in grf_headers[i]):
			new_grf_data[:,i] = filter_plate[:,0] * new_grf_data[:,i]
		# If not centre of pressure AND force plate 2
		elif ('p' not in grf_headers[i]) and ('2' in grf_headers[i]):
			new_grf_data[:,i] = filter_plate[:,1] * new_grf_data[:,i]

	if recalculate_COP:
		# Define for recalculating CoP - position of force plates
		x_offset = [0.2385, 0.7275]
		y_offset = [0, 0]

		# OpenSim Coordinate frame has y upwards. We will convert to x and y being the plane parallel
		# the ground for convenience (will return to OpenSim coordinates when creating new grf data)
		vz_inds = [i for i, s in enumerate(grf_headers) if 'vy' in s]
		px_inds = [i for i, s in enumerate(grf_headers) if 'px' in s]
		py_inds = [i for i, s in enumerate(grf_headers) if 'pz' in s]

		fZ = np.zeros(np.shape(filter_plate.T))
		pX = np.zeros(np.shape(filter_plate.T))
		pY = np.zeros(np.shape(filter_plate.T))
		oldmY = np.zeros(np.shape(filter_plate.T))
		oldmX = np.zeros(np.shape(filter_plate.T))

		# Back calculate moment measurements
		for i in range(len(plates)):
			side_inds = [j for j, s in enumerate(grf_headers) if str(i+1) in s]

			fZ[i,:] = full_grf_data[:,list(set(side_inds).intersection(vz_inds))].T
			pX[i,:] = full_grf_data[:,list(set(side_inds).intersection(px_inds))].T
			pY[i,:] = full_grf_data[:,list(set(side_inds).intersection(py_inds))].T

			oldmX[i,:] = (y_offset[i] + pY[i,:]) * fZ[i,:]
			oldmY[i,:] = (x_offset[i] - pX[i,:]) * fZ[i,:]

		# Filter old moments
		mX = signal.filtfilt(b, a, oldmX, axis=1)
		mY = signal.filtfilt(b, a, oldmY, axis=1)

		# Rezero moments
		for i in range(len(plates)):
			mX[i,:] = filter_plate[:,i].T * mX[i,:]
			mY[i,:] = filter_plate[:,i].T * mY[i,:]

		# Recalculate CoP with filtered forces and moments
		new_pX = np.zeros(np.shape(pX))
		new_pY = np.zeros(np.shape(pY))

		for i in range(len(plates)):
			side_inds = [j for j, s in enumerate(grf_headers) if str(i+1) in s]
			new_fZ = new_grf_data[:, list(set(side_inds).intersection(vz_inds))]
			
			for j in range(len(mY[i,:])):
				if new_fZ[j] != 0:
					new_pX[i,j] = x_offset[i] - (mY[i,j] / new_fZ[j])
					new_pY[i,j] = y_offset[i] + (mX[i,j] / new_fZ[j])
				else:
					new_pX[i,j] = 0
					new_pY[i,j] = 0

			new_grf_data[:, list(set(side_inds).intersection(px_inds))] = new_pX[i,:][:,np.newaxis]
			new_grf_data[:, list(set(side_inds).intersection(py_inds))] = new_pY[i,:][:, np.newaxis]

			plt.subplot(1,2,i+1)
			plt.plot(pX[i,:], pY[i,:], '*')
			plt.plot(new_pX[i,:], new_pY[i,:], 'x')

			print('Just associated new CoP for plate %i' % (i+1))

		plt.show()
		print('\n')

	grf_data = new_grf_data[int(np.where(np.float32(new_grf_data[:,0]) == time_range[0])[0]):int(np.where(np.float32(new_grf_data[:,0]) == time_range[-1])[0] + 1), :]

	new_headers = fix_grf_headers(grf_headers, steps, plates)
	
	new_filename = os.path.join(output_model_trial_dir, trial + "." + "mot")
	write_mot(grf_data, new_filename, new_headers)
	
	# Create the ID setup xml file using the OpenSim API
	setup_ID_xml(trial, model, output_directory, time_range, cut_off_frequency)

	# Create the external load setup xml file using the OpenSim API
	setup_load_xml(trial, model, output_directory, cut_off_frequency)

	''' EMG Processing '''

	if bad_EMG == 0:
		emg_env = emg_envelope(emg_data, emg_frequency)

		# Return correct frame and sub-frame numbers
		if ('Frame' in emg_headers[0]) and ('Frame' in emg_headers[1]):
			emg_env[:,0:2] = emg_data[:,0:2]
		
		emg_delay = 0.020 # 2 frames at 100 Hz or 4 frames at 200 Hz
		frame_offset = emg_delay / (1/data_rate)
		emg_start = np.where(emg_env[:,0] == (frame_range[0] - frame_offset))[0]
		emg_end = np.where(emg_data[:,0] == (frame_range[-1] - frame_offset))[0]

		emg_time = np.linspace((time_range[0] - emg_delay), (time_range[-1] - emg_delay), len(grf_data))

		# Clip the top and bottom of the EMG data to fit the grf data size
		clipped_emg = emg_env[emg_start[-1]:emg_end[-1]+1,:]

		emg = np.concatenate((emg_time[:, np.newaxis], clipped_emg[:, 2:]), axis=1)

		emg_labels = emg_headers[2:]
		emg_labels.insert(0, 'time')

		emg_new_filename = output_directory + "\\" + model + "\\" + trial + "\\" + trial + "_EMG.mot"
		write_emg(emg, emg_labels, emg_new_filename)


	''' Muscle Analysis Files '''

	# Create the muscle analysis setup xml file using the OpenSim API
	setup_muscle_analysis_xml(trial, model, output_directory, time_range, cut_off_frequency)
	filename = output_directory + "\\" + model + "\\" + trial + "\\" + trial + "MuscleAnalysisSetup.xml"
	xml_shorten(filename)

	# Create the muscle force direction setup xml file. This is a 3rd party analysis so does not use OpenSim API
	setup_muscle_force_direction_xml(muscle_force_direction_filename, trial, model, output_directory, time_range, cut_off_frequency)
	filename = output_directory + "\\" + model + "\\" + trial + "\\" + trial + "MuscleForceDirectionSetup.xml"
	xml_shorten(filename)

	print("Completed trial preparation for model = %s\ttrial = %s." % (model, trial))


# Let the user select the input and output directory folders in Jupyter notebook

# Put input directory and output directory as full file paths
output_directory = "C:\\Users\\alexw\\Dropbox\\ABI\\Level_8_Lab\\OpenSim Tools\\ProcessingTrialDataFromVicon\\Output"
input_directory = "C:\\Users\\alexw\\Dropbox\\ABI\\Level_8_Lab\\OpenSim Tools\\ProcessingTrialDataFromVicon\\InputDirectory"
xml_directory = "C:\\Users\\alexw\\Dropbox\\ABI\\Level_8_Lab\\OpenSim Tools\\ProcessingTrialDataFromVicon\\xmlTemplates"

prepare_trial_from_Vicon("AB28","_05Apr_ss_11ms_01", output_directory, input_directory)
#prepare_trial_from_Vicon("AB08","_12Mar_ss_12ms_01", output_directory, input_directory)

