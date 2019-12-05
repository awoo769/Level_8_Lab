import os
import sys
import numpy as np

from read_trc import read_trc
from read_mot import read_mot

def prepare_trial_from_Vicon(model: str, trial: str, output_directory: str, input_directory: str):
	'''
	prepare_trial_from_Vicon: A function to condition and collate trial data and setup all 
	necessary OpenSim analysis xmls.

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

	# List which contains the names of the motion capture trial which didn't record EMG data
	bad_EMG_trials = ['SAFIST015_SS21_20Jun_ss_035ms_02','SAFIST015_SS21_20Jun_fast_075ms_02',
	'SAFIST015_SS42_20Jun_ss_035ms_01','SAFIST015_SS42_20Jun_fast_055ms_01','SAFIST015_SS52_ss_04ms_02',
	'SAFIST015_SS52_fast_07ms_01','SS77_SAFIST015_18Jun_fast_04ms_02','SAFIST015_19Jun_SS90_ss_035ms_01',
	'SAFIST015_19Jun_SS90_fast_055ms_01','_12Mar_ss_12ms_01']

	bad_EMG = 0 # Set bad_EMG flag to 0 (trial contains good EMG data)
	recalculate_COP = 1 # Do you want to recalcuate the COP (recommended)

	# Check if the trial you are running is in the list of bad EMG trials
	if any(trial == s in s for s in bad_EMG_trials):
		bad_EMG = 1 # Set bad_EMG flag to 1 (trial does not contain good EMG data)

	# Identify files from Vicon export to read
	trc_filename = os.path.join(input_directory, trial + "." + "trc")
	mot_filename = os.path.join(input_directory, trial + "." + "mot")
	emg_filename = os.path.join(input_directory, trial + "_EMG." + "mot")

	# Check if the trc/mot/emg files do not exist
	if not os.path.exists(trc_filename):
		trc_filename = os.path.join(input_directory, model + trial + "." + "trc")

	if not os.path.exists(mot_filename):
		mot_filename = os.path.join(input_directory, model + trial + "." + "mot")

	if not os.path.exists(emg_filename):
		print('No EMG for subject %s.\n' % (model))
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

	# Generate filenames
	IK_filename = "IKSetup.xml"
	ID_filename = "IDSetup.xml"
	ex_loads_filename = "ExternalLoads.xml"
	muscle_analysis_filename = "MuscleAnalysisSetup.xml"
	muscle_force_direction_filename = "MuscleForceDirectionSetup.xml"

	# Add xml template file to path
	xml_directory = output_directory.replace("Output", "xmlTemplates")
	sys.path.insert(1,xml_directory)

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
	
	if not bad_EMG:
		#TODO
		a = 1

	grf_headers, full_grf_data = read_mot(8, mot_filename)

	if ('SS' in model) or ('AB' in model): # If SS or AB, recorded at AUT Millenium
		steps = ['l', 'r']
		plates = [1, 2]

	# Create time range
	time_range = []

# TODO find index of value, not the value
	time_range.append(round(max(time[0], 0) + 0.020, 3))
	time_range.append(time[-1])

	index_start = time[time == time_range[0]]
	index_end = time[time == time_range[1]]

	# Create frame range
	frame_range = []
	
	frame_range.append(frames[index_start]) 
	frame_range.append(frames[index_end])


	a = 1

# Let the user select the input and output directory folders in Jupyter notebook

# Put input directory and output directory as full file paths
output_directory = "C:\\Users\\alexw\\Dropbox\\ABI\\Level_8_Lab\\OpenSim Tools\\ProcessingTrialDataFromVicon\\Output"
input_directory = "C:\\Users\\alexw\\Dropbox\\ABI\\Level_8_Lab\\OpenSim Tools\\ProcessingTrialDataFromVicon\\InputDirectory"

prepare_trial_from_Vicon("AB08","_12Mar_ss_12ms_01", output_directory, input_directory)

