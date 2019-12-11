import numpy as np
import opensim as osim

def setup_load_xml(load_filename: str, trial: str, model: str, directory: str, cut_off_freq: np.float64):
	'''
	Takes the external loads xml file specified by load_filename, and sets the .mot files to those specified
	by trial, and writes to a new file of the two strings combined, i.e., "Walk1ExternalLoads.xml"

	Inputs: load_filename: full filename for the template external load setup xml file
			trial: trial name, e.g.,  "_12Mar_ss_12ms_01"
			model: model name, e.g., "AB08"
			directory: output directory name
			cuf_off_frequency: low pass cut-off frequency

	'''

	# Create external loads object
	external_loads = osim.ExternalLoads(load_filename, True)

	# Set name
	external_loads.setName(model)

	# Set motion file
	mot_string = directory + "\\" + model + "\\" + trial + "\\" + trial + ".mot"
	external_loads.setDataFileName(mot_string)

	# Set cut-off frequency, NOTE: Must be a double (np.float64)
	external_loads.setLowpassCutoffFrequencyForLoadKinematics(np.float64(cut_off_freq))
	
	''' Write new file '''

	new_filename = directory + "\\" + model + "\\" + trial + "\\" + trial + load_filename.split("\\")[-1]
	external_loads.printToXML(new_filename)
