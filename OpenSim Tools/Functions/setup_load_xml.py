import numpy as np
import opensim as osim

def setup_load_xml(trial: str, model: str, directory: str, cut_off_freq: np.float64):
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
	external_loads = osim.ExternalLoads()

	# Set name
	external_loads.setName(model)

	# Set motion file
	mot_string = directory + "\\" + model + "\\" + trial + "\\" + trial + ".mot"
	external_loads.setDataFileName(mot_string)

	# Set cut-off frequency, NOTE: Must be a double (np.float64)
	external_loads.setLowpassCutoffFrequencyForLoadKinematics(np.float64(cut_off_freq))

	''' Add external forces '''
	
	# Left side
	external_force_left = osim.ExternalForce()
	
	external_force_left.setName("left")

	external_force_left.set_applied_to_body("calcn_l")
	external_force_left.set_force_expressed_in_body("ground")
	external_force_left.set_point_expressed_in_body("ground")

	external_force_left.set_force_identifier("1_ground_force_v")
	external_force_left.set_point_identifier("1_ground_force_p")
	external_force_left.set_torque_identifier("1_ground_torque_")
	
	# Adopt and append is causing code to stop after printing to XML. Unsure why.
	external_loads.cloneAndAppend(external_force_left)
	
	# Right side
	external_force_right = osim.ExternalForce()
	
	external_force_right.setName("right")

	external_force_right.set_applied_to_body("calcn_r")
	external_force_right.set_force_expressed_in_body("ground")
	external_force_right.set_point_expressed_in_body("ground")

	external_force_right.set_force_identifier("ground_force_v")
	external_force_right.set_point_identifier("ground_force_p")
	external_force_right.set_torque_identifier("ground_torque_")
	
	external_loads.cloneAndAppend(external_force_right)
	
	''' Write new file '''

	new_filename = directory + "\\" + model + "\\" + trial + "\\" + trial + "ExternalLoads.xml"

	external_loads.printToXML(new_filename)

