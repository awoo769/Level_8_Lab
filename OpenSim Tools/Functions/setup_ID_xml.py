import numpy as np
import opensim as osim

def setup_ID_xml(trial: str, model: str, directory: str, time_range: list, cut_off_freq: np.float64):
	'''
	Rewrites the ID setup xml file for a new trial
	
	Inputs:	trial: trial name, e.g.,  "_12Mar_ss_12ms_01"
			model: model name, e.g., "AB08"
			directory: output directory name
			time_range: start and end times
			cuf_off_frequency: low pass cut-off frequency

	'''

	# Create an instance of the inverse dynamics tool
	ID_tool = osim.InverseDynamicsTool()

	# Set tool name
	ID_tool.setName(model)

	# Set the opensim model name
	ID_tool.setModelFileName(directory + "\\" + model + "\\" + model + ".osim")

	# Set excluded forces
	excluded_forces = osim.ArrayStr()
	excluded_forces.setitem(0,'Muscles')
	ID_tool.setExcludedForces(excluded_forces)

	# Set low pass cut-off frequency, NOTE: Must be a double (np.float64)
	ID_tool.setLowpassCutoffFrequency(np.float64(cut_off_freq))

	# Set the input and results directory
	ID_tool.setResultsDir(directory + "\\" + model + "\\" + trial)
	ID_tool.setInputsDir(directory + "\\" + model + "\\" + trial)

	# Set the time range, NOTE: Must be a double (np.float64)
	ID_tool.setStartTime(np.float64(time_range[0]))
	ID_tool.setEndTime(np.float64(time_range[-1]))

	# Set the external loads file
	external_loads_file = directory + "\\" + model + "\\" + trial + "\\" + trial + 'ExternalLoads.xml'
	ID_tool.setExternalLoadsFileName(external_loads_file)

	# Set the coordinates file
	coordindate_file = directory + "\\" + model + "\\" + trial + "\\" + trial + 'IKResults.mot'
	ID_tool.setCoordinatesFileName(coordindate_file)

	# Set the output file
	output_file_name = trial + "IDResults.sto"
	ID_tool.setOutputGenForceFileName(output_file_name)
	
	''' Write changes to an XML setup file '''

	xml_setup_path = directory + "\\" + model + "\\" + trial + "\\" + trial + "IDSetup.xml"
	ID_tool.printToXML(xml_setup_path)