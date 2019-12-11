import numpy as np
import opensim as osim

def setup_muscle_analysis_xml(muscle_filename: str, trial: str, model: str, directory: str, time_range: list, cut_off_freq: np.float64):
	'''
	NOTE: Any attribute not changed within this function must be changed in the original template 
	file 

	Inputs:	muscle_filename: full filename for the template muscle analysis setup xml file
			trial: trial name, e.g.,  "_12Mar_ss_12ms_01"
			model: model name, e.g., "AB08"
			directory: output directory name
			time_range: start and end times
			cuf_off_frequency: low pass cut-off frequency

	'''

	# Get analyze tool
	#analyze_tool = osim.AnalyzeTool()
	analyze_tool = osim.AnalyzeTool(muscle_filename, True)

	# Set tool name
	new_analyze_tool_name = model + trial
	analyze_tool.setName(new_analyze_tool_name)

	# Set the opensim model name
	analyze_tool.setModelFilename(directory + "\\" + model + "\\" + model + ".osim")
	
	# Set the results directory
	analyze_tool.setResultsDir(directory + "\\" + model + "\\" + trial)

	# Set the external loads file
	external_loads_file = directory + "\\" + model + "\\" + trial + "\\" + trial + 'ExternalLoads.xml'
	analyze_tool.setExternalLoadsFileName(external_loads_file)

	# Set the coordinates file
	coord_file = directory + "\\" + model + "\\" + trial + "\\" + trial + 'IKResults.mot'
	analyze_tool.setCoordinatesFileName(coord_file)

	# Set low pass cut-off frequency, NOTE: Must be a double (np.float64)
	analyze_tool.setLowpassCutoffFrequency(np.float64(cut_off_freq))

	# Set the time range, NOTE: Must be a double (np.float64)
	analyze_tool.setInitialTime(np.float64(time_range[0]))
	analyze_tool.setFinalTime(np.float64(time_range[-1]))

	for analysis in analyze_tool.getAnalysisSet():
		analysis.setStartTime(np.float64(time_range[0]))
		analysis.setEndTime(np.float64(time_range[-1]))
	
	''' Write file '''
	
	xml_setup_path = directory + "\\" + model + "\\" + trial + "\\" + trial + muscle_filename.split("\\")[-1]
	analyze_tool.printToXML(xml_setup_path)