import numpy as np
import opensim as osim
from xml.dom import minidom

def setup_muscle_analysis_xml(trial: str, model: str, directory: str, time_range: list, cut_off_freq: np.float64):
	'''
	NOTE: Any attribute not changed within this function must be changed in the original template 
	file 
	
	Inputs:	trial: trial name, e.g.,  "_12Mar_ss_12ms_01"
			model: model name, e.g., "AB08"
			directory: output directory name
			time_range: start and end times
			cuf_off_frequency: low pass cut-off frequency

	'''

	# Get analyze tool
	analyze_tool = osim.AnalyzeTool()	
	
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

	''' Add muscle analysis '''

	analysis_set = analyze_tool.getAnalysisSet()

	muscle_analysis = osim.MuscleAnalysis()

	muscle_analysis.setStartTime(np.float64(time_range[0]))
	muscle_analysis.setEndTime(np.float64(time_range[-1]))
	muscle_analysis.setComputeMoments(True) # Bug, this is not being set to true in the xml file

	analysis_set.cloneAndAppend(muscle_analysis)
	
	''' Write file '''
	
	xml_setup_path = directory + "\\" + model + "\\" + trial + "\\" + trial + "MuscleAnalysisSetup.xml"
	analyze_tool.printToXML(xml_setup_path)

	''' Temporary fix to set compute moments to true and to remove numerical inaccuracy in times '''
	
	dom = minidom.parse(xml_setup_path)
	analysis_set = dom.getElementsByTagName("AnalysisSet")
	analysis_set_child = analysis_set.item(0)

	objects_set = analysis_set_child.getElementsByTagName("objects")
	objects_set_child = objects_set.item(0)

	muscle_analysis = objects_set_child.getElementsByTagName("MuscleAnalysis")
	muscle_analysis_child = muscle_analysis.item(0)

	muscle_analysis_child.getElementsByTagName("compute_moments")[0].firstChild.nodeValue = "true"

	dom.getElementsByTagName("initial_time")[0].firstChild.nodeValue = round(time_range[0],3)
	dom.getElementsByTagName("final_time")[0].firstChild.nodeValue = round(time_range[-1],3)
	muscle_analysis_child.getElementsByTagName("start_time")[0].firstChild.nodeValue = round(time_range[0],3)
	muscle_analysis_child.getElementsByTagName("end_time")[0].firstChild.nodeValue = round(time_range[-1],3)

	with open(xml_setup_path, 'w') as xml_file:
		dom.writexml(xml_file, addindent='\t', newl='\n', encoding='UTF-8')