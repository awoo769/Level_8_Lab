import numpy as np
from xml.dom import minidom

def change_muscle_analysis_xmlfile(muscle_filename: str, trial: str, model: str, directory: str, time__range: list, cut_off_freq: int):
	'''
	NOTE: Any attribute not changed within this function must be changed in the original template 
	file 

	'''

	doc_node = minidom.parse(muscle_filename)

	''' Get Hierarchy Access '''
	analyze_tool = doc_node.getElementsByTagName("AnalyzeTool")
	analyze_tool_child = analyze_tool.item(0)

	res_directory = analyze_tool_child.getElementsByTagName("results_directory")
	res_directory_child = res_directory.item(0)

	model_file = analyze_tool_child.getElementsByTagName("model_file")
	model_file_child = model_file.item(0)

	initial_time = analyze_tool_child.getElementsByTagName("initial_time")
	initial_time_child = initial_time.item(0)

	final_time = analyze_tool_child.getElementsByTagName("final_time")
	final_time_child = final_time.item(0)

	ex_loads_file = analyze_tool_child.getElementsByTagName("external_loads_file")
	ex_loads_file_child = ex_loads_file.item(0)

	coords_file = analyze_tool_child.getElementsByTagName("coordinates_file")
	coords_file_child = coords_file.item(0)

	analysis_set = analyze_tool_child.getElementsByTagName("AnalysisSet")
	analysis_set_child = analysis_set.item(0)

	objects_set = analysis_set_child.getElementsByTagName("objects")
	objects_set_child = objects_set.item(0)

	filter_frequency = analyze_tool_child.getElementsByTagName("lowpass_cutoff_frequency_for_coordinates")
	filter_frequency_child = filter_frequency.item(0)

	''' Muscle analysis settings '''

	muscle_analysis = objects_set_child.getElementsByTagName("MuscleAnalysis")
	muscle_analysis_child = muscle_analysis.item(0)

	muscle_analysis_start_time = muscle_analysis_child.getElementsByTagName("start_time")
	muscle_analysis_start_time_child = muscle_analysis_start_time.item(0)

	muscle_analysis_end_time = muscle_analysis_child.getElementsByTagName("end_time")
	muscle_analysis_end_time_child = muscle_analysis_end_time.item(0)

	''' Set new directory, filenames, and number inputs '''

	new_analyze_tool_name = model + trial
	analyze_tool_child.setAttribute('name', new_analyze_tool_name)

	# Local directory
	res_directory_child.firstChild.data = ".\\"

	# OpenSim model name
	model_file_name = model + ".osim"
	model_file_child.firstChild.data = model_file_name
	
	external_loads_file = trial + 'ExternalLoads.xml'
	ex_loads_file_child.firstChild.data = external_loads_file

	coordsfile = trial + 'IKResults.mot'
	coords_file_child.firstChild.data = coordsfile

	filter_frequency_child.firstChild.data = str(cut_off_freq)

	# Set start and end time for all tools
	starttime = str(time__range[0])
	endtime = str(time__range[-1])

	initial_time_child.firstChild.data = starttime
	final_time_child.firstChild.data = endtime

	muscle_analysis_start_time_child.firstChild.data = starttime
	muscle_analysis_end_time_child.firstChild.data = endtime
	
	''' Write file '''
	
	new_filename = directory + "\\" + model + "\\" + trial + "\\" + trial + muscle_filename.split("\\")[-1]

	with open(new_filename, 'w') as xml_file:
		doc_node.writexml(xml_file, addindent='\t', newl='\n', encoding='UTF-8')
