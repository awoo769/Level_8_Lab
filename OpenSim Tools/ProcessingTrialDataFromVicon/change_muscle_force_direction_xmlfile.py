import numpy as np
from xml.dom import minidom

def change_muscle_force_direction_xmlfile(force_filename: str, trial: str, model: str, directory: str, time__range: list, cut_off_freq: int):
	'''
	NOTE: Any attribute not changed within this function must be changed in the original template 
	in "xmlTemplates"

	Inputs:	force_filename: full filename for the template muscle force direction setup xml file
			trial: trial name, e.g.,  "_12Mar_ss_12ms_01"
			model: model name, e.g., "AB08"
			directory: output directory name
			time_range: start and end times
			cuf_off_frequency: low pass cut-off frequency

	'''

	doc_node = minidom.parse(force_filename)

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

	''' Muscle force direction settings '''

	muscle_force_direction = objects_set_child.getElementsByTagName("MuscleForceDirection")
	muscle_force_direction_child = muscle_force_direction.item(0)

	muscle_force_direction_start_time = muscle_force_direction_child.getElementsByTagName("start_time")
	muscle_force_direction_start_time_child = muscle_force_direction_start_time.item(0)

	muscle_force_direction_end_time = muscle_force_direction_child.getElementsByTagName("end_time")
	muscle_force_direction_end_time_child = muscle_force_direction_end_time.item(0)

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

	muscle_force_direction_start_time_child.firstChild.data = starttime
	muscle_force_direction_end_time_child.firstChild.data = endtime
	
	''' Write file '''
	
	new_filename = directory + "\\" + model + "\\" + trial + "\\" + trial + force_filename.split("\\")[-1]

	with open(new_filename, 'w') as xml_file:
		doc_node.writexml(xml_file, addindent='\t', newl='\n', encoding='UTF-8')
