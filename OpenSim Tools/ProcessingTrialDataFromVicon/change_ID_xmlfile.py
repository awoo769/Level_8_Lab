import numpy as np
from xml.dom import minidom

def change_ID_xmlfile(id_filename: str, trial: str, model: str, directory: str, time__range: list, cut_off_freq: int):
	'''
	Rewrites the ID setup xml file for a new trial

	'''

	doc_node = minidom.parse(id_filename)

	''' Get Hierarchy Access '''
	ID_tool = doc_node.getElementsByTagName("InverseDynamicsTool")
	ID_tool_child = ID_tool.item(0)

	res_directory = ID_tool_child.getElementsByTagName("results_directory")
	res_directory_child = res_directory.item(0)

	input_directory = ID_tool_child.getElementsByTagName("input_directory")
	input_directory_child = input_directory.item(0)

	model_file = ID_tool_child.getElementsByTagName("model_file")
	model_file_child = model_file.item(0)

	time_range = ID_tool_child.getElementsByTagName("time_range")
	time_range_child = time_range.item(0)

	ex_loads_file = ID_tool_child.getElementsByTagName("external_loads_file")
	ex_loads_file_child = ex_loads_file.item(0)

	coords_file = ID_tool_child.getElementsByTagName("coordinates_file")
	coords_file_child = coords_file.item(0)

	filter_frequency = ID_tool_child.getElementsByTagName("lowpass_cutoff_frequency_for_coordinates")
	filter_frequency_child = filter_frequency.item(0)

	output_gen_force_file = ID_tool_child.getElementsByTagName("output_gen_force_file")
	output_gen_force_file_child = output_gen_force_file.item(0)

	''' Set new directory, filenames, and number inputs '''

	ID_tool_child.setAttribute('name', model)

	# Local directory
	res_directory_child.firstChild.data = ".\\"
	input_directory_child.firstChild.data = ".\\"

	# OpenSim model name
	model_file_name = directory + "\\" + model + "\\" + trial + "\\" + model + ".osim"
	model_file_child.firstChild.data = model_file_name
	
	# Time range
	timerange = ' '.join(list(map(str, time__range)))
	time_range_child.firstChild.data = timerange

	external_loads_file = trial + 'ExternalLoads.xml'
	ex_loads_file_child.firstChild.data = external_loads_file

	coordsfile = trial + 'IKResults.mot'
	coords_file_child.firstChild.data = coordsfile

	filter_frequency_child.firstChild.data = str(cut_off_freq)

	# Plain output name (for local results)
	output_file_name = trial + "IDResults.sto"
	output_gen_force_file_child.firstChild.data = output_file_name
	
	new_filename = directory + "\\" + model + "\\" + trial + "\\" + trial + id_filename.split("\\")[-1]

	with open(new_filename, 'w') as xml_file:
		doc_node.writexml(xml_file, addindent='\t', newl='\n', encoding='UTF-8')
