import numpy as np
from xml.dom import minidom

def change_load_xmlfile(load_filename: str, trial: str, model: str, directory: str, cut_off_freq: int):
	'''
	Takes the external loads xml file specified by load_filename, and sets the .mot files to those specified
	by trial, and writes to a new file of the two strings combined, i.e., "Walk1ExternalLoads.xml"

	'''

	doc_node = minidom.parse(load_filename)
	mot_string = directory + "\\" + model + "\\" + trial + "\\" + trial + ".mot"

	''' Get Hierarchy Access '''
	exL = doc_node.getElementsByTagName("ExternalLoads")
	exL_child = exL.item(0)

	data_file = exL_child.getElementsByTagName("datafile")
	data_file_child = data_file.item(0)

	exL_model = exL_child.getElementsByTagName("external_loads_model_kinematics_file")
	exL_model_child = exL_model.item(0)

	cut_off = exL_child.getElementsByTagName("lowpass_cutoff_frequency_for_load_kinematics")
	cut_off_child = cut_off.item(0)

	''' Write new ones '''

	data_file_child.firstChild.data = mot_string
	cut_off_child.firstChild.data = str(cut_off_freq)
	
	''' Write new file '''

	new_filename = directory + "\\" + model + "\\" + trial + "\\" + trial + load_filename.split("\\")[-1]

	with open(new_filename, 'w') as xml_file:
		doc_node.writexml(xml_file, addindent='\t', newl='\n', encoding='UTF-8')
