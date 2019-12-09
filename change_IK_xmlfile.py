import numpy as np
from xml.dom import minidom

def change_IK_xmlfile(ik_filename: str, trial: str, model: str, directory: str, time__range: list, good_marker_names: list, bad_marker_names: list):
	'''
	Rewrites the IK setup xml file for a new trial

	'''

	doc_node = minidom.parse(ik_filename)

	''' Get Hierarchy Access '''
	IK_tool = doc_node.getElementsByTagName("InverseKinematicsTool")
	IK_tool_child = IK_tool.item(0)

	res_directory = IK_tool_child.getElementsByTagName("results_directory")
	res_directory_child = res_directory.item(0)

	input_directory = IK_tool_child.getElementsByTagName("input_directory")
	input_directory_child = input_directory.item(0)

	model_file = IK_tool_child.getElementsByTagName("model_file")
	model_file_child = model_file.item(0)

	IK_task_set = IK_tool_child.getElementsByTagName("IKTaskSet")
	IK_task_set_child = IK_task_set.item(0)

	IK_task_set_objects = IK_task_set_child.getElementsByTagName("objects")
	IK_task_set_objects_child = IK_task_set_objects.item(0)

	IK_marker_tasks = IK_task_set_objects_child.getElementsByTagName("IKMarkerTask")
	num_markers = IK_marker_tasks.length

	marker_file = IK_tool_child.getElementsByTagName("marker_file")
	marker_file_child = marker_file.item(0)

	time_range = IK_tool_child.getElementsByTagName("time_range")
	time_range_child = time_range.item(0)

	output_motion_file = IK_tool_child.getElementsByTagName("output_motion_file")
	output_motion_file_child = output_motion_file.item(0)

	''' Set new directory, filenames, and number inputs '''

	res_directory_child.firstChild.data = ".\\"
	input_directory_child.firstChild.data = ".\\"

	# OpenSim model name
	model_file_name = directory + "\\" + model + "\\" + trial + "\\" + model + ".osim"
	model_file_child.firstChild.data = model_file_name
	
	# Hardcore input trc
	marker_file_name = directory + "\\" + model + "\\" + trial + "\\" + trial + ".trc"
	marker_file_child.firstChild.data = marker_file_name

	# Plain output name (for local results)
	output_file_name = trial + "IKResults.mot"
	output_motion_file_child.firstChild.data = output_file_name

	# Time range
	timerange = list(map(str, time__range))

	time_range_child.firstChild.data = timerange

	''' Remove any absent markers, set weighting for bony landmarks (anatomical markers) '''

	bony_landmarks = ['LMMAL','RMMAL','LLMAL','RLMAL','LASI','RASI','LPSI','RPSI']
	
	for i in range(num_markers):
		current_marker = IK_marker_tasks.item(i)
		current_marker_name = current_marker.getAttribute('name')
		apply = current_marker.getElementsByTagName('apply')
		apply_child = apply.item(0)
		weight = current_marker.getElementsByTagName('weight')
		weight_child = weight.item(0)

		if (current_marker_name in good_marker_names) and (current_marker_name not in bad_marker_names):
			apply_child.firstChild.data = 'true'
		else:
			apply_child.firstChild.data = 'false'
		
		if current_marker_name in bony_landmarks:
			weight_child.firstChild.data = '10'
		else:
			weight_child.firstChild.data = '1'
	
	new_filename = directory + "\\" + model + "\\" + trial + "\\" + trial + ik_filename.split("\\")[-1]

	with open(new_filename, 'w') as xml_file:
		doc_node.writexml(xml_file)
