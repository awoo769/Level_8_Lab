import numpy as np
import opensim as osim
from xml.dom import minidom

def setup_IK_xml(trial: str, model: str, directory: str, time_range: list, marker_names: list):
	'''
	Rewrites the IK setup xml file for a new trial
	
	Inputs:	trial: trial name, e.g.,  "_12Mar_ss_12ms_01"
			model: model name, e.g., "AB08"
			directory: output directory name
			time_range: start and end times
			marker_names: list of the markers names which we are using

	'''

	# Create an instance of the inverse kinematics tool
	IK_tool = osim.InverseKinematicsTool()

	# Set the name of the tool
	IK_tool.setName(model)

	# Set the input and results directory
	IK_tool.setInputsDir(directory + "\\" + model + "\\" + trial)
	IK_tool.setResultsDir(directory + "\\" + model + "\\" + trial)

	# Set the time range, NOTE: Must be a double (np.float64)
	IK_tool.setStartTime(np.float64(time_range[0]))
	IK_tool.setEndTime(np.float64(time_range[-1]))

	# Set the marker file
	marker_file_name = directory + "\\" + model + "\\" + trial + "\\" + trial + ".trc"
	IK_tool.setMarkerDataFileName(marker_file_name)

	# Set the coordinate file
	coordinate_file_name = ''
	IK_tool.setCoordinateFileName(coordinate_file_name)

	# Set the output motion file
	output_file_name = trial + "IKResults.mot"
	IK_tool.setOutputMotionFileName(output_file_name)

	''' Add markers and set weighting '''

	# List of bony anatomical landmarkers to give high weighting
	bony_landmarks = ['LMMAL','RMMAL','LLMAL','RLMAL','LASI','RASI','LPSI','RPSI']

	# Create IKTaskSet
	IK_task_set = IK_tool.getIKTaskSet()

	# Assign markers and weights
	for marker in marker_names:
		IK_marker_task = osim.IKMarkerTask()
		IK_marker_task.setName(marker)
		
		if marker in bony_landmarks:
			IK_marker_task.setApply(True)
			IK_marker_task.setWeight(10)
		else:
			IK_marker_task.setApply(True)
			IK_marker_task.setWeight(1)
			
		IK_task_set.cloneAndAppend(IK_marker_task)

	''' Write changes to an XML setup file '''

	xml_setup_path = directory + "\\" + model + "\\" + trial + "\\" + trial + "IKSetup.xml"
	IK_tool.printToXML(xml_setup_path)

	''' Temporary fix for setting model name using XML parsing '''

	dom = minidom.parse(xml_setup_path)
	dom.getElementsByTagName("model_file")[0].firstChild.nodeValue = directory + "\\" + model + "\\" + model + ".osim"

	
	with open(xml_setup_path, 'w') as xml_file:
		dom.writexml(xml_file, addindent='\t', newl='\n', encoding='UTF-8')
