import numpy as np
import opensim as osim
from xml.dom import minidom

def setup_IK_xml(ik_filename: str, trial: str, model: str, directory: str, time_range: list, marker_names: list):
	'''
	Rewrites the IK setup xml file for a new trial

	Inputs:	ik_filename: full filename for the template inverse kinematics setup xml file
			trial: trial name, e.g.,  "_12Mar_ss_12ms_01"
			model: model name, e.g., "AB08"
			directory: output directory name
			time_range: start and end times
			marker_names: list of the markers names which we are using

	'''

	# Create an instance of the inverse kinematics tool
	IK_tool = osim.InverseKinematicsTool(ik_filename)

	# Set name
	IK_tool.setName(model)

	# Set the time range, NOTE: Must be a double (np.float64)
	IK_tool.setStartTime(np.float64(time_range[0]))
	IK_tool.setEndTime(np.float64(time_range[-1]))

	# Set the marker file
	marker_file_name = trial + ".trc"
	IK_tool.setMarkerDataFileName(marker_file_name)

	# Set the coordinate file
	coordinate_file_name = ''
	IK_tool.setCoordinateFileName(coordinate_file_name)

	# Set the output motion file
	output_file_name = trial + "IKResults.mot"
	IK_tool.setOutputMotionFileName(output_file_name)

	# Set the input and results directory
	IK_tool.setInputsDir(".\\")
	IK_tool.setResultsDir(".\\")

	''' Remove any absent markers, set weighting for bony landmarks (anatomical markers) '''

	# List of bony anatomical landmarkers to give high weighting
	bony_landmarks = ['LMMAL','RMMAL','LLMAL','RLMAL','LASI','RASI','LPSI','RPSI']

	for marker in IK_tool.getIKTaskSet():
		# markers.getName() is the name of each marker
		current_marker = marker.getName()

		# If the marker in the inverse kinematics tool is one which we are using
		if (current_marker in marker_names):
			marker.setApply(True)
		else:
			marker.setApply(False)

		if current_marker in bony_landmarks:
			marker.setWeight(10)
		else:
			marker.setWeight(1)

	''' Write changes to an XML setup file '''

	xml_setup_path = directory + "\\" + model + "\\" + trial + "\\" + trial + ik_filename.split("\\")[-1]
	IK_tool.printToXML(xml_setup_path)

	''' Temporary fix for setting model name using XML parsing '''

	dom = minidom.parse(xml_setup_path)
	dom.getElementsByTagName("model_file")[0].firstChild.nodeValue = model + ".osim"

	
	with open(xml_setup_path, 'w') as xml_file:
		dom.writexml(xml_file, addindent='\t', newl='\n', encoding='UTF-8')
