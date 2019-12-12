import numpy as np
import opensim as osim
from xml.dom import minidom

def setup_scale_xml(scale_filename: str, trial: str, model: str, output_directory: str, input_directory: str):
	'''
	Rewrites the scale setup xml file for a new trial

	Inputs:	scale_filename: full filename for the template inverse kinematics setup xml file
			trial: trial name, e.g.,  "_12Mar_ss_12ms_01"
			model: model name, e.g., "AB08"
			directory: output directory name

	References: Megan Schroeder, 11/01/2014

	'''
	
	# Static trc filename
	static_trc = input_directory + "\\" + model + trial + "_Static.trc"

	# Create marker_data object to read starting and ending times from trc file
	marker_data = osim.MarkerData(static_trc)

	time_range = osim.ArrayDouble()
	time_range.setitem(0, marker_data.getStartFrameTime())
	time_range.setitem(1, marker_data.getLastFrameTime())

	# Create scale_tool object
	scale_tool = osim.ScaleTool(scale_filename)
	scale_tool.setName(model)

	# Modify top-level properties

	# TODO, get height (optional), age (optional) and mass (required) of subject
	scale_tool.setPathToSubject(model)
	#scale_tool.setSubjectMass()
	#scale_tool.setSubjectHeight()
	#scale_tool.setSubjectAge

	# Update generic_model_maker
	generic_model_marker = scale_tool.getGenericModelMaker()
	generic_model_marker.setModelFileName(model + ".osim")
	generic_model_marker.setMarkerSetFileName(model + '_Scale_MarkerSet.xml')

	# Update model_scaler
	model_scaler = scale_tool.getModelScaler()
	model_scaler.setApply(True)

	scale_order = osim.ArrayStr()
	scale_order.setitem(0,'Measurements')

	model_scaler.setScalingOrder(scale_order)
	measurements_set = model_scaler.getMeasurementSet()
	measurements_set.assign(osim.MeasurementSet().makeObjectFromFile(model + '_Scale_MarkerSet.xml'))
	model_scaler.setMarkerFileName(static_trc)
	model_scaler.setTimeRange(time_range)
	model_scaler.setPreserveMassDist(True)

	output_filename = trial + 'TempScaled.osim' 
	model_scaler.setOutputModelFileName(output_filename)

	ouput_scale_filename = trial + '_ScaleSet.xml'
	model_scaler.setOutputScaleFileName(ouput_scale_filename)

	# Update MarkerPlacer
	
