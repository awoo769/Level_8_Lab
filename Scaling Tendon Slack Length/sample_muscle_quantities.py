import opensim as osim
from get_joints_spanned_by_muscle_v2 import get_joints_spanned_by_muscle
from get_indep_coord_and_joint import get_indep_coord_and_joint
import numpy as np
import math

def sample_muscle_quantities(osim_model: osim.Model, osim_muscle: osim.Muscle, muscle_quant: str, N_eval_points: int):
	
	''' Settings '''

	# Limit (1) or not (0) the discretisation of the joint space sampling
	limit_discr = 0

	# Minimum angular discretisation
	min_increm_in_deg = 2.5

	''' Initialise the model '''
	current_state = osim_model.initSystem()

	# Getting the joint crossed by a muscle
	mus_name = osim_muscle.getName()
	muscle_crossed_joint_list = get_joints_spanned_by_muscle(osim_model, current_state, mus_name)
	
	# Index for effective DoFs
	n_dof = 0
	dof_index = []

	coordinate_boundaries = []
	deg_increm = []

	for n_joint in range(len(muscle_crossed_joint_list)):
		# Current joint
		curr_joint = muscle_crossed_joint_list[n_joint]

		# Initial estimation of the nr of Dof of the CoordinateSet for that joint before 
		# checking for locked and constraint dofs.
		joint = osim_model.getJointSet().get(curr_joint)

		nDOF = joint.numCoordinates()
		
		# Skip welded joint and removes welded joint from muscleCrossedJointSet
		if nDOF == 0:
			continue

		# Calculating effective dof for that joint
		effect_dof = nDOF

		for n_coord in range(nDOF):
			# Get coordinate
			curr_coord = joint.get_coordinates(n_coord)
			curr_coord_name = curr_coord.getName()

			# Skip if locked
			if curr_coord.getLocked(current_state):
				continue

			# If coordinate is constrained then the independent coordinate and associated joint
			# will be listed in the sampling "map"
			if curr_coord.isConstrained(current_state):
				constraint_coord_name = curr_coord_name

				# Finding the independent coordinate
				ind_coord_name, ind_coord_joint_name = get_indep_coord_and_joint(osim_model, constraint_coord_name)
			
				# Updating the name to be saved in the list
				curr_coord_name = ind_coord_name
				effect_dof = effect_dof - 1

				# Ignoring constrained dof if they point to an independent coordinate that has
				# already been stored
				if osim_model.getCoordinateSet().getIndex(curr_coord_name) in dof_index:
					continue

				# Skip dof if independent coordinate locked (the coord correspondent to the name
				# needs to be extracted)
				if osim_model.getCoordinateSet().get(curr_coord_name).getLocked(current_state):
					continue

			# NB: dof_index is used later in the string generated code.
			# CRUCIAL: the index of dof now is model based ("global") and different from the 
			# joint based used until now
			dof_index.append(osim_model.getCoordinateSet().getIndex(curr_coord_name))

			# Necessary update/reload the curr_coord to avoid problems with the dependent coordinates
			curr_coord = osim_model.getCoordinateSet().get(dof_index[n_dof])

			# Getting the values defining the range
			joint_range = []
			joint_range.append(curr_coord.getRangeMin())
			joint_range.append(curr_coord.getRangeMax())

			# Storing range of motion conveniently
			coordinate_boundaries.append(joint_range)

			# Increments in the variables when sampling the mtl space. Increments are different
			# for each dof and based on N_eval.

			# Defining the increments
			deg_increm.append((joint_range[-1] - joint_range[0]) / (N_eval_points - 1))

			# Limit or not the discretisation of the joint space sampling
			if limit_discr == 1:
				# A limit to the increase can be set though
				if deg_increm[n_dof] < min_increm_in_deg / 180 * math.pi:
					deg_increm[n_dof] = min_increm_in_deg / 180 * math.pi
				
			
			# Updating list index
			n_dof = n_dof + 1

	# Initialise the counter to save the results
	counter = 1

	# Assign an initial and a final value variable for each dof X, calling them
	# set_angle_start_dof and set_limit_dof respectively
	
	set_angle_start_dof = []
	set_limit_dof = []

	for n_instr in range(len(dof_index)):
		# Setting up variables
		set_angle_start_dof.append(coordinate_boundaries[n_instr][0])
		set_limit_dof.append(coordinate_boundaries[n_instr][-1])
	
	# setting up for loops in order to explore all the possible combination of joint angles 
	# (looping on all the dofs of each joint for all the joint crossed by the muscle).
	# The model pose is updated via: " coordToUpd.setValue(currentState,setAngleDof)".
	# The right dof to update is chosen via: "coordToUpd = osimModel.getCoordinateSet.get(n_instr)"
	
	# Initialise nested for loops
	string_to_execute_1 = ''

	for n_instr in range(len(dof_index)):
		tabs = ''
		start_tab = ''
		for _ in range(n_instr+1):
			tabs = tabs + '\t'
		for _ in range(n_instr):
			start_tab = start_tab + '\t'
			
		string_to_execute_1 = (string_to_execute_1 + start_tab + 'for set_angle_dof' + str(n_instr+1) + 
		' in np.arange(set_angle_start_dof['+ str(n_instr) + '], set_limit_dof['+ str(n_instr) + ']+0.01,deg_increm['
		+ str(n_instr) + ']):\n' + tabs + 'coord_to_upd = osim_model.getCoordinateSet().get(dof_index['+ 
		str(n_instr) + '])\n' + tabs + 'coord_to_upd.setValue(current_state, set_angle_dof' + str(n_instr+1) + ')\n\n')

	# Calculate muscle length for the muscle
	if muscle_quant == 'MTL':
		muscle_output = []
		string_to_execute_2 = (tabs + 'muscle_output.append(osim_muscle.getGeometryPath().getLength(current_state))')

	elif muscle_quant == 'LfirNorm':
		muscle_output = []
		string_to_execute_2 = (tabs + 'osim_muscle.setActivation(current_state, 1.0)\n' + 
		tabs + 'osim_model.equilibrateMuscles(current_state)\n' + 
		tabs + 'muscle_output.append(osim_muscle.getNormalizedFiberLength(current_state))')
		
	elif muscle_quant == 'Lten':
		muscle_output = []
		string_to_execute_2 = (tabs + 'osim_muscle.setActivation(current_state, 1.0)\n' + 
		tabs + 'osim_model.equilibrateMuscles(current_state)\n' + 
		tabs + 'muscle_output.append(osim_muscle.getTendonLength(current_state))')

	elif muscle_quant == 'Ffib':
		muscle_output = []
		string_to_execute_2 = (tabs + 'osim_muscle.setActivation(current_state, 1.0)\n' + 
		tabs + 'osim_model.equilibrateMuscles(current_state)\n' + 
		tabs + 'muscle_output.append(osim_muscle.getActiveFiberForce(current_state))')

	elif muscle_quant == 'all':
		# Set mus_name to have length 5
		muscle_output = [[] for i in range(5)]
		string_to_execute_2 = (tabs + 'osim_muscle.setActivation(current_state, 1.0)\n' + 
		tabs + 'osim_model.equilibrateMuscles(current_state)\n' + 
		tabs + 'muscle_output[0].append(osim_muscle.getGeometryPath().getLength(current_state))\n' + 
		tabs + 'muscle_output[1].append(osim_muscle.getNormalizedFiberLength(current_state))\n' + 
		tabs + 'muscle_output[2].append(osim_muscle.getTendonLength(current_state))\n' + 
		tabs + 'muscle_output[3].append(osim_muscle.getActiveFiberForce(current_state))\n' + 
		tabs + 'muscle_output[4].append(osim_muscle.getPennationAngle(current_state))')

	# Excecute strings to get muscle_output
	exec(string_to_execute_1 + string_to_execute_2)
 
	return muscle_output