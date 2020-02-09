import opensim as osim

def get_indep_coord_and_joint(osim_model: osim.Model, constraint_coord_name: str):
	'''
	Function that given a dependent coordinate finds the independent coordinate and the 
	associated joint. The function assumes that the constraint is a CoordinateCoupleConstraint 
	as used by Arnold, Delp and LLLM. The function can be useful to manage the patellar joint 
	for instance. 

	'''

	# Initial state
	current_state = osim_model.initSystem

	# Get coordinate
	constraint_coord = osim_model.getCoordinateSet().get(constraint_coord_name)

	# Double check: if not constrained then function returns
	if not constraint_coord.isConstrained(current_state):
		print("%s is not a constrained coordinate." % (constraint_coord_name))
		return
	
	# Otherwise, search through the constraints
	for n in range(osim_model.getConstraintSet().getSize()):
		# Get current constraint
		curr_constr = osim_model.getConstraintSet().get(n)

		# This function assumes that the constraint will be a coordinate coupler contraint 
		# (Arnold's model and LLLM uses this) cast down constraint
		curr_constr_casted = osim.CoordinateCouplerConstraint().safeDownCast(curr_constr)

		# Get dep coordinate and check if it is the coordinate of interest
		dep_coord_name = curr_constr_casted.getDependentCoordinateName()

		if constraint_coord_name in dep_coord_name:
			ind_coord_name_set = curr_constr_casted.getIndependentCoordinateNames()
		
			# Extract independent coordinate and independent joint to which the coordinate refers
			if ind_coord_name_set.getSize() == 1:
				ind_coord_name = curr_constr_casted.getIndependentCoordinateNames().get(0)
				ind_coord_joint_name = osim_model.getCoordinateSet().get(ind_coord_name).getJoint().getName()

				return ind_coord_name, ind_coord_joint_name

			elif ind_coord_name_set.getSize() > 1:
				print("get_indep_coord_and_joint.py. The CoordinateCouplerConstraint has more than one independent coordinate and this is not managed by this function yet.")


