import opensim as osim
import numpy as np

def get_joints_spanned_by_muscle(osim_model: osim.Model, state: osim.State, osim_muscle_name: str):
	
	# Get a reference to the concrete muscle class.
	force = osim_model.getMuscles().get(osim_muscle_name)
	muscle_class = str(force.getConcreteClassName())

	# Initialise
	nCoord = osim_model.getCoordinateSet().getSize()
	muscle_coord = []
	joint_name_set = []
	exec('muscle = osim.' + muscle_class + '.safeDownCast(force)');
	
	# Iterate through coordinates, finding nonzero moment arms
	for k in range(nCoord):
		# Get a reference to a coordinate
		a_coord = osim_model.getCoordinateSet().get(k)
		
		# Get coordinate's max and min values
		rMax = a_coord.getRangeMax()
		rMin = a_coord.getRangeMin()
		rDefault = a_coord.getDefaultValue()

		# Define three points in the range to test the moment arm
		total_range = rMax - rMin

		p = []
		p.append(rMin + total_range/2)
		p.append(rMin + total_range/3)
		p.append(rMin + 2*(total_range/3))

		for i in range(len(p)): # length will be 3
			a_coord.setValue(state, p[i])

			# Compute the moment arm of the muscle for this coordinate
			moment_arm = locals()['muscle'].computeMomentArm(state, a_coord)

			# Avoid false positives due to roundoff error
			tol = 1e-6

			if abs(moment_arm) > tol:
				muscle_coord.append(k)
				break

		# Set the coordinate back to its original value
		a_coord.setValue(state, rDefault)

		# Cycle through each coordinate and get the joint associated with it.
		for u in range(len(muscle_coord)):
			# Get a reference to the coordinate
			a_coord = osim_model.getCoordinateSet().get(muscle_coord[u])

			# Get the joint attached to the coordinate
			joint = a_coord.getJoint().getName()

			if joint not in joint_name_set:
				joint_name_set.append(joint)

	return joint_name_set


