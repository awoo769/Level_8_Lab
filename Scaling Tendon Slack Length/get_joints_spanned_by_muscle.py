import opensim as osim
import numpy as np

def get_joints_spanned_by_muscle(osim_model: osim.Model, osim_muscle_name: str):
	
	# Convert to string just in case muscle name is given as a java string
	muscle_name = str(osim_muscle_name)

	# Useful initialisations
	body_set = osim_model.getBodySet()
	muscle = osim_model.getMuscles().get(muscle_name)

	# Extracting the path point set via geometry path
	muscle_path = muscle.getGeometryPath()
	muscle_path_point_set = muscle_path.getPathPointSet()

	# Get the attachment bodies
	muscle_attach_bodies = []
	muscle_attach_index = []

	for n_point in range(muscle_path_point_set.getSize()):
		# Get the current muscle point
		current_attach_body = str(muscle_path_point_set.get(n_point).getBodyName())

		# Initialise
		if n_point == 0:
			previous_attach_body = current_attach_body
			muscle_attach_bodies.append(current_attach_body)
			muscle_attach_index.append(body_set.getIndex(current_attach_body))

		# Building vectors of the bodies attached to the muscles
		if current_attach_body != previous_attach_body:
			muscle_attach_bodies.append(current_attach_body)
			muscle_attach_index.append(body_set.getIndex(current_attach_body))
			previous_attach_body = current_attach_body

	# From distal body checking the joint names going up until the desired opensim joint name is
	# found or the proximal body is reached as parent body
	distal_body_name = muscle_attach_bodies[-1]
	body_name = distal_body_name
	
	proximal_body_name = muscle_attach_bodies[0]
	body = body_set.get(distal_body_name)

	spanned_joint_name_old = ''
	no_dof_joint_set_name = []
	joint_name_set = []

	joint_set = osim_model.getJointSet()

	# If there is more than one body attached to the body (which there should be)
	if len(muscle_attach_bodies) > 1:
		counter = 1
		next_proximal_body_name = muscle_attach_bodies[-1 - counter]

	# If not, body_name == proximal_body_name and the while loop will not be entered

	while body_name != proximal_body_name:

		# Work around because Body.getJoint() has been removed in OpenSim 4.0
		for joint in joint_set:
			# Parent body of joint is the more proximal body to the current body
			if next_proximal_body_name in joint.getParentFrame().getName():
				spanned_joint = joint
				spanned_joint_name = str(spanned_joint.getName())

		if spanned_joint_name == spanned_joint_name_old:
			# Get the parent body. Parent frame naming in the form e.g., pelvis_offset
			parent_name = spanned_joint.getParentFrame().getName().rsplit('_',1)[0] # Split 1 from the end
			body = body_set.get(parent_name)
			spanned_joint_name_old = spanned_joint

		else:
			if spanned_joint.numCoordinates() != 0:
				joint_name_set.append(spanned_joint_name)
			else:
				no_dof_joint_set_name.append(spanned_joint_name)
			
			spanned_joint_name_old = spanned_joint_name
			parent_name = spanned_joint.getParentFrame().getName().rsplit('_',1)[0] # Split 1 from the end
			body = body_set.get(parent_name)

		body_name = str(body.getName())
		next_proximal_body_name = muscle_attach_bodies[-1 - counter]

	if len(joint_name_set) == 0:
		print('No joint detected for muscle %s' % (muscle_name))
	
	if len(no_dof_joint_set_name) != 0:
		for n_v in range(no_dof_joint_set_name):
			print('Joint %s has no DoF.' % (no_dof_joint_set_name[n_v]))
	
	varargout = no_dof_joint_set_name

	return joint_name_set


