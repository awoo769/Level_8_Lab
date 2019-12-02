
import numpy as np

def coord_tr(org: np.ndarray, dist: np.ndarray, prox: np.ndarray, med: np.ndarray, lat: np.ndarray, xyz: np.ndarray, f: int):

	'''
	The input to the function coord_tr are 5 anatomical points, expressed in the global coordinate system
	(or force plate coordinate system), an nx3 vector matrix (xyz; e.g 3D force vector), expressed in the 
	the same coordinate system, and a variable (f) defining whether to use the long axis of a segment or 
	the flexion/extension axis of a segment as the first defining axis (**see below).

	The function defines a local cartesion (right handed) coordinate system from the 5 anatomical points 
	and the transformation matrix between it and the global coordinate system.  The output is an nx3 matrix 
	of the vector xyz transformed (rotated and translated) in the local coordinate system (xyz_tr).

	NB: the local coordinate system can be constructed from only three points.  If this is the case, the 
	same point can be entered into the function twice (e.g. the org may also be the dist point).  However, 
	five points are included because axes of coordinate systems are often defined using points common only 
	to one axis.

	Reference:  Craig, J.J. (1989).  Introduction to Robotics: mechanics and control.  
	Addison-Wesley Publishing Company, Inc., Reading, MA.

	FEB 01, 2006.  J. Rubenson

	'''

	'''
	Initially, the first and second defining vectors of the local coordinate system are defined.
	The first defining vector is usually the vector between adjacent joint centers (long axis).  
	Here it is the vector between dist and prox.
	The second defining vector is usually the flexion/extension axis.  Here it is the vector between med 
	and lat.

	This code assumes the right limb is being analyzed. If the left limb is analysed swap the med and lat 
	points in the input.

	** Sometimes it is more suitable to use the flex/ext axis as the first defining axis.  This may be the 
	case if the final flex/ext axis after the coordinate system has been constructed is very different 
	from the true flexion/extension axis.  Use the flag f to define whether the long axis is the first 
	defining axis (1) or if the flexion/extension axis is the first defining axis (2)

	'''

	if f == 1:
		org					 # The origin of the local coordinate system.  Usually the distal joint center of the segment, but can be the proximal joint.
		y_axis = prox - dist # The first defining axis.  Usually the long axis.
		z_axis = med - lat   # The second defining axis.  Usually the flex/ext axis.  med = medial point, lat = lateral point

		# Create unit vectors for the local coordinate system.
		# The unit vercor is the normal vector divided by itself and describes the sense of the vector in space.
		# The individual elements of the unit vector are the direction cosines of the vector, which is the cosine of the angle between the vector and
		#each of the axes of the 3D global coordinate system.
		
		#Since the dot product between two vectors is equal to the length of each vector multiplied together, and this multiplied by the cosine of the angle between
		# the 2 vectors, the dod product is a convinient way to calculate the vector length.

		e1 = np.zeros((len(xyz),3))
		e2 = np.zeros((len(xyz),3))
		e3 = np.zeros((len(xyz),3))

		for i in range(len(xyz)):
			e1[i,:] = y_axis[i,:]/np.sqrt(np.dot(y_axis[i,:],y_axis[i,:])) # long axis unit vector
			e2[i,:] = z_axis[i,:]/np.sqrt(np.dot(z_axis[i,:],z_axis[i,:])) # unit vector of second defining axis

		for i in range(len(xyz)):
			e3[i,:] = np.cross(e2[i,:],e1[i,:]) # vector orthogonal to long axis unit vector and unit vector of second defining axis
												# (adduction/abduction axis)

		for i in range(len(xyz)):
			e3[i,:] = e3[i,:]/np.sqrt(np.dot(e3[i,:],e3[i,:])) # adduction/abduction unit vector

		for i in range(len(xyz)):
			e2[i,:] = np.cross(e3[i,:],e1[i,:]) # unit vector orthogonal to long unit vector and adduction/abduction unit vector
												# i.e. flexion/extension axis
		
		# Create a rotation matrix for the local coordinate system relative to the global coordinate system
		# The rotation matrix is a 3x3 matrix of the three unit vectors (or 12 direction cosines).

		rotation_matrix = np.zeros((len(xyz), 3, 3))

		for i in range(len(xyz)):
			rotation_matrix[i,:,:] = np.array([e3[i,:], e1[i,:], e2[i,:]])

	if f == 2:
		org					 # The origin of the local coordinate system.  Usually the distal joint center of the segment, but can be the proximal joint.
		z_axis = lat - med # The first defining axis. In this case the flex/ext axis. med = medial point, lat = lateral point
		y_axis = prox - dist   # The second defining axis. In this case the long axis.

		e1 = np.zeros((len(xyz),3))
		e2 = np.zeros((len(xyz),3))
		e3 = np.zeros((len(xyz),3))

		for i in range(len(xyz)):
			e1[i,:] = z_axis[i,:]/np.sqrt(np.dot(z_axis[i,:],z_axis[i,:])) # flex/ext unit vector
			e2[i,:] = y_axis[i,:]/np.sqrt(np.dot(y_axis[i,:],y_axis[i,:])) # unit vector of second defining axis

		for i in range(len(xyz)):
			e3[i,:] = np.cross(e2[i,:],e1[i,:]) # vector orthogonal to long axis unit vector and unit vector of second defining axis
												# (adduction/abduction axis)

		for i in range(len(xyz)):
			e3[i,:] = e3[i,:]/np.sqrt(np.dot(e3[i,:],e3[i,:])) # adduction/abduction unit vector

		for i in range(len(xyz)):
			e2[i,:] = np.cross(e1[i,:],e3[i,:]) # unit vector orthogonal to flex/ext unit vector and adduction/abduction unit vector
												# i.e. long axis
		
		# Create a rotation matrix for the local coordinate system relative to the global coordinate system
		# The rotation matrix is a 3x3 matrix of the three unit vectors (or 12 direction cosines).

		rotation_matrix = np.zeros((len(xyz), 3, 3))

		for i in range(len(xyz)):
			rotation_matrix[i,:,:] = np.array([e3[i,:], e2[i,:], e1[i,:]])

	# Define the translation vector
	tr_vector = np.transpose(org) * -1

	for i in range(len(xyz)):
		tr_vector[:,i] = np.matmul(rotation_matrix[i,:,:], tr_vector[:,i])

	# Define the transformation matrix
	transformation_matrix = np.zeros((len(xyz), 4, 4))

	for i in range(len(xyz)):
		tmp = (tr_vector[:,i])[:, np.newaxis]
		tmp2 = np.concatenate((rotation_matrix[i,:,:],tmp),axis=1)
		bottom = (np.array([0.0, 0.0, 0.0, 1.0]))[:, np.newaxis]

		transformation_matrix[i,:,:] = np.concatenate((tmp2, bottom.T))

	# Transform the vector xyz into the local coordinate system

	xyz = np.concatenate((xyz,np.ones(len(xyz))[:,np.newaxis]),axis=1).T # inverse vector xyz and append a "1" to each element
	
	length = np.shape(xyz)[-1]
	
	xyz_tr = np.zeros((np.shape(xyz)))

	for i in range(length):
		xyz_tr[:,i] = np.matmul(transformation_matrix[i,:,:],xyz[:,i]) # multiply new xyz vector by transformation matrix

	xyz_tr = xyz_tr.T # invert the vector back to an nx4 vector
	xyz_tr = xyz_tr[:,0:3] # delete the '1' column

	return xyz_tr
