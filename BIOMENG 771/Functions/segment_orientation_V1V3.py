
import numpy as np

def segment_orientation_V1V3(V1: np.ndarray,V3: np.ndarray):

	e1 = np.zeros((len(V1),3))
	e2 = np.zeros((len(V1),3))
	e3 = np.zeros((len(V1),3))

	for i in range(len(V1)):
		e1[i,:] = V1[i,:]/np.sqrt(np.dot(V1[i,:],V1[i,:]))
		e3[i,:] = V3[i,:]/np.sqrt(np.dot(V3[i,:],V3[i,:]))

	for i in range(len(V1)):
		e2[i,:] = np.cross(e3[i,:],e1[i,:])

	for i in range(len(V1)):
		e3[i,:] = np.cross(e1[i,:],e2[i,:])

	return e1, e2, e3