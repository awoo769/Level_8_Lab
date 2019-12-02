__author__ = "Chirstoph Reinschmidt, HPL, The University of Calgary"
__version__ = "2.0"
__date__ = "November, 1996"

import numpy as np

def distance(XYZ_marker1: np.ndarray, XYZ_marker2: np.ndarray):
	'''
	This function calculates the distance between two markers.
	Input:	XYZmarker1: [X,Y,Z] coordinates of marker 1
			XYZmarker2: [X,Y,Z] coordinates of marker 2
 			Note: The distances are calculated for each row
	Output:	distance
	
	'''

	s1, t1 = np.shape(XYZ_marker1)
	s2, t2 = np.shape(XYZ_marker2)

	if s1 != s2 or t1 != t2 or t2 != 3:
		print('The input matrices must have 3 columns and the same number of rows. Try again.')

		return
	
	tmp = np.power((XYZ_marker1 - XYZ_marker2),2)
	dist = np.power(np.sum(tmp),0.5)

	return dist

