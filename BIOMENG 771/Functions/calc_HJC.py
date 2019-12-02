'''
Calculate the hip joint centre, as defined by Gamage, Lasenby J. (2002)

'''

__author__ = "Andrea Cereatti, Alex Woodall"

import numpy as np
from distance import distance

def calc_HJC(TrP: np.ndarray):
	'''
	This function calculates the hip joint centre (HJC)

	Input: TrP clean matrix containing markers'trajectories in the proximal system of reference.
           dim(TrP)=Nc*3p where Nc is number of good samples and p is the number of distal markers

	Output: Cm vector with the coordinates of hip joint center (Cx,Cy,Cz).

	Comments: metodo1b extracts HJC position as the centre of the optimal spherical suface that minimizes the root 
			  mean square error between the radius(unknown) and the distance of the centroid of marker's coordinates 
			  from sphere center(unknown). Using definition of vector differentiation is it possible to put the 
			  problem in the form: A*Cm=B that is a linear equation system

	References: Gamage, Lasenby J. (2002). 
            	New least squares solutions for estimating the average centre of rotation and the axis of rotation.
            	Journal of Biomechanics 35, 87-93 2002   
                Halvorsen correzione bias
	'''

	r, c = np.shape(TrP)
	D = np.zeros((3,3))
	V2 = []
	b1 = np.array([0, 0, 0])

	for j in range (0, c, 3):
		d1 = np.zeros((3,3))
		V2a = 0
		V3a = np.array([0, 0, 0])

		for i in range(r):

			tmp = (TrP[i,j:(j+3)])[:,np.newaxis]

			d1 = d1 + np.matmul(tmp,tmp.transpose())

			a = np.power(TrP[i,j],2) + np.power(TrP[i,j+1],2) + np.power(TrP[i,j+2],2)
			V2a = V2a + a
			V3a = V3a + a*TrP[i,j:(j+3)]

		D = D + d1/r

		V2.append(V2a/r)
		b1 = b1 + V3a/r

	# Convert V2 to array
	V2 = np.array(V2)

	V1 = np.mean(TrP,axis=0)
	
	p = np.size(V1)

	e1 = 0
	E = np.zeros((3,3))
	f1 = np.array([0, 0, 0])
	F = np.array([0, 0, 0])

	for k in range(0,p,3):

		tmp = (V1[k:(k+3)])[:,np.newaxis]

		e1 = np.matmul(tmp,tmp.transpose())
		E = E + e1
		f1 = V2[int(k/3)] * V1[k:(k+3)]
		F = F + f1

	# Equation 5 of Gamage and Lasenby
	A = 2 * (D - E)
	B = (np.transpose(b1 - F))[:,np.newaxis]
	U, S, V = np.linalg.svd(A)

	# Convert S to a diagonal matrix
	S = np.diag(S)
	V = np.transpose(V)

	Cm_in = np.matmul(np.matmul(V,np.linalg.inv(S)), np.matmul(np.transpose(U), B))
	Cm_old = Cm_in + (np.transpose(np.array([1, 1, 1])))[:,np.newaxis]

	while distance(np.transpose(Cm_old), np.transpose(Cm_in)) > 0.0000001:
		Cm_old = Cm_in
		sigma2 = []

		for j in range(0,c,3):
			marker = TrP[:,j:(j+3)]
			Ukp = marker - np.transpose(Cm_in * np.ones((1,r)))

			# Computation of u^2
			u2 = 0
			app = []

			for i in range(r):
				u2 = u2 + np.matmul(Ukp[i,:], np.transpose(Ukp[i,:]))
				app.append(np.matmul(Ukp[i,:], np.transpose(Ukp[i,:])))
			
			u2 = u2/r

			# Computation of sigma
			sigmaP = 0

			for i in range(r):
				sigmaP = sigmaP + np.power((app[i] - u2), 2)
			
			sigmaP = sigmaP/(4 * u2 * r)
			sigma2.append(sigmaP)

		sigma2 = np.mean(sigma2)

		# Computation of deltaB
		deltaB = 0

		for j in range(0,c,3):
			deltaB = deltaB + (np.transpose(V1[j:(j+3)]))[:,np.newaxis] - Cm_in
		
		deltaB = 2 * sigma2 * deltaB

		Bcorr = B - deltaB # Corrected term B
		
		# Iterative estimation of the centre of rotation
		Cm_in = np.matmul(np.matmul(V,np.linalg.inv(S)), np.matmul(np.transpose(U), Bcorr))
	
	Cm = Cm_in

	return Cm
	