import opensim as osim
import os
from diary import Diary # Need to install diary - pip install diary
import numpy as np
import scipy.linalg
import scipy.optimize
from sample_muscle_quantities import sample_muscle_quantities

def optimal_muscle_parameters(osim_model_ref_filepath: str, osim_model_target_filepath: str, N_eval: int, log_folder: str):

	# Results file identifier
	results_file_id_exp = '_N' + str(N_eval)

	# Import models
	osim_model_ref = osim.Model(osim_model_ref_filepath)
	osim_model_targ = osim.Model(osim_model_target_filepath)

	# Model details
	name, ext = os.path.split(osim_model_target_filepath)[-1].split('.')

	# Assigning a new name to the model
	osim_model_opt_name = name + '_opt' + results_file_id_exp + "." + ext
	osim_model_targ.setName(osim_model_opt_name)

	# Initialising log file
	log_file = log_folder + "\\" + name + '_opt' + results_file_id_exp + ".log"

	# Clean file (otherwise it appends)
	fid = open(log_file, 'w+')
	fid.close()

	# Start a logger
	logger = Diary(log_file)

	muscles = osim_model_ref.getMuscles()
	muscles_scaled = osim_model_targ.getMuscles()

	# Initialise with recognisable values
	LmOptLts_opt = np.ones((muscles.getSize(), 2)) * -1000

	for n_mus in range(muscles.getSize()):
		# Current muscle name (here so it is possible to choose a single muscle when developing)
		curr_muscle_name = muscles.get(n_mus).getName()
		print('Processing mus %d: %s' % (n_mus, curr_muscle_name))

		# Import muscles
		curr_mus = muscles.get(n_mus)
		curr_mus_scaled = muscles_scaled.get(curr_muscle_name)

		# Extracting the muscle parameters from reference model
		LmOptLts = [curr_mus.getOptimalFiberLength(), curr_mus.getTendonSlackLength()]
		pen_ang_opt = curr_mus.getPennationAngleAtOptimalFiberLength()
		mus_ref = sample_muscle_quantities(osim_model_ref, curr_mus, 'all', N_eval)
		
		# Calculating minimum fiber length before having pennation 90 degrees
		# acos(0.1) = 1.47 rad = 84 deg, chosen as in OpenSim
		limit_pen_angle = np.arccos(0.1)

		# This is the minimum length the fiber can be for geometrical reasons
		Lfib_norm_min = np.sin(pen_ang_opt) / np.sin(limit_pen_angle)

		# Lfib_norm as calculated about can be shorter than the minimum length at which
		# the fiber can generate force (taken to be 0.5 Zajac 1989)
		if (Lfib_norm_min < 0.5):
			Lfib_norm_min = 0.5
		
		# Checking the muscle configuration that do not respect the condition
		Lfib_norm_ref = np.array(mus_ref[1])
		ok_list = (Lfib_norm_ref > Lfib_norm_min)

		# Keeping only acceptable values
		Lfib_norm_ref = Lfib_norm_ref[ok_list]
		Lten_norm_ref = np.array(mus_ref[2])[ok_list] / LmOptLts[1]
		MTL_ref = np.array(mus_ref[0])[ok_list]
		pen_angle_ref = np.array(mus_ref[4])[ok_list]
		Lfib_norm_on_ten_ref = Lfib_norm_ref*np.cos(pen_angle_ref)

		# In the target only MTL is needed for all muscles
		MTL_targ = sample_muscle_quantities(osim_model_targ, curr_mus_scaled, 'MTL', N_eval)
		eval_total_points = len(MTL_targ)
		MTL_targ = np.array(MTL_targ)[ok_list]
		eval_ok_points = len(MTL_targ)

		# The problem to be solved is:
		# [Lm_norm*cos(pen_angle) Lt_norm]*[Lm_opt Lts]' = MTL
		# Written as Ax = b
		A = np.concatenate((Lfib_norm_on_ten_ref[:,np.newaxis], Lten_norm_ref[:, np.newaxis]), 1)
		b = MTL_targ[:,np.newaxis]

		n = np.shape(A)[1]
		Q, R = np.linalg.qr(A, mode="complete")
		x = scipy.linalg.solve_triangular(R[:n], Q.T[:n].dot(b), lower=False)

		LmOptLts_opt[n_mus,:] = x.T

		# Checking the results
		if min(x) < 0:
			# Inform the user
			logger.log('Negative value estimated for muscle parameter of muscle ' + curr_muscle_name)
			logger.log('                         Lm Opt        Lts')
			logger.log('Template model       : ' + ' '.join(map(str,LmOptLts)))
			logger.log('Optimized param      : ' + ' '.join(map(str,LmOptLts_opt[n_mus,:])))

			''' Implementing corrections if estimations are not correct '''

			# First try lsqnonlin
			b = MTL_targ
			x = scipy.optimize.nnls(A,b)[0]
			LmOptLts_opt[n_mus,:] = x

			logger.log('Opt params (nnls): ' + str(LmOptLts_opt[n_mus,:]))

			# In our tests, if something goes wrong, it is generally the tendon slack length becoming
			# negative or zero because the tendon length doesn't change throughout the range of motion, so 
			# lowering rank of A.

			if min(x) <= 0:
				if max(np.array(mus_ref[2])[ok_list]) - min(np.array(mus_ref[2])[ok_list]) < 0.0001:
					logger.log('Tendon length not changing throughout range of motion')
				
				# Calculating proportion of tendon and fiber
				Lten_fraction = np.array(mus_ref[2])[ok_list] / MTL_ref
				Lten_targ = (Lten_fraction * MTL_targ)

				x = []
				# First round: optimising Lopt maintaining the proportion of tendon as in the reference
				# model
				A_1 = Lfib_norm_on_ten_ref
				b_1 = (MTL_targ - Lten_targ)

				n = np.shape(A_1[:,np.newaxis])[1]
				Q, R = np.linalg.qr(A_1[:,np.newaxis], mode="complete")
				x.append((scipy.linalg.solve_triangular(R[:n], Q.T[:n].dot(b_1), lower=False))[0])

				# Second round: using the optimised Lopt to recalculate Lts
				b_2 = (MTL_targ - A_1*x[0]).T
				A_2 = Lten_norm_ref

				n = np.shape(A_2[:,np.newaxis])[1]
				Q, R = np.linalg.qr(A_2[:,np.newaxis], mode="complete")
				x.append((scipy.linalg.solve_triangular(R[:n], Q.T[:n].dot(b_2), lower=False))[0])

				LmOptLts_opt[n_mus,:] = x

		# Here, tests about '\' against optimisers were implemented

		# Calculating the error (sum of squared errors)
		fval = np.linalg.norm(np.matmul(A,x) - b) ** 2

		# Update muscles from scaled model
		curr_mus_scaled.setOptimalFiberLength(LmOptLts_opt[n_mus,0])
		curr_mus_scaled.setTendonSlackLength(LmOptLts_opt[n_mus,1])
				




		
		





		a = 1



optimal_muscle_parameters('C:\\Users\\alexw\\Desktop\\MuscleParamOptimizer_ManuscriptPackage_04Jan2016\\Example1\\MSK_Models\\Reference_Hamner_L.osim',
 'C:\\Users\\alexw\\Desktop\\MuscleParamOptimizer_ManuscriptPackage_04Jan2016\\Example1\\MSK_Models\\Target_Hamner_scaled_L.osim',
  5,
   'C:\\Users\\alexw\\Desktop\\MuscleParamOptimizer_ManuscriptPackage_04Jan2016\\Example1\\OptimModels')