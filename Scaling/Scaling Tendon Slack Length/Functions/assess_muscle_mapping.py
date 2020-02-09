import opensim as osim
import os
import pickle
import numpy as np
from scipy.stats import pearsonr

from sample_muscle_quantities import sample_muscle_quantities

def assess_muscle_mapping(template_osim_model: osim.Model, optimised_osim_model: osim.Model, N_eval: int, N_error: int, results_folder: str):
	'''
	Script to evaluate the results of the muscle optimization. The script calculated the normalized fiber 
	lengths for the model template and for the scaled model with optimized muscle parameters at N_error 
	points and computes an RMSE (assuming that the optimization is aiming to "track" the normalized FL 
	curve of the muscles) in the reference model and the mean and maximum error in the tracking (ranges 
	across muscles). N_eval identifies the optimized model, obtained by sampling each degree of freedom's 
	range in N_Eval points.

	'''

	# Extract muscle sets from the two models
	muscles_ref = template_osim_model.getMuscles()
	muscles_opt = optimised_osim_model.getMuscles()

	''' File where to store the metrics '''
	# Results file identifier
	results_file_id_exp = '_N' + str(N_eval)

	# Results file name
	res_file_name = 'Results_MusVarMetrics' + results_file_id_exp + '_NError' + str(N_error)

	if os.path.isfile(results_folder + "\\" + res_file_name + '.pckl'):
		pref = input(res_file_name + ' exists. Do you want to re-evaluate muscle param % variations? [y/n]. ')

		if pref == 'n':
			print('Loading existing file')
			with open(results_folder + "\\" + res_file_name + '.pckl', 'rb') as f:
				Results_MusMapMetrics = pickle.load(f)
			return
		
		elif pref == 'y':
			print('Re-evaluating mapping results.')

	else:
		print('Evaluating mapping results.')
	
	# Set up lists to be saved
	colheaders = []
	RMSE = []
	MaxPercError = []
	MinPercError = []
	MeanPercError = []
	StandDevPercError = []
	corrCoeff = []

	i = 0
	for n_mus in range(muscles_ref.getSize()):
		i += 1 # increase counter by 1
		curr_mus_name = 'ext_hal_l'
		# Current muscle name
		curr_mus_name = muscles_ref.get(n_mus).getName()
		print('Processing muscle ' + str(i) + '/' + str(muscles_ref.getSize()) + ': ' + curr_mus_name)

		# Extracting the current muscle from the two models
		current_muscle_templ = muscles_ref.get(curr_mus_name)
		current_muscle_optim = muscles_opt.get(curr_mus_name)

		''' Normalised fiber lengths calculated at N_Error points '''
		# Normalised fiber lengths from the reference model at N_Error points per dof
		Lm_Norm_ref = sample_muscle_quantities(template_osim_model, current_muscle_templ, 'LfibNorm', N_error)
		# Normalised fiber length from the optimised model at N_Error points per dof
		Lm_Norm_opt = sample_muscle_quantities(optimised_osim_model, current_muscle_optim, 'LfibNorm', N_error)

		''' Check for NaN '''
		# Checks on NaN on the results
		if np.isnan(np.array(Lm_Norm_ref)).any():
			print('NaN detected for ' + curr_mus_name + ' in the template model.')

		if np.isnan(np.array(Lm_Norm_opt)).any():
			print('NaN detected for ' + curr_mus_name + ' in the optimised model.')

		''' Check for unrealistic fiber lengths '''
		# Check on the results of the sampling: if the reference model sampling gave some unrealistic
		# fiber lengths, this is where this should be corrected
		# Boundaries for normalised fiber lengths

		''' Fiber lengths that make pennation angle >= 90 degrees '''
		# Calculating a minimum fiber length before having pennation 90 deg
		# acos(0.1) = 1.47 rad = 84 deg, chosen as in OpenSim
		limit_pen_angle = np.arccos(0.1)

		# This is the minimum length the fiber can be for geometrical reasons
		pen_angle_opt = current_muscle_templ.getPennationAngleAtOptimalFiberLength()
		Lfib_norm_min_templ = np.sin(pen_angle_opt) / np.sin(limit_pen_angle)

		''' Normalised fiber lengths < 0.5 '''
		# Lfib_norm as calculated above can be shorter than the minimum length at which the fiber can generate
		# force (taken to be 0.5 Zajac 1989)
		if (Lfib_norm_min_templ < 0.5):
			Lfib_norm_min_templ = 0.5

		# Indices of points that are okay according to previous criteria
		ok_point_ind = np.array(Lm_Norm_ref) > Lfib_norm_min_templ
		# Checking the muscle configuration that do not respect the condition
		Lm_Norm_ref = np.array(Lm_Norm_ref)[ok_point_ind]
		# Checking the muscle configuration that do not respect the condition
		Lm_Norm_opt = np.array(Lm_Norm_opt)[ok_point_ind]

		''' Null normalised fiber lengths '''
		# Muscle normalised length cannot be zero either
		if min(Lm_Norm_ref) == 0:
			print('Zero Lm_Norm for muscle ' + curr_mus_name + ' in template model. Removing points with zero length')

			ok_points = Lm_Norm_ref != 0
			Lm_Norm_ref = Lm_Norm_ref[ok_points]
			Lm_Norm_opt = Lm_Norm_opt[ok_points]

		''' Calculating the metrics '''
		# Difference between the two normalised fiber length vectors evaluated at N_error
		Diff_Lfnorm = Lm_Norm_ref - Lm_Norm_opt

		# Structure of results
		colheaders.append(current_muscle_templ.getName())
		RMSE.append(np.sqrt(sum(Diff_Lfnorm * Diff_Lfnorm) / len(Lm_Norm_ref)))
		MaxPercError.append(max(abs(Diff_Lfnorm) / Lm_Norm_ref) * 100)
		MinPercError.append(min(abs(Diff_Lfnorm) / Lm_Norm_ref) * 100)
		MeanPercError.append(np.mean(abs(Diff_Lfnorm) / Lm_Norm_ref) * 100)
		StandDevPercError.append(np.std(abs(Diff_Lfnorm) / Lm_Norm_ref) * 100)

		rho, Pval = pearsonr(Lm_Norm_ref.T, Lm_Norm_opt.T)
		corrCoeff.append([rho, Pval])

		# Clear variables to avoid issues for different sampling
		del Lm_Norm_ref
		del Lm_Norm_opt

	print('\n') # New line for ease of reading

	Results_MusMapMetrics = {}
	Results_MusMapMetrics['colheaders'] = colheaders
	Results_MusMapMetrics['RMSE'] = RMSE
	Results_MusMapMetrics['MaxPercError'] = MaxPercError
	Results_MusMapMetrics['MinPercError'] = MinPercError
	Results_MusMapMetrics['MeanPercError'] = MeanPercError
	Results_MusMapMetrics['StandDevPercError'] = StandDevPercError
	Results_MusMapMetrics['corrCoeff'] = corrCoeff

	# Computing min and max RMSE
	RMSE_max = max(RMSE)
	ind_max = RMSE.index(max(RMSE))
	RMSE_min = min(RMSE)
	ind_min = RMSE.index(min(RMSE))

	Results_MusMapMetrics['RMSE_range'] = [RMSE_min, RMSE_max]
	Results_MusMapMetrics['RMSE_range_mus'] = [colheaders[ind_min], colheaders[ind_max]]

	# Computing min and max MeanPercError
	MeanPercError_max = max(MeanPercError)
	ind_max = MeanPercError.index(max(MeanPercError))
	MeanPercError_min = min(MeanPercError)
	ind_min = MeanPercError.index(min(MeanPercError))

	Results_MusMapMetrics['MeanPercError_range'] = [MeanPercError_min, MeanPercError_max]
	Results_MusMapMetrics['MeanPercError_range_mus'] = [colheaders[ind_min], colheaders[ind_max]]

	# Computing min and max variations for MaxPercError
	MaxPercError_max = max(MeanPercError)
	ind_max = MeanPercError.index(max(MeanPercError))
	MaxPercError_min = min(MaxPercError)
	ind_min = MeanPercError.index(min(MeanPercError))

	Results_MusMapMetrics['MeanPercError_range'] = [MaxPercError_min, MaxPercError_max]
	Results_MusMapMetrics['MeanPercError_range_mus'] = [colheaders[ind_min], colheaders[ind_max]]

	# Extracting max and min corr coeff and p values
	Results_MusMapMetrics['rho_pval_range'] = [min(corrCoeff), max(corrCoeff)]

	# Save dictionary with results
	with open(results_folder + "\\" + res_file_name + '.pckl', 'wb') as f:
		pickle.dump([Results_MusMapMetrics], f)

	return Results_MusMapMetrics
