import opensim as osim
import os
import pickle

def assess_muscle_param_var(template_osim_model: osim.Model, optimised_osim_model: osim.Model, N_eval: int, results_folder: str):
	'''
	Script to evaluate the results of the optimized muscle scaling.
	The script calculates metrics on the variation of the muscle parameters
	with respect to the reference model.

	'''

	# Results file identifier
	results_file_id_exp = '_N' + str(N_eval)

	# Results file name
	res_file_name = 'Results_MusVarMetrics' + results_file_id_exp

	# Extract muscle sets from the two models
	muscles_ref = template_osim_model.getMuscles()
	muscles_opt = optimised_osim_model.getMuscles()

	if os.path.isfile(results_folder + "\\" + res_file_name + '.pckl'):
		pref = input(res_file_name + ' exists. Do you want to re-evaluate muscle param % variations? [y/n]. ')

		if pref == 'n':
			print('Loading existing file')
			with open(results_folder + "\\" + res_file_name + '.pckl', 'rb') as f:
				Results_MusVarMetrics = pickle.load(f)
			return
		
		elif pref == 'y':
			print('Re-evaluating muscle percentage variations.')

	else:
		print('Evaluating muscle percentage variations.')
	
	# Set up lists to be saved
	colheaders = []
	Lopt_templ = []
	Lopt_opt = []
	Lts_templ = []
	Lts_opt = []
	Lopt_var_list = []
	Lts_var_list = []

	i = 0
	for n_mus in range(muscles_ref.getSize()):
		# Current muscle name
		i += 1 # Increase counter by 1
		curr_mus_name = muscles_ref.get(n_mus).getName()
		print('Processing muscle ' + str(i) + '/' + str(muscles_ref.getSize()) + ': ' + curr_mus_name)

		# Extracting the current muscle from the two models
		current_muscle_templ = muscles_ref.get(curr_mus_name)
		current_muscle_optim = muscles_opt.get(curr_mus_name)

		# Normalised fiber lengths for the template
		Lopt_var = 100 * (current_muscle_optim.getOptimalFiberLength() - current_muscle_templ.getOptimalFiberLength()) / current_muscle_templ.getOptimalFiberLength()
		Lts_var = 100 * (current_muscle_optim.getTendonSlackLength() - current_muscle_templ.getTendonSlackLength()) / current_muscle_templ.getTendonSlackLength()

		# Structure of results
		colheaders.append(current_muscle_templ.getName())
		Lopt_templ.append(current_muscle_templ.getOptimalFiberLength())
		Lopt_opt.append(current_muscle_optim.getOptimalFiberLength())
		Lts_templ.append(current_muscle_templ.getTendonSlackLength())
		Lts_opt.append(current_muscle_optim.getTendonSlackLength())
		Lopt_var_list.append(Lopt_var)
		Lts_var_list.append(Lts_var)

	print('\n') # New line for ease of reading
	
	Results_MusVarMetrics = {}

	Results_MusVarMetrics['colheaders'] = colheaders
	Results_MusVarMetrics['Lopt_templ'] = Lopt_templ
	Results_MusVarMetrics['Lopt_opt'] = Lopt_opt
	Results_MusVarMetrics['Lts_templ'] = Lts_templ
	Results_MusVarMetrics['Lts_opt'] = Lts_opt
	Results_MusVarMetrics['Lopt_var'] = Lopt_var_list
	Results_MusVarMetrics['Lts_var'] = Lts_var_list

	# Extracting max and min variations for Lopt
	Lopt_var_max = max(Lopt_var_list)
	ind_max = Lopt_var_list.index(max(Lopt_var_list))
	Lopt_var_min = min(Lopt_var_list)
	ind_min = Lopt_var_list.index(min(Lopt_var_list))

	Results_MusVarMetrics['Lopt_var_range'] = [Lopt_var_min, Lopt_var_max]
	Results_MusVarMetrics['Lopt_var_range_mus'] = [colheaders[ind_min], colheaders[ind_max]]

	# Extracting max and min variations for Lts
	Lts_var_max = max(Lts_var_list)
	ind_max = Lts_var_list.index(max(Lts_var_list))
	Lts_var_min = min(Lts_var_list)
	ind_min = Lts_var_list.index(min(Lts_var_list))

	Results_MusVarMetrics['Lts_var_range'] = [Lts_var_min, Lts_var_max]
	Results_MusVarMetrics['Lts_var_range_mus'] = [colheaders[ind_min], colheaders[ind_max]]

	# Save dictionary with results
	with open(results_folder + "\\" + res_file_name + '.pckl', 'wb') as f:
		pickle.dump([Results_MusVarMetrics], f)

	return Results_MusVarMetrics