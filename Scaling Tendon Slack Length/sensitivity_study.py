import numpy as np
import os
import opensim as osim
import pickle

from optimal_muscle_parameters import optimal_muscle_parameters
from assess_muscle_param_var import assess_muscle_param_var
from assess_muscle_mapping import assess_muscle_mapping

'''
This script:
1) Optimises muscle parameters varying the number of points used in the optimisation from 5 to 15 per degree
of freedom. Optimised models and optimised log are saved in the folder "..."
2) Evaluates the results of the optimisation in terms of muscle parameters variation and muscle mapping metrics
(and saves structures summarising the results in the folder "...")

'''

''' Initialising folders and setup '''

# Getting example details
case_ID = "Example1"
osim_model_ref_file = "Reference_Hamner_L.osim"
osim_model_targ_file = "Target_Hamner_scaled_L.osim"

# Folders used
ref_model_folder = "C:\\Users\\alexw\\Desktop\\MuscleParamOptimizer_ManuscriptPackage_04Jan2016\\Example1\\MSK_Models"
targ_model_folder = ref_model_folder
optimsed_model_folder =  'C:\\Users\\alexw\\Desktop\\MuscleParamOptimizer_ManuscriptPackage_04Jan2016\\Example1\\OptimModels'
results_folder =  'C:\\Users\\alexw\\Desktop\\MuscleParamOptimizer_ManuscriptPackage_04Jan2016\\Example1\\Results'
log_folder = optimsed_model_folder

# Check if results and optimised model folder exists
if not os.path.exists(optimsed_model_folder):
	os.makedirs(optimsed_model_folder)

if not os.path.exists(results_folder):
	os.makedirs(results_folder)

# Model files with paths
osim_model_ref_filepath = os.path.join(ref_model_folder, osim_model_ref_file)
osim_model_targ_filepath = os.path.join(ref_model_folder, osim_model_targ_file)

# Reference model for calculating results metrics
osim_model_ref = osim.Model(osim_model_ref_filepath)

for N_eval in range(5, 15+1):

	sims_info = {}
	''' Muscle optimiser '''
	# Optimising target model based on reference model for N_eval points per degree of freedom
	osim_model_opt, sim_info = optimal_muscle_parameters(osim_model_ref_filepath, osim_model_targ_filepath, N_eval, log_folder)

	# Add simulation information to overall dictionary
	sims_info[N_eval] = sim_info

	''' Printing optimised model '''
	# Setting the output folder
	if optimsed_model_folder == '' or len(optimsed_model_folder) == 0:
		optimsed_model_folder = targ_model_folder

	# Printing the optimised model
	osim_model_opt.printToXML(os.path.join(optimsed_model_folder, osim_model_opt.getName()))

	''' Saving Results '''
	# Variation in muscle parameters
	results_musvarmetrics = assess_muscle_param_var(osim_model_ref, osim_model_opt, N_eval, results_folder)
	# Assess muscle mapping (RMSE, max error, etc) at n_Metrics points between reference and optimised model
	n_Metrics = 10
	results_musmapmetrics = assess_muscle_mapping(osim_model_ref, osim_model_opt, N_eval, n_Metrics, results_folder)
	
# Save simulation infos
with open(results_folder + "\\SimsInfo" + '.pckl', 'wb') as f:
	pickle.dump([sims_info], f)
