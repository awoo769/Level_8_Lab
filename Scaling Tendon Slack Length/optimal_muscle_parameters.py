import opensim as osim
import os
from diary import Diary # Need to install diary - pip install diary
import numpy as np
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
		

		# Extracting 
	a = 1



optimal_muscle_parameters('C:\\Users\\alexw\\Desktop\\MuscleParamOptimizer_ManuscriptPackage_04Jan2016\\Example1\\MSK_Models\\Reference_Hamner_L.osim',
 'C:\\Users\\alexw\\Desktop\\MuscleParamOptimizer_ManuscriptPackage_04Jan2016\\Example1\\MSK_Models\\Target_Hamner_scaled_L.osim',
  5,
   'C:\\Users\\alexw\\Desktop\\MuscleParamOptimizer_ManuscriptPackage_04Jan2016\\Example1\\OptimModels')