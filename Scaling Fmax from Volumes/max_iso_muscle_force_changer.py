import opensim as osim
import numpy as np
import re

from muscle_volume_calculator import muscle_volume_calculator

# Subject data/model info
main_path = 'C:\\Users\\alexw\\Desktop\\Hip_OA_models\\'

# Format for subject_data
# 1 subject per row
# column 1 = height (m)
# column 2 = mass (kg)
# column 3 = path to OpenSim model, will be concatenated with main_path

subject_data = np.array([[1.706,	76.2,	'H20s1_LLM_Hip_OA_simbody_updated_opt_N10.osim'],
						 [1.652,	64.4,	'H21s1_LLM_Hip_OA_simbody_updated_opt_N10.osim'],
						 [1.588,	70.5,	'H22s1_LLM_Hip_OA_simbody_updated_opt_N10.osim'],
						 [1.725,	76.5,	'H23s1_LLM_Hip_OA_simbody_updated_opt_N10.osim'],
						 [1.762,	84.3,	'H24s1_LLM_Hip_OA_simbody_updated_opt_N10.osim'],
						 [1.817,	108.4,	'H25s1_LLM_Hip_OA_simbody_updated_opt_N10.osim'],
						 [1.657,	101.6,	'H26s1_LLM_Hip_OA_simbody_updated_opt_N10.osim'],
						 [1.67,		83.7,	'H27s1_LLM_Hip_OA_simbody_updated_opt_N10.osim'],
						 [1.50,		80.15,	'H28s1_LLM_Hip_OA_simbody_updated_opt_N10.osim'],
						 [1.67,		57.9,	'H30s1_LLM_Hip_OA_simbody_updated_opt_N10.osim']])

# s = rows in subject_data = number of subjects
for s in range(np.shape(subject_data)[0]):
	subject_height = float(subject_data[s, 0]) # input('Enter subject height (m): ')
	subject_mass = float(subject_data[s, 1])   # input('Enter subject mass (kg): ')
	subject_path = main_path + subject_data[s, 2]

	# Calculate muscle volumes using Handsfield (2004)
	osim_abbr, muscle_volume = muscle_volume_calculator(subject_height, subject_mass)

	# Load OpenSim model and its muscle set
	osim_model = osim.Model(subject_path)
	all_muscles = osim_model.getMuscles()

	all_muscles_names = [None] * all_muscles.getSize()
	old_value = np.zeros((all_muscles.getSize(), 1))
	optimal_fibre_length = np.zeros((all_muscles.getSize(), 1))
	pen_ang_at_opt_fib_length = np.zeros((all_muscles.getSize(), 1))

	for i in range(all_muscles.getSize()):
		all_muscles_names[i] = all_muscles.get(i).getName()
		old_value[i, 0] = all_muscles.get(i).getMaxIsometricForce()
		optimal_fibre_length[i, 0] = all_muscles.get(i).getOptimalFiberLength()
		pen_ang_at_opt_fib_length[i, 0] = all_muscles.get(i).getPennationAngleAtOptimalFiberLength()
	
	# Convert optimal fiber length from m to cm to match volume units (cm^3)
	optimal_fibre_length = optimal_fibre_length * 100

	all_muscles_names_cut = [None] * all_muscles.getSize()

	for i in range(all_muscles.getSize()):
		# Delete trailing _r or _l
		curr_mus_name = all_muscles_names[i][:-2]

		# Split the name from any digit in its name and only keep the first string
		all_muscles_names_cut[i] = re.split(r'\d', curr_mus_name)[0]

	# Calculate ratio of old max isometric forces for multiple-lines-of-action muscles
	new_abs_volume = np.zeros((all_muscles.getSize(), 1))
	frac_of_group = np.zeros((all_muscles.getSize(), 1))

	for i in range(all_muscles.getSize()):
		curr_mus_name = all_muscles_names_cut[i]
		
		try: 
			curr_index = osim_abbr.index(curr_mus_name)
			curr_value = muscle_volume[curr_index]
			new_abs_volume[i,0] = curr_value

			curr_muscle_name_index = []
			tmp_index = [j for j in range(len(all_muscles_names_cut)) if all_muscles_names_cut[j] == curr_mus_name]
			curr_muscle_name_index.append(tmp_index[:int(len(tmp_index)/2)])

		except ValueError: # muscle name not found
			# The peroneus longus/brevis and the extensors (EDL, EHL) have to be treated separately as they are
			# represented as a combined muscle group in Handsfield (2014).
			if ('per_brev' in curr_mus_name) or ('per_long' in curr_mus_name):
				curr_muscle_name_index = []

				tmp_index = all_muscles_names_cut.index('per_brev') # .index() finds the first occuring element
				curr_muscle_name_index.append(tmp_index)

				tmp_index = all_muscles_names_cut.index('per_long') # .index() finds the first occuring element
				curr_muscle_name_index.append(tmp_index)

				curr_index = osim_abbr.index('per_')
				curr_value = muscle_volume[curr_index]
				new_abs_volume[i] = curr_value

			elif ('ext_dig' in curr_mus_name) or ('ext_hal' in curr_mus_name):
				curr_muscle_name_index = []

				tmp_index = all_muscles_names_cut.index('ext_dig') # .index() finds the first occuring element
				curr_muscle_name_index.append(tmp_index)

				tmp_index = all_muscles_names_cut.index('ext_hal') # .index() finds the first occuring element
				curr_muscle_name_index.append(tmp_index)

				curr_index = osim_abbr.index('ext_')
				curr_value = muscle_volume[curr_index]
				new_abs_volume[i] = curr_value
			
			else:
				curr_muscle_name_index = []
				tmp_index = [j for j in range(len(all_muscles_names_cut)) if all_muscles_names_cut[j] == curr_mus_name]
				curr_muscle_name_index.append(tmp_index[:int(len(tmp_index)/2)])
				
		frac_of_group[i,0] = old_value[i]/sum(old_value[curr_muscle_name_index])

	''' Calculate the new maximal isometric muscle forces '''
	specific_tension = 61 # N/cm^2 from Zajac 1989
	new_volume = frac_of_group*new_abs_volume

	# Calculates muscle force, OpenSim likely wants the fibre force
	max_iso_muscle_force = specific_tension * (new_volume/optimal_fibre_length) * np.cos(pen_ang_at_opt_fib_length)

	''' Update muscles of loaded model (in workspace only), change model name and print new osim file '''
	for i in range(all_muscles.getSize()):
		# Only update if new value is not 0. Else do not override the original value
		if max_iso_muscle_force[i][0] != 0:
			all_muscles.get(i).setMaxIsometricForce(max_iso_muscle_force[i][0])
	
	# Create and set new model name by adding '_newFmax' at the end
	osim_name_old = osim_model.getName().split('.')[0]
	osim_name_new = osim_name_old + '_newFmax.osim'
	osim_model.setName(osim_name_new)

	print_new_model = 'yes'

	# Print new model in specified path
	if print_new_model == 'yes':
		print_path = main_path + osim_name_new

		osim_model.printToXML(print_path)
	
	else:
		print('----------------------------------------')
		print('WARNING: no new model file was printed!')
		print('Change in settings and re-run if wanted.')
		print('----------------------------------------')
