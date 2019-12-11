def fix_grf_headers(grf_headers: list, steps: list, plates: list):
	'''
	Takes Vicon output headers and converts them to logical ones (which also fit the existing External 
	Load File template)

	Inputs:	grf_headers: header names from the trc file
			steps: list containing the step order ['l', 'r']
			plates: list containing the plates which correspond to each step [1, 2]

	Output:	new_headers: the corrected headers

	'''

	left_plate = str(plates[(steps.index('l'))])
	right_plate = str(plates[(steps.index('r'))])

	new_headers = []

	for i in range(len(grf_headers)):
		if left_plate in grf_headers[i]:
			num_ind = grf_headers[i].find(left_plate)
			# Remove the number from the header and replace with nothing
			clean_header = (grf_headers[i])[:num_ind] + (grf_headers[i])[num_ind+1:]
			#new_headers.append('L_' + clean_header) # old style
			new_headers.append(clean_header) # OpenSim 4.0

		elif right_plate in grf_headers[i]:
			num_ind = grf_headers[i].find(right_plate)
			# Remove the number from the header and replace with 1
			clean_header = (grf_headers[i])[:num_ind] + (grf_headers[i])[num_ind+1:]
			#new_headers.append('R_' + clean_header) # old style
			new_headers.append('1_' + clean_header) # OpenSim 4.0
		
		else:
			new_headers.append(grf_headers[i])

	return new_headers