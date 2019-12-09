def fix_grf_headers(grf_headers: list, steps: list, plates: list):
	'''
	Takes Vicon output headers and converts them to logical ones (which also fit the existing External 
	Load File template)

	'''

	left_plate = str(plates[(steps.index('l'))])
	right_plate = str(plates[(steps.index('r'))])

	new_headers = []

	for i in range(len(grf_headers)):
		if left_plate in grf_headers[i]:
			num_ind = grf_headers[i].find(left_plate)
			# Remove the number from the header and replace with L
			clean_header = (grf_headers[i])[:num_ind] + (grf_headers[i])[num_ind+1:]
			new_headers.append('L_' + clean_header)

		elif right_plate in grf_headers[i]:
			num_ind = grf_headers[i].find(right_plate)
			# Remove the number from the header and replace with R
			clean_header = (grf_headers[i])[:num_ind] + (grf_headers[i])[num_ind+1:]
			new_headers.append('R_' + clean_header)
		
		else:
			new_headers.append(grf_headers[i])

	return new_headers