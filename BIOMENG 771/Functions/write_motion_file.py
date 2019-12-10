
import numpy as np

def write_motion_file(grf_complete: np.ndarray, file_path: str, *headers: list):
	'''
	This file will export the complete ground reaction force data into a _grf.mot file

	Inputs:	grf_complete: an array of the complete ground reaction forces
			file_path: a full file path to write a motion file to
			headers: a list of all the headers (optional), will default to a set list of headers

	'''

	if len(headers) == 0:
		headers = ['time', 'ground_force_vx', 'ground_force_vy', 'ground_force_vz', 'ground_force_px',
		'ground_force_py', 'ground_force_pz', 'ground_torque_x', 'ground_torque_y', 'ground_torque_z']

	# Get dimensions of the pre-processed grf data
	m,n = np.shape(grf_complete)

	# Get new file name
	new_file = file_path.replace('.csv', '.mot')

	# Split path to get file_name
	file_name = file_path.rsplit('/',1)[-1]

	# Create new file to write to
	fid = open(new_file, 'w+')

	# Print header
	fid.write("%s\n" % (file_name.split("\\")[-1]))
	fid.write("version=1\n")
	fid.write("nRows=%d\n" % (m))
	fid.write("nColumns=%d\n" % (n))
	fid.write("inDegrees=yes\n")
	fid.write("endheader\n\n")

	# [time vector1 point1 vector2 point2 torque1 torque2]
	for header in headers:
		fid.write("%s\t" % (header))

	fid.close()

	# Print data
	# Open the file to append to
	fid = open(new_file, "a+")

	# Save the force array to the file
	np.savetxt(fid, grf_complete, fmt='%0.6f', delimiter='\t')

	# Close the file
	fid.close()

	# Print new file name for the user
	print("New MOT location: %s" % (new_file))




