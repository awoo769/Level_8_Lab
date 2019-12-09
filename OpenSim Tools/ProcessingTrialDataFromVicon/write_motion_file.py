
import numpy as np

def write_motion_file(grf_complete: np.ndarray, file_path: str, headers: list):
	'''
	This file will export the complete ground reaction force data into a _grf.mot file
	'''

	# Get dimensions of the pre-processed grf data
	m,n = np.shape(grf_complete)

	# Get new file name
	new_file = file_path.replace('.csv', '.mot')

	# Split path to get file_name
	file_name = file_path.rsplit('/',1)[-1]

	# Create new file to write to
	fid = open(new_file, 'w+')

	# Print header
	fid.write("%s\n" % (file_name))
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




