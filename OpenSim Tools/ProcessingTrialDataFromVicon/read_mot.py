from tkinter import filedialog
import numpy as np

def read_mot(data_start: int, *file_path: str):

	# If optional argument is given, don't find filepath, if not, then do
	if len(file_path) == 0:
		file_path = filedialog.askopenfilename(initialdir = "r",title = "Select file",filetypes = (("mot files","*.mot"),("all files","*.*")))

	else:
		# Convert file path tuple to string
		file_path = str(file_path[0])

	# Open mot file with read only access
	mot_fid = open(file_path,'r')

	mot_lines = mot_fid.readlines()
	mot_fid.close()

	# First line of data
	mot_data = mot_lines[data_start-1:]

	# Headers will be on the line before the data starts
	mot_headers = mot_lines[data_start-2].split('\t')

	data = []

	# Split into an array for easier manipulation
	for i in range(len(mot_data)):
		tmp = [float(i) for i in mot_data[i].split()]
		data.append(tmp)

	# Convert to array of floats
	data = np.array(data)



	return mot_headers, data