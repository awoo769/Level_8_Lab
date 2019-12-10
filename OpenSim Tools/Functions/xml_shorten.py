import re

def xml_shorten(file_path):
	'''
	Removes extra lines created within the xml document

	Inputs: file_path: the full file path to be opened
	
	'''

	# read the file
	fid = open(file_path, 'r')
	file_contents = fid.readlines()
	fid.close()

	# write to a new file
	fid = open(file_path,'w')

	# Remove extra lines
	for line in file_contents:
		line_replace = re.sub(r'\s*\n', '\n', line)
		if line == line_replace and not line == '\n': # Nothing has been changed, otherwise we don't want the line
			fid.write(line)
	fid.close()