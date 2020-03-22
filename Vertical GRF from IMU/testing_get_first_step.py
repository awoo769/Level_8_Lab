import csv
import numpy as np
import glob
import sys
import os

from prepare_data_copy import read_csv, rezero_filter


''' Read in file '''
path = 'C:\\Users\\alexw\\Desktop\\RunningData\\'
ext = 'csv'
os.chdir(path)
files = glob.glob('*.{}'.format(ext))

for f in files:
	print('Running file: ' + str(f))
	data = read_csv(f)

	a = 1

