import csv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def read_csv(filename: str, first_data: int):
	'''
	This function opens and reads a csv file, returning a numpy array (data) of the contents.

	'''

	with open(filename, 'r') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')

		data = []
		i = 0
		
		for row in reader:
			i += 1

			# First data row on line 8
			if i >= first_data:
				if len(row) != 0:
					data.append(row)

	return np.array(data)

direct = "C:/Users/alexw/Desktop/Un-synced data for Alex Woodall/0024/0024 L.csv"
first_data = 0

columns = ['time', 'ax', 'ay', 'az']
df = pd.DataFrame(read_csv(direct, first_data), columns=columns)

plt.plot(df['time'].head(1000), df['ax'].head(1000))
plt.show()

	