"""
Plot average and standard deviation

"""

__author__ = "Thor Besier, Alex Woodall"
__version__ = "2.0"
__license__ = "ABI"

import numpy as np
import matplotlib.pyplot as plt

def plot_avsd(x: np.ndarray, y: np.ndarray, sd: np.ndarray):
	tmp_x = np.concatenate((x, np.flipud(x), x[0]),axis=None)
	tmp_y = np.concatenate((y+sd, np.flipud(y - sd), y[0] + sd[0]),axis=None)

	fig, ax = plt.subplots()
	ax.plot(x,y,'-',color='blue')
	ax.plot(tmp_x,tmp_y,color='orange')
	ax.fill(tmp_x,tmp_y,color='gray')

	plt.xlabel('Normalised time')
	plt.ylabel('x')
	plt.show()

