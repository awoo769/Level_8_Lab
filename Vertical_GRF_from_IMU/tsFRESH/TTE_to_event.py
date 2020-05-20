import pandas as pd
from sklearn.metrics import r2_score

def TTE_to_event(y_estimate: pd.Series):
	'''
	This function will take in an estimated TTE and will convert to either a time of an event or no event

	Alex Woodall
	19/05/2020

	'''

	import numpy as np
	import math

	# Prepare a new series for the converted output
	y_events = y_estimate.copy()

	next_event = []
	for i in range(y_estimate.size):
		time = (y_estimate.iloc[i])[0]
		round_whole = math.floor(time / 100) * 100

		event_estimate = time - round_whole

		next_event.append(event_estimate)

		if round_whole == 0:
			# Then we think the event occurs in this sample
			estimate = int(np.mean(next_event))

			y_events.iloc[i] = estimate

			# Reset next_event list
			next_event = []
		
		else:
			y_events.iloc[i] = -1
		
	return y_events


if __name__ == "__main__":
	filepath_est = 'C:\\Users\\alexw\\Desktop\\Harvard_data\\0_1_2_3_4_5_6_7\\FS_time\\y_estimate_test_NDR0208ITLaMaRl01.csv'
	y_estimate = pd.read_csv(filepath_est, index_col=0)
	y_pred = TTE_to_event(y_estimate)

	filepath_real = 'C:\\Users\\alexw\\Desktop\\Harvard_data\\0_1_2\\FS_time\\NDR0208ITLaMaRl01_y.csv'
	y_real = pd.read_csv(filepath_real, index_col=0)
	y_real = TTE_to_event(y_real)

	score = r2_score(y_real, y_pred)

	a = 1
