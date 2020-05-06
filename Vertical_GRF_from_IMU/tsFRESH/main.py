from feature_extraction import extract_data
from model_fitting import learn
from utils import get_directory, load_features

import time
import numpy as np
import pandas as pd

if __name__ == "__main__":
	
	# HS or TO
	events = ['HS', 'TO']
	event_types = ['time']

	data_folder = "C:\\Users\\alexw\\Desktop\\tsFRESH\\data\\"
	
	# columns in X = ['id', 'time', 'ax_l', 'ay_l', 'az_l', 'ax_r', 'ay_r', 'az_r',
	# 				'ax_diff', 'ay_diff', 'az_diff', 'a_res_l', 'a_res_r', 'a_res_diff']
	columns_entire = [
						#['id', 'time', 'ax_l', 'ay_l', 'az_l', 'ax_r', 'ay_r', 'az_r'], # Just raw data
						['id', 'time', 'ax_diff', 'ay_diff', 'az_diff', 'a_res_diff'], # Differences in raw data
						#['id', 'time', 'a_res_l', 'a_res_r'], # Resultants
					#	['id', 'time', 'ax_l', 'ay_l', 'az_l', 'ax_r', 'ay_r', 'az_r', 'ax_diff', 'ay_diff', 'az_diff', 'a_res_l', 'a_res_r', 'a_res_diff'], # All
						#['id', 'time', 'ax_diff', 'ay_diff', 'az_diff', 'a_res_l', 'a_res_r', 'a_res_diff'],
						#['id', 'time', 'ax_l', 'ay_l', 'az_l', 'a_res_l'],  # Only left foot
						#['id', 'time', 'ax_r', 'ay_r', 'az_r', 'a_res_r'] # Only right foot
						]

	columns_entire = [['ax_l'],
					['ay_l'], 
					['az_l'],
					['ax_r'], 
					['ay_r'], 
					['az_r'],
	 				['ax_diff'], 
					['ay_diff'], 
					['az_diff'], 
					['a_res_l'], 
					['a_res_r'], 
					['a_res_diff']]

	start_time = time.time()

	overlap = True
	'''
	for columns in columns_entire:
		for event in events:
			for event_type in event_types:
				start_time_i = time.time()

				print("Running using: {} for {} with {}".format(columns, event, event_type))

				extract_data(data_folder=data_folder, columns=columns, all=True, est_events=True, event=event, event_type=event_type, overlap=overlap)

				#directory = get_directory(initial_directory=data_folder, columns=columns, est_events=True, event=event, event_type=event_type)

				# Load features (after extract data has been run)
				#X_dictionary, y_dictionary, groups = load_features(data_folder, directory, est_events=True, overlap=overlap)

				#learn(X_dictionary, y_dictionary, directory, groups)

				end_time_i = time.time()
				print('Run time for trial = {}'.format(end_time_i - start_time_i))
	
	end_time = time.time()

	print('Total run time = {}'.format(end_time - start_time))
	
	'''
	cols = [
		['id', 'time', 'ax_l', 'ay_l', 'az_l', 'ax_r', 'ay_r', 'az_r'],
		['id', 'time', 'ax_diff', 'ay_diff', 'az_diff', 'a_res_l', 'a_res_r', 'a_res_diff']
	]


	# Columns that we want to train on
	cols = [ 'ax_l', 'ay_l', 'az_l', 'ax_r', 'ay_r', 'az_r', 'a_res_l', 'a_res_r']

	for event in events:
			for event_type in event_types:
				x = []
				
				for col in cols:
					directory = get_directory(initial_directory=data_folder, columns=col, est_events=True, event=event, event_type=event_type)

					# Load features (after extract data has been run)
					X_dictionary, y_dictionary, groups = load_features(data_folder, directory, est_events=True)

					x.append(X_dictionary)

				X = {}

				for k in X_dictionary.keys():
					concat_list = []

					for idx in x:
						concat_list.append(idx[k])

					X[k] = pd.concat(concat_list, axis=1)
				
				y = y_dictionary

				# See if directory for merge exists, and create if not
				directory = get_directory(initial_directory=data_folder, columns=cols, est_events=True, event=event, event_type=event_type)

				# Train
				learn(X, y, directory, groups)
