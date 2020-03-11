import tensorflow as tf
from tensorflow import keras

import numpy as np
from matplotlib import pyplot
import sys

from matplotlib import pyplot as plt

# Disables the tensorflow AVX2 warning, doesn't enable AVX2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from utils import construct_model, train_model, save_model


if __name__ == '__main__':

	# Load datasets and true outputs
	dataset = np.load(file="C:\\Users\\alexw\\Dropbox\\ABI\\Level_8_Lab\\Vertical GRF from IMU\\dataset.npy", allow_pickle=True)
	HS_TO = np.load(file="C:\\Users\\alexw\\Dropbox\\ABI\\Level_8_Lab\\Vertical GRF from IMU\\HS_TO.npy", allow_pickle=True)