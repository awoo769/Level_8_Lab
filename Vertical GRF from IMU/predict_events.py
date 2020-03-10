import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed, Dropout, Bidirectional

import numpy as np
from numpy import array
from random import random
from numpy import cumsum
from matplotlib import pyplot
from pandas import DataFrame
import sys

from matplotlib import pyplot as plt

# Disables the tensorflow AVX2 warning, doesn't enable AVX2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def construct_model(hidden = 32, lstm_layers = 2, input_dim = 6, output_dim = 1):
    model = Sequential()
    model.add(LSTM(input_shape=(None, input_dim), return_sequences=True, units=hidden))

    for _ in range(lstm_layers-1):
        model.add(LSTM(return_sequences=True, units=hidden))

    model.add(TimeDistributed(Dense(output_dim, activation='sigmoid')))
    model.compile(loss=weighted_binary_crossentropy, optimizer='sgd', metrics=['accuracy'])

    return model


def weighted_binary_crossentropy(y_true, y_pred):
	# Assume this is HS/FS (foot strike)
	a1 = tf.keras.backend.mean(tf.keras.backend.binary_crossentropy(y_pred[0,:], y_true[0,:]) * (y_true[0,:] + 0.01), axis=-1) # try + 0.001
	
	# Assume this is TO/FO (foot off)
	a2 = tf.keras.backend.mean(tf.keras.backend.binary_crossentropy(y_pred[1,:], y_true[1,:]) * (y_true[1,:] + 0.01), axis=-1)

	return a1 + a2


def plot_history(history):
    nepoch = len(history.history['loss'])

    plt.plot(range(nepoch),history.history['loss'],'r')
    plt.plot(range(nepoch),history.history['val_loss'],'b')

    axes = plt.gca()
    axes.set_ylim([0.001,0.005])

    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()

def peakdet(v, delta, x = None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html
    
    Returns two arrays
    
    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %      
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.
    
    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.
    
    """
    maxtab = []
    mintab = []
       
    if x is None:
        x = np.arange(len(v))
    
    v = np.asarray(v)
    
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    
    if not np.isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    
    if delta <= 0:
        sys.exit('Input argument delta must be positive')
    
    mn, mx = np.Inf, -np.Inf
    mnpos, mxpos = np.NaN, np.NaN
    
    lookformax = True
    
    for i in np.arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        
        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return array(maxtab), array(mintab)


def peak_cmp(annotated, predicted):
    dist = []
    if len(predicted) == 0 or len(annotated) == 0:
        return -1
    if len(predicted) != len(annotated):
        return -1
    
    for a in annotated:
        dist = dist + [min(np.abs(predicted - a))]
    if not len(dist):
        return -1
    return min(dist)


def eval_prediction(likelihood, true, patient, plot = True, shift = 0):
    sdist = []
    
    peakind = peakdet(likelihood[:,0],0.5)
    for k,_ in peakind[0]:
        if plot:
            plt.axvline(x=k)
    sdist.append(peak_cmp(np.where(true[:,0] > 0.5)[0], [k + shift for k,v in peakind[0]]))

    if plot:
        plt.plot(likelihood) # continous likelihood process
        plt.plot(true) # spikes on events
        plt.title(patient)
        axes = plt.gca()
        axes.set_xlim([0,true.shape[0]])
        plt.show()
    return sdist


if __name__ == '__main__':

	model = construct_model()

	# Load datasets and true outputs
	dataset = np.load(file="C:\\Users\\alexw\\Dropbox\\ABI\\Level_8_Lab\\Vertical GRF from IMU\\dataset.npy", allow_pickle=True)
	HS_TO = np.load(file="C:\\Users\\alexw\\Dropbox\\ABI\\Level_8_Lab\\Vertical GRF from IMU\\HS_TO.npy", allow_pickle=True)

	# Train on 90 % of data, test on 10 %
	# For now, use bottom 10 %.

	n_samples = len(dataset)

	n_training = int(n_samples * 0.9)

	training = dataset[:n_training]
	validation = dataset[n_training:]

	training_truths = HS_TO[:n_training]
	validation_truths = HS_TO[n_training:]

	# Each timestep has 6 "features" (ax_L, ay_L, az_L, ax_R, ay_R, az_R)
	# Shape of dataset = (n, 636, 6)

	# To plot:
	# plt.plot(training[0])
	# plt.plot(training_truths[0][0],(training[0][:])[training_truths[0][0]],'o','g') # HS
	# plt.plot(training_truths[0][1],(training[0][:])[training_truths[0][1]],'o','r') # TO

	a = 1

	# Constrained peak detection algorithm for RNN - structured prediction model TODO - see if this is needed first
	# IC and TO event of the same foot are speparated by at least 35 ms and at most 200 ms
	# TO and IC event of opposing feet are separated by at least 160 ms and at most 350 ms
