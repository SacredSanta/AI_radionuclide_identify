import pickle
import numpy as np
from joblib import Parallel, delayed, cpu_count
import lzma
import matplotlib.pyplot as plt
import pandas as pd
from modwt import modwt, imodwt

# signal processing function
def noisefiltering(data, outlayer, shift):
    layer = 6
    wavelet = 'haar'
    output = np.zeros(data.shape)
    for i in range(data.shape[0]):
        '''major signal extraction by using modwt method'''
        coefficient = modwt(data[i],wavelet,layer)
        output[i] = imodwt(coefficient[layer-outlayer:layer+1,:],wavelet)
        output[i] = np.roll(output[i],shift)
        '''scaling for preservation of signal data'''
        max_val_out = np.max(output[i])
        output[i] = output[i] / max_val_out
        '''thresholding unavailable data'''
        output[i][output[i]<0] = 0
    return output

def noisefiltering2(data,outlayer,shift):
    layer = 6
    wavelet = 'haar'
    output = np.zeros(data.shape)
    for i in range(data.shape[0]):
        '''major signal extraction by using modwt method'''
        coefficient = modwt(data[i],wavelet,layer)
        output[i] = imodwt(coefficient[layer-outlayer:layer+1,:],wavelet)
        output[i] = np.roll(output[i],shift)
        '''scaling for preservation of signal data'''
        max_val_out = np.max(output[i])
        output[i] = output[i] / max_val_out

    return output

def derivative_signal(data):
    output = np.zeros(np.shape(data))
    output[:,1:999] += (data[:,1:999] - data[:,0:998])
    return output