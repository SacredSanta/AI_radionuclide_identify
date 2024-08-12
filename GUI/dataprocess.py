'''
최종 수정 : 2024.05.28.
사용자 : 서동휘

<수정 내용> 

<처음>
MODWT package 와 연결되어있는 noise filtering 관련 코드
'''


import pickle
import numpy as np
from joblib import Parallel, delayed, cpu_count
import lzma
import matplotlib.pyplot as plt
import pandas as pd
from modwt_pkg import modwt, imodwt

# signal processing function
def noisefiltering(data, outlayer, shift):
    # data shape should be 2 dimensional 
    layer = 6
    wavelet = 'haar'
    output = np.zeros(data.shape)
    for i in range(data.shape[0]):
        '''major signal extraction by using modwt method'''
        coefficient = modwt(data[i],wavelet,layer) #  layer만큼 행이 나옴.(layer, datalength)
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


def my_pos_dev(xt, ut_hat):
    epsilon = 1e-5
    return ( 2 * (xt * np.log(xt / (sum(abs(xt)) * ut_hat + epsilon) + epsilon))
                - (xt - (sum(abs(xt)) * ut_hat)))