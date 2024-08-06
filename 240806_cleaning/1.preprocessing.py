'''
최종 수정 : 2024.08.06.
사용자 : 서동휘

<수정 내용> 

<24.08.06>
- 중간에 process_filtering_1data 에 poisson deviance 를 구하는 부분 추가
- poisson deviance 계산을 위한 background ut load 도 포함.
'''

















#%% ---------------------------------------------------------
# ------------------  import setting -------------------------
# -------------------------------------------------------------
import sys
lib_direc = "./lib/"
if lib_direc not in sys.path:
    sys.path.append(lib_direc)

import pickle
import numpy as np
from joblib import Parallel, delayed, cpu_count
import lzma
import matplotlib.pyplot as plt
import pandas as pd
from modwt_pkg import *
from modi_hist_extract import modi_hist_extract
from poisson_deviance import my_pos_dev



#%% ---------------------------------------------------------
# --------------------- make for TF data ----------------------
# -------------------------------------------------------------

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



'''
====================================================================================
'''

# background histogram : ut - for poisson deviance
RA__ = "background"
dtpath__ = f"../../Data/240603_nucare/newmodi/{RA__}_close_5min.csv"
ut_dt = modi_hist_extract(dtpath__)
ut = ut_dt.hist
ut_hat = ut/sum(abs(ut))



RA_2 = "Background"
num__ = np.random.randint(1, 490, size=1)
dtpath_2 = f"../../Data/230000_forTensorflow_xz/{RA_2}/{RA_2}_{num__[0]}.xz"
with lzma.open(dtpath_2, 'rb') as f:
    temp__ = pickle.load(f)
ut2 = temp__.reshape(1,1000)
ut2 = ut2[0]
ut2_hat = ut2/sum(abs(ut2))


def process_filtering_1data(task):
    global pre_data
    global ut2_hat
    
    mixed_spectrum = pre_data[task, :, :]    # shape (1,1000)
    mixed_spectrum = mixed_spectrum / (np.max(mixed_spectrum))
    filtered1 = noisefiltering(mixed_spectrum, 0, 0)
    derivatives1 = derivative_signal(filtered1)
    filtered2 = noisefiltering2(derivatives1, 0, 0)
    pos_dev = my_pos_dev(mixed_spectrum[0,:], ut2_hat)
    
    x_temp = np.zeros((1, 1, 1000, 3))
    x_temp[0, :, :, 0] = mixed_spectrum
    x_temp[0, :, :, 1] = filtered2
    x_temp[0, :, :, 2] = pos_dev
    
    return x_temp


# 0.dataset에 있는 npy로 최종 model input용 dataset 생성.
def process_dataset_csv(start:int, end:int):
    num_cores = cpu_count()
    
    tasks = range(start, end)
    #x_component = []
    results = Parallel(n_jobs=num_cores, verbose=10)(delayed(process_filtering_1data)(task) for task in tasks)
    #for task in tasks:
    #    x_temp = process_filtering_1data(task)
    #    x_component.append(x_temp)
    return results




def process_all(train_n, val_n, test_n):
    global pre_data
    
    train_flag = 0
    vali_flag = 0
    test_flag = 1
    
    train_s = 0
    train_e = train_n
    
    val_s = train_n
    val_e = val_s + val_n
    
    test_s = val_e
    test_e = test_s + test_n

    # train
    if train_flag:
        results1 = process_dataset_csv(train_s, train_e)
    else:
        results1 = None
    
    # validation
    if vali_flag:
        results2 = process_dataset_csv(val_s, val_e)
    else:
        results2 = None
    
    # test
    if test_flag:
        results3 = process_dataset_csv(test_s, test_e)
    else:
        results3 = None


    # Step 2
    #* return 될 list 초기화
    x_train = []
    x_val = []
    x_test = []

    # result로 나온 값들 넣어주기.
    # ? extend list 값의 요소를, 요소로 추가.
    if train_flag:
        for x_comp in results1:
            x_train.extend(x_comp)
        x_train = np.vstack(x_train)
        y_train = pre_data_y[train_s:train_e]
    else:
        y_train = []
    
    if vali_flag:
        for x_comp in results2:
            x_val.extend(x_comp)
        x_val = np.vstack(x_val)
        y_val = pre_data_y[val_s:val_e]
    else:
        y_val = []
    
    if test_flag:
        for x_comp in results3:
            x_test.extend(x_comp)
        x_test = np.vstack(x_test)
        y_test = pre_data_y[test_s:test_e]
    else:
        y_test = []

    
    print("Work Done!")
    return x_train, y_train, x_val, y_val, x_test, y_test


'''
<<< main >>>
'''

#%% 1. "0.dataset"에서 data load ===========================================================
filename = "240806_4source_3000_xzfile"
pre_data = np.load(f"./0.dataset/{filename}.npy")   # (datanum, 1, 1000)
pre_data_y = np.load(f"./1.preprocess_data/{filename}_y.npy")  # [[4개] x datanum]





#%% 2. MAKE DATA ===========================================================
train_n = 0
val_n = 0
test_n = 3000
x_train, y_train, x_val, y_val, x_test, y_test = process_all(train_n, val_n, test_n)    # 2.4분만에 완료!





#%% 3. save data ===========================================================
# %% (선택!) save the only one kind of data
np.save(f"./1.preprocess_data/{filename}_xtest.npy", x_test)

# %% (선택!) 만약 모든 것을 저장한다면,
np.savez(f"./1.preprocess_data/{filename}_all.npz", 
         x_train=x_train,
         y_train=y_train,
         x_val=x_val,
         y_val=y_val,
         x_test=x_test,
         y_test=y_test)
# %%
