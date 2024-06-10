#%% ---------------------------------------------------------
# ------------------  import setting -------------------------
# -------------------------------------------------------------
import pickle
import numpy as np
from joblib import Parallel, delayed, cpu_count
import lzma
import matplotlib.pyplot as plt
import pandas as pd



#%% ---------------------------------------------------------
# --------------------- make for TF data ----------------------
# -------------------------------------------------------------

#%% function ---------------------------------------------------

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


# gate simulation 에서 얻은 data로 npy 생성
def process_single_component_gate(components:list, iter_num:int)->list:
    x_component = []
    y_component = []

    for _ in range(iter_num):
        if _ % 100 == 0: print((_//100)*100,"iter passed")
    # -----------------------------
        '''
        mixed_spectrum = np.zeros((1000, 1))
        
        for i in range(len(components)):
            nu_data = np.random.randint(4, 400, size=1)
            with lzma.open('/tf/latest_version/3.AI/Tensorflow/Data/single/{}/{}_{}.xz'.format(components[i],components[i],nu_data[0]), 'rb') as f:
                temp = pickle.load(f)
                mixed_spectrum += temp
        # ... [rest of the processing] ...
        mixed_spectrum = np.reshape(mixed_spectrum,(1,1000))/(np.max(mixed_spectrum))
        filtered1 = noisefiltering(mixed_spectrum,0,0)
        derivatives1 = derivative_signal(filtered1)
        filtered2 = noisefiltering2(derivatives1,0,0)
        derivatives2 = derivative_signal(filtered2)
        filtered3 = noisefiltering2(derivatives2,0,0) #DEPRECATED

        x_temp = np.zeros((1, 1, 1000, 2))  # numpy array 차원은 앞으로 붙음
        y_temp = np.zeros((1, 4))
        x_temp[0, :, :, 0] = mixed_spectrum
        # x_temp[0, :, :, 1] = filtered1
        x_temp[0, :, :, 1] = filtered2
        
        for i in range(len(components)):
            for j, k in enumerate(['Ba133', 'Na22', 'Cs137', 'Background']):
                if components[i] == k:
                    y_temp[0, j] += 1
        '''
    # ----------------------------------
    # count가 random으로 하나의 case에 대해서 3000번을 뽑는거다
    
        # component = [0,1,2] source number가 들어있는 list 형태로 옴.
        random_count = np.random.choice(range(900,2000), 1)
        mixed_spectrum = dtt_fil.make_rancombi(components, random_count)   
            #? dtt_fil -> filtered 된 data (class)
            #? make_rancombi -> return self.histogram
        mixed_spectrum = mixed_spectrum / (np.max(mixed_spectrum))
        filtered1 = noisefiltering(mixed_spectrum, 0, 0)
        derivatives1 = derivative_signal(filtered1)
        filtered2 = noisefiltering2(derivatives1, 0, 0)
        
        x_temp = np.zeros((1, 1, 1000, 2))
        x_temp[0, :, :, 0] = mixed_spectrum
        x_temp[0, :, :, 1] = filtered2
        
        y_temp = np.zeros((1, 11))
        for i in components:
            y_temp[0,i] = 1
                
        x_component.append(x_temp)
        y_component.append(y_temp)

    return x_component, y_component


def process_all():
    num_cores = cpu_count()  # or set to the number of cores you want to use

    train_flag = 0
    vali_flag = 0
    test_flag = 1
    
    train_num = 3000
    vali_num = 1000
    test_num = 1000


    # Step 1
    #* src_combi : source combination, 2d array, 각 조합별 source 지정 번호로
    #* mixing_list의 value가 components
    
    # train
    if train_flag:
        results1 = Parallel(n_jobs=num_cores)(delayed(process_single_component_gate)(components, train_num) for components in src_combi) 
    else:
        results1 = None
    
    # validation
    if vali_flag:
        results2 = Parallel(n_jobs=num_cores)(delayed(process_single_component_gate)(components, vali_num) for components in src_combi)
    else:
        results2 = None
    
    # test
    if test_flag:
        results3 = Parallel(n_jobs=num_cores)(delayed(process_single_component_gate)(components, test_num) for components in src_combi)
    else:
        results3 = None


    # Step 2
    #* return 될 list 초기화
    x_train = []
    y_train = []
    x_val = []
    y_val = []
    x_test = []
    y_test = []

    # result로 나온 값들 넣어주기.
    # ? extend list 값의 요소를, 요소로 추가.
    for x_comp, y_comp in results1:
        x_train.extend(x_comp)
        y_train.extend(y_comp)
    for x_comp, y_comp in results2:
        x_val.extend(x_comp)
        y_val.extend(y_comp)
    for x_comp, y_comp in results3:
        x_test.extend(x_comp)
        y_test.extend(y_comp)

    # 종합된 result를 row 방향으로 정리.
    x_train = np.vstack(x_train)
    y_train = np.vstack(y_train)
    x_val = np.vstack(x_val)
    y_val = np.vstack(y_val)
    x_test = np.vstack(x_test)
    y_test = np.vstack(y_test)

    return x_train, y_train, x_val, y_val, x_test, y_test