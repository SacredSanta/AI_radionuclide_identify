#%% 0.초기화 -------------------------------------------
import numpy as np
import os
import matplotlib.pyplot as plt

debug = 1

dir_ = "/tf/latest_version/3.AI/Tensorflow/Data/simulation/allmix"
id = 1065

pathALL = os.path.join(dir_, "id{}.Singles.npy".format(id))

dt = np.load(pathALL)
dtype_list = [(field_name, dt.dtype[field_name]) for field_name in dt.dtype.fields]

#%% 1. class 정의 및 simulation data instance화  -------------------------------------------

# single.npy를 class로 정리.
class SinglesData():
    def __init__(self, dt, dtype_list):
        for i in range(len(dtype_list)):
            temp = dtype_list[i][0]
            setattr(self, dtype_list[i][0], dt[dtype_list[i][0]])
        self.dtype = dtype_list
        
    def info(self):
        print(dtype_list)
        print("shape : ", dt.shape)


class filteredData():  # 선원 번호 지정 부분만 빼오기
    def __init__(self, filter_source_idx:dict, data=dtt):
        self.energy = dtt.energy
        self.idx_dict = filter_source_idx
        
    # 고른 source 모두 histogram에 저장    
    # ? .sortall([filter할 sourceID])
    def sortall(self, num:list):    
        # histogram 초기화
        self.histogram = np.zeros([1,1000])
        # 여러 source에 대해서 반복
        for srcnum in num:
            # 한 source가 해당되는 id
            idx = self.idx_dict[str(srcnum)]
            
            # 각 idx에 해당하는 energy값 histogram에 더해주기
            for i in idx:
                ene = int(round(float(self.energy[i])*1000,0))
                if ene < 0 or ene > 999:
                    continue
                else:
                    self.histogram[0,ene] += 1
        
        return self.histogram
    
    
    
    # 고른 source 정한 count 까지만 저장
    # ? .sortall([filter할 sourceID], 저장할 총 카운트 수)
    def make_uni(self, components:list, count:int):
        # histogram 초기화
        self.histogram = np.zeros([1,1000], dtype=int)
        
        seed = len(components)
        
        # 각 src마다 count 개수가 달라 idx를 동일하게 보면서 가면 bound 넘음.
        idx = np.zeros([1,seed], dtype=int)
        
        # num -> [0,3,5 ...]  source list
        # seed -> source list의 index
        # idx -> 특정 source에서 energy array 안에서의 index
                
        while int(sum(self.histogram[0]))!=count:
            data_idx = self.idx_dict[str(components[seed-1])][idx[0,seed-1]]
                
            # 각 idx에 해당하는 energy값 histogram에 더해주기
            ene = int(round(float(self.energy[data_idx])*1000,0))
            if ene < 0 or ene > 999:
                continue
            else:
                self.histogram[0,ene] += 1
            
            idx[0,seed-1] += 1   # 특정 source energy array의 다음 index로    
            seed += 1
            if seed > len(components) : seed = 1
        
        return self.histogram
                
        
        
    # 고른 source 중 count 까지 random하게 선별
    def make_rancombi(self, components:list, count:int):
        # histogram 초기화
        self.histogram = np.zeros([1,1000], dtype=int)
        
        while int(sum(self.histogram[0]))!=count:
            for src in components:
                idx_ = np.random.choice(len(self.idx_dict[str(src)])-1)  # 해당 source의 data의 index             
            
                data_idx = self.idx_dict[str(src)][idx_]
                
                ene = int(round(float(self.energy[data_idx])*1000,0))
                if ene < 0 or ene > 999:
                    continue
                else:
                    self.histogram[0,ene] += 1
                
                
                if int(sum(self.histogram[0]))==count : return self.histogram
                
        return self.histogram
        
        
        
    def show(self):   # 히스토그램 plot
        plt.plot(self.histogram[0])

        
dtt = SinglesData(dt, dtype_list)
dtt.info()
print(dtt.energy)

#%% sourceID 별 index 정리 ---------------------------------------------------------

# gate 기준 source name으로 filter할 source를 지정해야함.
src_name = list(set(dtt.sourceID))
filter_source_idx = {}
for srcnum in src_name:
    filter_source_idx[str(srcnum)] = np.where(dtt.sourceID==srcnum)[0].astype(int)

# filteredData에 초기화용 argument로 target_idx를 넣어줘야함.
dtt_fil = filteredData(filter_source_idx, dtt)
#%%  using sortall  -------------------------------------------
dtt_fil.sortall([0])
dtt_fil.show()

#%%  using make_uni -------------------------------------------
a = dtt_fil.make_uni([0,1,2], 1000)
dtt_fil.show()

#%%  using make_ran
b = dtt_fil.make_rancombi([0], 1000)
dtt_fil.show()




#%% (extra) import setting ----------------------------------
import sys
sys.path.append("/tf/latest_version/3.AI/Tensorflow/Data/")
from tf_makedata import *
from modwt import *

import pickle
import numpy as np
from joblib import Parallel, delayed, cpu_count
import lzma
import matplotlib.pyplot as plt
import pandas as pd

#%% make for TF data -------------------------------------------
# -------------------------------------------------------------
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


def process_single_component_gate(components:list, iter_num:int):
    x_component = []
    y_component = []

    for _ in range(iter_num):
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
        mixed_spectrum = dtt_fil.make_rancombi(components, 1000)   # * random으로 하는거 구현해야함.
        mixed_spectrum = mixed_spectrum / (np.max(mixed_spectrum))
        filtered1 = noisefiltering(mixed_spectrum, 0, 0)
        derivatives1 = derivative_signal(filtered1)
        filtered2 = noisefiltering2(derivatives1, 0, 0)
        derivatives2 = derivative_signal(filtered2)
        
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

    # train
    results1 = Parallel(n_jobs=num_cores)(delayed(process_single_component_gate)(components, 3000) for components in src_combi) # mixing_list의 value가 components
    # validation
    results2 = Parallel(n_jobs=num_cores)(delayed(process_single_component_gate)(components, 1000) for components in src_combi)
    # test
    results3 = Parallel(n_jobs=num_cores)(delayed(process_single_component_gate)(components, 1000) for components in src_combi)

    # mixed_list로 0~9 까지의 각각 조합이 있는 list가 담긴 list로 하는게 나을듯.


    x_train = []
    y_train = []
    x_val = []
    y_val = []
    x_test = []
    y_test = []

    for x_comp, y_comp in results1:
        x_train.extend(x_comp)
        y_train.extend(y_comp)
    for x_comp, y_comp in results2:
        x_val.extend(x_comp)
        y_val.extend(y_comp)
    for x_comp, y_comp in results3:
        x_test.extend(x_comp)
        y_test.extend(y_comp)

    x_train = np.vstack(x_train)
    y_train = np.vstack(y_train)
    x_val = np.vstack(x_val)
    y_val = np.vstack(y_val)
    x_test = np.vstack(x_test)
    y_test = np.vstack(y_test)

    return x_train, y_train, x_val, y_val, x_test, y_test


#%% 1. init of source ------------------------------------------------------------------

# mix single 
'''
mixing_list = {'Ba':['Ba133'],
            'Na':['Na22'],
            'Cs':['Cs137'],
            'BaCs':['Ba133','Cs137'],
            'BaNa':['Ba133','Na22'], 
            'CsNa':['Cs137','Na22'],
            'BaNaCs':['Ba133','Na22','Cs137'],
            'BG':['Background']
           }

mixed_source_name = list(mixing_list.keys())
mixed_list = list(mixing_list.values())
'''

# gate source information
# Ba133, Cs137, Am241, Co60, Ga67, I131, Ra226, Tc99m, Th232, Tl201, Na22
import itertools

#source_seed = [i for i in range(0,10)]
source_seed = [0,1]
src_combi = []
for i in range(1,len(source_seed)+1):
    for combi in itertools.combinations(source_seed, i):
        src_combi.append(list(combi))

print(src_combi)
        



#%% 2. make data ------------------------------------------------------------------
# Then, you would call process_all to start the processing:
x_train, y_train, x_val, y_val, x_test, y_test = process_all()

#print(x_train.shape)
np.savez('spectrum_off_gate_1.npz', x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test)





























#%% test --------------------------------------------------------------
import numpy as np
import os

debug = 1

dir_ = "C:/Users/MIPL/Desktop/gatetest/mac/data"
id = 1050

if debug:
    print("id{}.txtSingles.dat".format(id))

pathAll = os.path.join(dir_, "id{}.txtSingles.dat".format(id))

if debug:
    #print(pathAll)
    print(type(pathAll))

with open(pathAll, 'r') as f:
    ff = f.read()
    #print(ff)
    if debug:
        print(type(ff))

#%%
import numpy as np
import os

debug = 1

dir_ = "C:/Users/MIPL/Desktop/gatetest/mac/data"
id = 1050

if debug:
    print("id{}.txtSingles.dat".format(id))

pathAll = os.path.join(dir_, "id{}.Singles.npy".format(id))

dt = np.load(pathAll)

print(dt)
