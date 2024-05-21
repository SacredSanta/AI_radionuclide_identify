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
    def make_uni(self, num:list, count:int):
        # histogram 초기화
        self.histogram = np.zeros([1,1000], dtype=int)
        
        seed = len(num)
        
        # 각 src마다 count 개수가 달라 idx를 동일하게 보면서 가면 bound 넘음.
        idx = np.zeros([1,seed], dtype=int)
        
        # num -> [0,3,5 ...]  source list
        # seed -> source list의 index
        # idx -> 특정 source에서 energy array 안에서의 index
                
        while int(sum(self.histogram[0]))!=count:
            data_idx = self.idx_dict[str(num[seed-1])][idx[0,seed-1]]
                
            # 각 idx에 해당하는 energy값 histogram에 더해주기
            ene = int(round(float(self.energy[data_idx])*1000,0))
            if ene < 0 or ene > 999:
                continue
            else:
                self.histogram[0,ene] += 1
            
            idx[0,seed-1] += 1   # 특정 source energy array의 다음 index로    
            seed += 1
            if seed > len(num) : seed = 1
        
        return self.histogram
        
        
        
    # 고른 source 중 count 까지 random하게 선별
    def make_rancom(self, num:list, count:int):
        pass
        
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
dtt_fil.make_uni([0,1,2], 1000)
dtt_fil.show()





#%% (extra) import setting ----------------------------------
import sys
sys.path.append("/tf/latest_version/3.AI/Tensorflow/Data/")
from tf_makedata import *

#%% make for TF data -------------------------------------------



def process_single_component_gate(components,iter_num):
    x_component = []
    y_component = []
    for _ in range(iter_num):
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
        
        x_component.append(x_temp)
        y_component.append(y_temp)

    return x_component, y_component


































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
