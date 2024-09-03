#%%
import numpy as np
import os
import lzma
import pickle
import matplotlib.pyplot as plt
data = np.load("/tf/latest_version/3.AI/Tensorflow/Data/spectrum_off1.npz")
x_train = data['x_train']
y_train = data['y_train']
x_val = data['x_val']
y_val = data['y_val']
x_test = data['x_test']
y_test = data['y_test']


plt.plot(x_train[15000,:,:,0].flatten())
# x_train = np.reshape(x_train[:,:,:,2],(24000,1,1000,1))
# x_val = np.reshape(x_val[:,:,:,2],(8000,1,1000,1))
# Check data

# %% simulation
debug = 0

def gate_data_conversion(dpath):
    global debug
    dt = np.zeros((1,1000))
    if debug: print(dt)
    with open(dpath, 'r') as f:
        for line in f:
            dtlist = list(line.split())[:-2]
            if debug:print("--- gate original data ---"); print(float(dtlist[-8]), len(dtlist))
            '''
            dtlist[2] source id
            dtlist[3] x source
            dtlist[4] y source
            dtlist[5] z source
                        
            dtlist[-9] time
            dtlist[-8] energy
            dtlist[-7] x position of detector
            dtlist[-6] y position of detector
            dtlist[-5] z position of detector                    
            '''
            idx = int(round(float(dtlist[-8])*1000,0))   # 나중에 이 floor 한 것이 data 신뢰도에 영향을 줄 수 있음.
            if debug: print("idx : ",idx); break;
            if not idx>=1000 and not idx<0:
                dt[:,idx] += 1
    return dt
            
        
components = ["Ba133"]
i=0
nu_data=[100]

#%% gate data plot ---------------------------------------------------------------------
debug = 0
datadir = "/tf/latest_version/3.AI/Tensorflow/Data/simulation"
datapath = "Ba133/id1061.dat"

pathh = os.path.join(datadir,datapath)
if debug:print(pathh)
dt = gate_data_conversion(pathh)
if debug:print(dt)

plt.plot(dt[0,:])
plt.show()
# 시뮬레이션 데이터가 정상적으로 안나옴/??
print("max : ", max((dt[0,:])))
print("count : ", sum(dt[0,:]))

#%% gate data with nz background--------------------------------------------------------------------
idname = 'id1061'
datadir = "/tf/latest_version/3.AI/Tensorflow/Data/"
datapath = "simulation/Ba133/{}.dat".format(idname)

pathh = os.path.join(datadir,datapath)

dt = gate_data_conversion(pathh)

print(max(dt[0]))
print(sum(dt[0]))
npzpath = "single/"
components = ["Background"]
dtname = "{}/{}_{}.xz".format(components[0], components[0],10)
pathh2 = os.path.join(datadir,npzpath,dtname)

with lzma.open(pathh2, 'rb') as f:
    temp = pickle.load(f)
    final = dt + temp.reshape(1,1000)
    print(sum(max(temp)))

saveflag = 0

if saveflag:
    savepath = os.path.join(datadir, "simulation/Ba133/withBack/{}_back.npz".format(idname))
    print(savepath)
    with open(savepath,'wb') as f:
        np.save(f, final)
plt.plot(final[0])
plt.show()



#%% check npz data--------------------------------------------------------------------
from IPython import display
import numpy as np

datadir = "/tf/latest_version/3.AI/Tensorflow/Data/"

npzpath = "single/"
components = ["Ba133"]

superposition = 0
arr = np.zeros([1000,1])
superpositioncut = 1  # 신호 합칠 개수

for datanum in range(5,1000):
    dtname = "{}/{}_{}.xz".format(components[0], components[0],datanum)
    pathh2 = os.path.join(datadir,npzpath,dtname)
    superposition += 1
    with lzma.open(pathh2, 'rb') as f:
        temp = pickle.load(f)
        arr += temp
        if superposition == superpositioncut:
            plt.plot(arr)
            plt.ylim([0,30])
            display.display(plt.gcf())
            plt.clf()
            print("total : ", sum(arr))
            arr = np.zeros([1000,1])
            superposition = 0
    display.clear_output(wait=True)
    
# %%  gate npy data plot
import numpy as np
import matplotlib.pyplot as plt
from unpack_simul_numpy import *

data = np.load("/tf/latest_version/3.AI/Tensorflow/Data/simulation/allmix/id1066.all_50cm.npy")

plt.plot(data[])

# %%
import numpy as np
import matplotlib.pyplot as plt

data = np.load("/tf/latest_version/3.AI/Tensorflow/Data/spectrum_off1.npz")

data1 = data["x_train"][5000]
#%%
import sys
sys.path.append()
plt.plot(data1[0,:,1])
plt.title("Derivative of Energy Spectrum", fontsize=18)
plt.xlabel('Energy(keV)', fontsize=14)
plt.ylabel('Value(A.U.)', fontsize=14)
















#%% 
#%% 0.초기화 -------------------------------------------
import numpy as np
import os
import matplotlib.pyplot as plt

debug = 1

dir_ = "/tf/latest_version/3.AI/Tensorflow/Data/simulation/allmix"
id = 1065

pathALL = os.path.join(dir_, "id{}.all_10cm.npy".format(id))

dt = np.load(pathALL)
dtype_list = [(field_name, dt.dtype[field_name]) for field_name in dt.dtype.fields]

#%% 1. class 정의 및 simulation data instance화  ------------------------------------------- 

# single.npy를 class로 정리.
# DEPRECATED! 그냥 energy랑 ID만 뽑아서 써야겠다.
class SinglesData_old():
    def __init__(self, dt, dtype_list, additional_data=None):
        for i in range(len(dtype_list)):
            temp = dtype_list[i][0]
            setattr(self, dtype_list[i][0], dt[dtype_list[i][0]])
        self.dtype = dtype_list
        
    def info(self):
        print(dtype_list)
        print("shape : ", dt.shape)
        
    def concatenate(self, newdata):
        for attr in dtype_list:
            pass


# 1.1 get Single
class SinglesData():
    def __init__(self, dt, dtype_list, additional_data=None, *more_data):
        newnum = 10      
        print(type(additional_data))
        try:
            print("additional data inserted")
            self.sourceID = np.concatenate( (dt["sourceID"], newnum*np.ones(len(additional_data["sourceID"]), dtype=int)) )
            self.energy = np.concatenate( (dt["energy"], additional_data["energy"]) )
        except:
            self.sourceID = dt["sourceID"]
            self.energy = dt["energy"]
            

            
# 1.2 filter the data
class filteredData():  # 선원 번호 지정 부분만 빼오기
    def __init__(self, filter_source_idx:dict, data):
        self.energy = data.energy
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


add_dt = np.load("/tf/latest_version/3.AI/Tensorflow/Data/simulation/allmix/id1065_na22.Singles_10cm.npy")
        
dtt = SinglesData(dt, dtype_list, add_dt)
print("max source number : ", max(dtt.sourceID))

#%% sourceID 별 index 정리 ---------------------------------------------------------

# gate 기준 source name으로 filter할 source를 지정해야함.
src_name = list(set(dtt.sourceID))
filter_source_idx = {}
for srcnum in src_name:
    filter_source_idx[str(srcnum)] = np.where(dtt.sourceID==srcnum)[0].astype(int)

# filteredData에 초기화용 argument로 target_idx를 넣어줘야함.
dtt_fil = filteredData(filter_source_idx, dtt)
# %%
dtt_fil.sortall([10])
dtt_fil.show()

# %%
print("H")
# %%







#%% bin data 뜯어보기 ------------------------------------------------------------------
import lzma
import random

source = "Ba133"
num = random.choice(range(4,629))

with lzma.open(f'/tf/latest_version/3.AI/Tensorflow/Data/single/{source}/{source}_{num}.xz', 'rb') as f:
    temp = pickle.load(f)
# %%












# %% ================================================================================
# 24.08.05.
# last modified : Dong Hui Seo
# csv는 modi_hist_extract 가서 확인.


#%% xz file extraction - 2023

import lzma

import pickle
import numpy as np
import matplotlib.pyplot as plt

#foldername = "230000_unknown"
#filename = "3.5t_8mm_ba133_1M_1.bin"

foldername = "230000_forTensorflow_xz"
filename = "Ba133/Ba133_6.xz"

dtpath = f"../../Data/{foldername}/{filename}"

num = np.random.randint(1, 400, size=1)
with lzma.open(dtpath, 'rb') as f:
    temp = pickle.load(f)
    
plt.plot(temp)



#%% csv file - 2024

