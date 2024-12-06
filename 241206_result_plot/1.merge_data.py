#%% Load data
import numpy as np

dt1 = "241129_dataset1_set5000_min5014_max5999_xzfile_combi1_norm"
path1 = f"./1.preprocess_data/241129/norm/{dt1}.npz"
data1 = np.load(path1)

dt2 = "241129_dataset2_set10000_min5000_max6000_combi1_norm"
path2 = f"./1.preprocess_data/241129/norm/{dt2}.npz"
data2 = np.load(path2)

# dt3 = "241121_multiple240627_dataset3_set5000_min500_max1000_norm"
# path3 = f"./1.preprocess_data/241121/norm/{dt3}.npz"
# data3 = np.load(path3)

#%% merge and shuffle data

# 8000 | 1000 | 1000
#             | 3000

# => 3000 ++ 2000 | 1000 ++ 500 | 1000 ++ 500
x_train = np.concatenate((data1["x_train"], data2["x_train"]))#, data3["x_train"]), axis=0)
x_val = np.concatenate((data1["x_val"], data2["x_val"]))#, data3["x_val"]), axis=0)
x_test = np.concatenate((data1["x_test"], data2["x_test"]))#, data3["x_test"]), axis=0)

y_train = np.concatenate((data1["y_train"], data2["y_train"]))#, data3["y_train"]), axis=0)
y_val = np.concatenate((data1["y_val"], data2["y_val"]))#, data3["y_val"]), axis=0)
y_test = np.concatenate((data1["y_test"], data2["y_test"]))#, data3["y_test"]), axis=0)

print("x_train : ", x_train.shape)
print("x_val : ", x_val.shape)
print("x_test : ", x_test.shape)
print("y_train : ", y_train.shape)
print("y_val : ", y_val.shape)
print("y_test : ", y_test.shape)


#* 순서 섞기 x ~ y 간 관계 틀어짐. 모든 걸 같이 섞는 방법을 찾아야함.
#np.random.shuffle(total_x)
#np.random.shuffle(total_y)


#%% 
filename = "241129_set15000_min5000_max6000_norm"

np.savez(f"./1.preprocess_data/{filename}_all.npz", 
         x_train=x_train,
         y_train=y_train,
         x_val=x_val,
         y_val=y_val,
         x_test=x_test,
         y_test=y_test)






# %% ===================================================================
# noise filtering || normalize
# %% 데이터 변형 test -------------------------------------------------------------
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

def normalize(array):
    return (array-min(array)) / (array.max() - array.min())

dttname = '241129_dataset2_set10000_min5000_max6000_combi1.npz_fullbackground'

dt = np.load(f"./1.preprocess_data/{dttname}.npz")

x_train = dt["x_train"]
x_val = dt["x_val"]
x_test = dt["x_test"]

y_train = dt["y_train"]
y_val = dt["y_val"]
y_test = dt["y_test"]

print("x_train : ", x_train.shape)
print("x_val : ", x_val.shape)
print("x_test : ", x_test.shape)
print("y_train : ", y_train.shape)
print("y_val : ", y_val.shape)
print("y_test : ", y_test.shape)
#%%
plt.plot(x_train[0,:,0])

#%%
test = normalize(x_train[0,:,0])
plt.plot(test)

#%% normalize 진행
for i in range(len(x_train)):
    if i%1000==0: print(f"data {i} processed.")
    x_train[i,:,0] = normalize(x_train[i,:,0])

for i in range(len(x_val)):
    if i%1000==0: print(f"data {i} processed.")
    x_val[i,:,0] = normalize(x_val[i,:,0])
    
for i in range(len(x_test)):
    if i%1000==0: print(f"data {i} processed.")
    x_test[i,:,0] = normalize(x_test[i,:,0])



#%% noise filtering 진행
for i in range(len(x_train)):
    if i%1000==0: print(f"data {i} processed.")
    x_train[i,:,0] = noisefiltering(x_train[np.newaxis,0,:,0],12,0)[0]

for i in range(len(x_val)):
    if i%1000==0: print(f"data {i} processed.")
    x_val[i,:,0] = noisefiltering(x_val[np.newaxis,0,:,0],12,0)[0]
    
for i in range(len(x_test)):
    if i%1000==0: print(f"data {i} processed.")
    x_test[i,:,0] = noisefiltering(x_test[np.newaxis,0,:,0],12,0)[0]


#%%    
np.savez(f"./1.preprocess_data/{dttname[:-19]}_norm.npz", 
         x_train=x_train,
         y_train=y_train,
         x_val=x_val,
         y_val=y_val,
         x_test=x_test,
         y_test=y_test)
# %%




















#%% =============================================================
# change spectrum
# ===============================================================
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

# dataset 1
#dt_name = "241121_dataset1_set5000_min5001_max6000_xzfile"
#pre_data = np.load(f"./1.preprocess_data/241121/ori/{dt_name}.npz_fullbackground.npz")
#cha_data = np.load(f"./0.dataset/{dt_name}.npz")

# dataset 2
dt_name = "241121_multiple240627_dataset3_set5000_min5000_max6000"
pre_data = np.load(f"./1.preprocess_data/241121/ori/{dt_name}.npz_fullbackground.npz")
cha_data = np.load(f"./0.dataset/{dt_name}.npz")


x_train = pre_data["x_train"]
y_train = pre_data["y_train"]
x_val = pre_data["x_val"]
y_val = pre_data["y_val"]
x_test = pre_data["x_test"]
y_test = pre_data["y_test"]

x_train[:,:,0] = cha_data["x_ori"][:3000,0,:]
x_val[:,:,0] = cha_data["x_ori"][3000:4000,0,:]
x_test[:,:,0] = cha_data["x_ori"][4000:,0,:]

#np.savez("./1.preprocess_data/241128/")
np.savez(f"./1.preprocess_data/241128/{dt_name}_rawori.npz", 
         x_train=x_train,
         y_train=y_train,
         x_val=x_val,
         y_val=y_val,
         x_test=x_test,
         y_test=y_test)
































#%% =============================================================================
# Test Zone
# ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

dt = np.load("./1.preprocess_data/241121/ori/241121_dataset1_set5000_min1040_max1999_xzfile.npz_fullbackground.npz")


