#%% Load data
import numpy as np

path1 = "./1.preprocess_data/240923_10to20_3source_10000_xzfile_all.npz"
data1 = np.load(path1)

path2 = "./1.preprocess_data/241003_240603_10to20sec_8source_10000_fullbackground_all.npz"
data2 = np.load(path2)

path3 = "./1.preprocess_data/241004_240627_multiple10to20sec_8source_10000_fullbackground_all.npz"
data3 = np.load(path3)

#%% merge and shuffle data

# 8000 | 1000 | 1000
#             | 3000

# => 3000 ++ 2000 | 1000 ++ 500 | 1000 ++ 500
x_train = np.concatenate((data1["x_train"], data2["x_train"], data3["x_train"]), axis=0)
x_val = np.concatenate((data1["x_val"], data2["x_val"], data3["x_val"]), axis=0)
x_test = np.concatenate((data1["x_test"], data2["x_test"], data3["x_test"]), axis=0)

y_train = np.concatenate((data1["y_train"], data2["y_train"], data3["y_train"]), axis=0)
y_val = np.concatenate((data1["y_val"], data2["y_val"], data3["y_val"]), axis=0)
y_test = np.concatenate((data1["y_test"], data2["y_test"], data3["y_test"]), axis=0)

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
filename = "241004_10to20sec_3series_merged_30000"

np.savez(f"./1.preprocess_data/{filename}_all.npz", 
         x_train=x_train,
         y_train=y_train,
         x_val=x_val,
         y_val=y_val,
         x_test=x_test,
         y_test=y_test)







# %% 데이터 변형 test
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

dt = np.load("./1.preprocess_data/241004_10to20sec_3series_merged_orispectrum_normed_all.npz")

x_train = dt["x_train"]
x_val = dt["x_val"]
x_test = dt["x_test"]

y_train = dt["y_train"]
y_val = dt["y_val"]
y_test = dt["y_test"]


#%%
plt.plot(x_train[0,:,0])

#%%
test = noisefiltering(x_train[np.newaxis,0,:,0],2,0)
plt.plot(test[0])

#%%
for i in range(len(x_train)):
    if i%1000==0: print(f"data {i} processed.")
    x_train[i,:,0] = noisefiltering(x_train[np.newaxis,0,:,0],2,0)[0]

for i in range(len(x_val)):
    if i%1000==0: print(f"data {i} processed.")
    x_val[i,:,0] = noisefiltering(x_val[np.newaxis,0,:,0],2,0)[0]
    
for i in range(len(x_test)):
    if i%1000==0: print(f"data {i} processed.")
    x_test[i,:,0] = noisefiltering(x_test[np.newaxis,0,:,0],2,0)[0]
    
np.savez(f"./1.preprocess_data/241004_10to20sec_3series_merged_orispectrum_normed_noisefiltered.npz", 
         x_train=x_train,
         y_train=y_train,
         x_val=x_val,
         y_val=y_val,
         x_test=x_test,
         y_test=y_test)
# %%
