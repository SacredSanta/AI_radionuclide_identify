#%%

import numpy as np
import sys
import os
import matplotlib.pyplot as plt
sys.path.append('./')

from lib.modi_hist_extract import modi_hist_extract
from lib.modwt_pkg import *

#%%

data_folder = "./1.preprocess_data/"
data_name = "240923_10to20_3source_10000_xzfile_all.npz"

data = np.load(os.path.join(data_folder, data_name))

data

#%%
data["x_train"][1]

#%%
data2_folder = "./0.dataset/"
data2_name = "240923_10to20_3source_10000_xzfile.npz"

data2 = np.load(os.path.join(data2_folder, data2_name))

data2
#%%
x = data2["x_ori"][7]
x = noisefiltering(x,0,0)

plt.plot(x[0])



#%%
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

#%%
def normalize(array):
    return (array-min(array)) / (array.max() - array.min())
# %%
dt = np.load("./0.dataset/240924_multiple240627_10to20sec_8source_10000.npz")

ut_data_name = "../../Data/integrated_modified/240627_t700/background/240627_t700.csv"
ut_dt = modi_hist_extract(ut_data_name)
ut = ut_dt.hist
ut = ut[np.newaxis,:]  # (1,1000)
#%% dt
x_ori = dt["x_ori"]
y = dt["y"]
x_back = dt["x_back"]
x_cutted = dt["x_cutted"]
#%%
dtnum = 100
# %%
xt = x_ori[dtnum,0,:]
l1norm_xt = x_ori[dtnum,0,:]/sum(x_ori[dtnum,0,:])
norm_xt = normalize(x_ori[dtnum,0,:])
plt.plot(xt)
plt.xlabel("Energy bin(keV)")
plt.ylabel("Counts")
plt.title("Signal Spectrum")
# %%
plt.plot(x_back[dtnum,0,:])



# %% ut
l1norm_ut = ut[0]/sum(ut[0])
norm_ut = normalize(ut[0])
#plt.plot(ut[0]/sum(ut[0]))
plt.plot(ut[0], color='orange')
plt.xlabel("Energy bin(keV)")
plt.ylabel("Counts")
plt.title("Background Spectrum")

#%% 같이
plt.plot(l1norm_xt, label='signal')
plt.plot(l1norm_ut, label='background', alpha=0.7)
plt.legend()
plt.xlabel("Energy bin(keV)")
plt.ylabel("Counts (Normalized)")
plt.title("Gamma Energy Spectrum")


#%% poisson
result = my_pos_dev(xt=xt, ut_hat=l1norm_ut)
#%%
plt.plot(result, color='green')
plt.xlabel("Energy bin(keV)")
plt.ylabel("Poisson Unit Deviance")
plt.title("Poisson Deviance Spectrum")
# %%
result_n = noisefiltering(result[np.newaxis,:],1,0)
plt.plot(result_n[0], color='green')
plt.xlabel("Energy bin(keV)")
plt.ylabel("Poisson Unit Deviance")
plt.title("Poisson Deviance Spectrum (Filtered)")



























#%% Model Result Plot
#       1 1 1          0.8 1 1       0.6 1 1      0.4 1 1     0.2 1 1     0.1 1 1
md1_x = [1_034_890_440, 746_990_016, 395_952_518, 177_154_030, 58_899_236, 23_849_950]
md1_y = [90.23,         96.33,       97.46,       96.56,       96.26,      95.23]
plt.plot(md1_x, md1_y, label='depth=1.0', marker='x', alpha=0.7)

#        1 0.8 1       0.8 0.8 1     0.6 0.8 1    0.4 0.8 1    0.2 0.8 1  0.1 0.8 1
md08_x = [746_990_016,  473_607_212, 289_629_926, 131_998_844, 45_703_738, 19_618_818]
md08_y = [96.34,        97.47,        97.47,       97.6,        95.73,      94.37]
plt.plot(md08_x, md08_y, label='depth=0.8', marker='o', alpha=0.7)

#
md06_x = []
md06_y = []
plt.plot(md06_x, md06_y, label='depth=0.6', marker='*', alpha=0.7)

plt.title("FLOPs vs Accuracy for blocks [6,12,24,16]")
plt.xticks([i*15_000_000 for i in range(0,80,5)],
                [i for i in range(0,80,5)])
plt.yticks([i+85 for i in range(0,20,5)])
plt.ylabel("Top-1 Accuracy")
plt.xlabel("FLOPs (*15_000_000)")
plt.legend()
# %%
'''
o: 원형
*: 별
s: 사각형(square)
D: 다이아몬드
x: x 모양
+: 더하기 기호
^: 위쪽 삼각형
v: 아래쪽 삼각형
<: 왼쪽 삼각형
>: 오른쪽 삼각형
'''