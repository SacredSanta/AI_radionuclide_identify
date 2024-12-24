#%% ===================================================================================
# noise filtering 관련 plot
# =====================================================================================


import numpy as np
import sys
import os
import matplotlib.pyplot as plt
sys.path.append('./')

import tensorflow as tf
from lib.modi_hist_extract import modi_hist_extract
from lib.modwt_pkg import *

#%%
spec = np.load("./0.dataset/241211_dataset2_set5000_min1000_max2000_combi3_src5.npz")
x_train = spec["x_ori"]

test = x_train[156]
x = noisefiltering2(test,10,0)

plt.plot(x[0])


















#%% ===================================================================================
# 
# =====================================================================================


import numpy as np
import sys
import os
import matplotlib.pyplot as plt
sys.path.append('./')

from lib.modi_hist_extract import modi_hist_extract
from lib.modwt_pkg import *
#%%
#dt = np.load("./0.dataset/241113_250to300sec_8source_5000.npz",)
#dt = np.load("./0.dataset/241113_multiple240627_250to300sec_8source_5000.npz",)


ddt = dt["x_ori"][:,0,:]
ddt_sum = np.sum(ddt,axis=1)
#%%
idx = 4225
dddt = ddt[idx]
plt.plot(dddt)
print(ddt_sum[idx])
print(dt['y'][idx])
#plt.title("Signal Spectrum")
#plt.xlabel("Energy bin(keV)")
#plt.ylabel("Counts")
#%%
count_hist = np.zeros((56))
for i in range(len(ddt_sum)):
    count_hist[int(ddt_sum[i]/100)] += 1
#%%
plt.bar(range(len(count_hist)), count_hist)


#%%
import tensorflow as tf

model_dir = "./2.model/241113_densenet1d_combi/241113_count5000down_3series_merged_12000_all_orispec_normed/block2,4,8,4/1,1,1_w/1,1,1/241113_count5000down_3series_merged_12000_all_noisefiltered_orispec_normed_241107_densenet1d_w1_d1_r1.keras"
Model = tf.keras.models.load_model(model_dir)
#%%
#pred_data = np.load("./1.preprocess_data/241113_multiple240627_250to300sec_8source_5000_fullbackground.npz")
pred_data = np.load("./1.preprocess_data/241113_250to300sec_8source_5000_fullbackground.npz")
#%%
plt.plot(pred_data['x_test'][225,:,0])
print(pred_data['y_test'][225])

#%%
th_count_idx = np.where(ddt_sum>1400)[0]
t_th = np.where(th_count_idx > 4000)[0]
test_idx = th_count_idx[t_th]
#%%
test_idx = test_idx - 4000
#%%
pred = 0
for i in test_idx:
    pp = Model.predict(pred_data['x_test'][tf.newaxis,i,:,:])
    if sum(sum(pp==pred_data["y_test"][i,:,:][0])) == 8:
        pred += 1
print(pred)
#%%
pp = Model.predict(pred_data['x_test'][tf.newaxis,1,:,:])
pp = pp>0.5
pp == pred_data["y_test"][1,:,:]

























#%%
# ========================================================================================================================================================================================
# Poisson Deviance 관련 plotting
# ========================================================================================================================================================================================


#%%

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
from poisson_deviance import my_pos_dev, fixed_pos_dev

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
signaled_ut = l1norm_xt*norm_ut
#plt.plot(ut[0]/sum(ut[0]))
plt.plot(signaled_ut, color='orange')
plt.xlabel("Energy bin(keV)")
plt.ylabel("Counts")
plt.title("Background Spectrum")

#%%
deviance_a = xt*np.log(xt / sum(abs(xt))) * l1norm_ut
#plt.plot(deviance_a)

deviance_b = xt - (sum(abs(xt))*l1norm_ut)
deviance_c = sum(abs(xt))*l1norm_ut
deviance_d = xt/(sum(abs(xt))) * l1norm_ut

dev = fixed_pos_dev(xt, l1norm_ut)
plt.plot(dev)

#%% 같이
plt.plot(l1norm_xt, label='signal', alpha=0.8)
plt.plot(, label='background', alpha=0.8)
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
result_n = noisefiltering(dev[np.newaxis,:],0,0)
plt.plot(result_n[0], color='green')
plt.xlabel("Energy bin(keV)")
plt.ylabel("Poisson Unit Deviance")
plt.title("Poisson Deviance Spectrum (Filtered)")
#%%
plt.plot(noisefiltering(xt[np.newaxis,:],1,0)[0])
plt.legend()
plt.xlabel("Energy bin(keV)")
plt.ylabel("Counts (Normalized)")
plt.title("Filtered Original Spectrum")
#%%
nf_xt = noisefiltering(xt[np.newaxis,:],0,0)
plt.plot(noisefiltering2(derivative_signal(nf_xt),0,0)[0], color='red')
plt.legend()
plt.xlabel("Energy bin(keV)")
plt.ylabel("Value (Normalized)")
plt.title("Filtered 1st derivative of Original Spectrum")






#%%
all_dt = np.load("./1.preprocess_data/241004_10to20sec_3series_merged_orispectrum_normed_noisefiltered.npz")
x_data = all_dt["x_train"][0]
#%%
plt.plot(x_data[:,0], label='original spectrum')
plt.plot(x_data[:,1], label='1st derivatives')
plt.plot(x_data[:,2], label='poisson deviance spectrum')
plt.legend()
plt.xlabel("Energy bin(keV)")
plt.ylabel("Poisson Unit Deviance")
plt.title("Inputs")























#%% Model Result Plot
#       1 1 1          0.8 1 1       0.6 1 1      0.4 1 1     0.2 1 1      0.1 1 1      0.05 1 1
md1_x = [1_034_890_440, 746_990_016, 395_952_518, 177_154_030, 58_899_236, 23_849_950,  9_424_586]
md1_y = [90.23,         96.33,       97.46,       96.56,       96.26,      95.23,       96.8]
plt.plot(md1_x, md1_y, label='depth=1.0', marker='x', alpha=0.7)

#        1 0.8 1       0.8 0.8 1     0.6 0.8 1    0.4 0.8 1    0.2 0.8 1   0.1 0.8 1    0.05 0.8 1
md08_x = [746_990_016,  473_607_212, 289_629_926, 131_998_844, 45_703_738, 19_618_818,  8_494_280]
md08_y = [96.33,        97.47,        97.47,      97.6,        95.73,      94.37,       97.63]
plt.plot(md08_x, md08_y, label='depth=0.8', marker='o', alpha=0.7)

#        1 0.6 1       0.8 0.6 1     0.6 0.6 1    0.4 0.6 1    0.2 0.6 1   0.1 0.6 1    0.05 0.6 1   
md06_x = [485_864_280, 311_493_370,  192_860_998, 90_423_040,  33_447_484, 15_631_078,  7_705_834]
md06_y = [96.97,       96.63,        96.8,        96.57,       85.73,      89.5,        96.73]
plt.plot(md06_x, md06_y, label='depth=0.6', marker='o', alpha=0.7)

#        1 0.4 1        0.8 0.4 1     0.6 0.4 1    0.4 0.4 1    0.2 0.4 1   0.1 0.4 1    0.05 0.4 1   
md04_x = [257_346_240,  167_507_312,  106_406_276, 52_720_532,  21_742_342, 11_551_440,  6_681_620]
md04_y = [97.67,        98.3,         97.2,        87.03,       95.6,       95.07,       95.73]
plt.plot(md04_x, md04_y, label='depth=0.4', marker='o', alpha=0.7)

#        1 0.2 1        0.8 0.2 1     0.6 0.2 1    0.4 0.2 1   0.2 0.2 1   0.1 0.2 1    0.05 0.2 1   
md02_x = [114_929_752,  77_475_370,  51_340_486,   27_813_472,   13_616_332, 8_583_910,  5_994_202]
md02_y = [97.33,        97.67,       97.63,        97.6,       94.4,       96.2,       95.20]
plt.plot(md02_x, md02_y, label='depth=0.2', marker='o', alpha=0.7)

#        1 0.1 1        0.8 0.1 1     0.6 0.1 1    0.4 0.1 1   0.2 0.1 1   0.1 0.1 1    0.05 0.1 1   
md01_x = [55_263_872,   38_483_262,   26_867_412,  16_158_716, 9_363_790,  6_832_704,   5_440_674]
md01_y = [93.33,        96.97,        96.0,        96.87,      95.57,      96.2,        93.93]
plt.plot(md01_x, md01_y, label='depth=0.1', marker='o', alpha=0.7)

#        1 0.05 1        0.8 0.05 1     0.6 0.05 1    0.4 0.05 1   0.2 0.05 1   0.1 0.05 1    0.05 0.05 1   
md01_x = [37_119_536,    26_739_704,    19_363_994,   12_557_780,  8_058_418,   6_270_582,    5_310_680]
md01_y = [97.07,         97.13,         95.76,        94.27,       94.53,       93.1,         93.23]
plt.plot(md01_x, md01_y, label='depth=0.05', marker='o', alpha=0.7)

plt.title("FLOPs vs Accuracy for blocks [6,12,24,16]")
plt.xticks([i*15_000_000 for i in range(0,40,5)],
                [i for i in range(0,40,5)])
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






#%%
# ========================================================================================================================================================================================
# arale handheld energy resoltuion 구하기
# ========================================================================================================================================================================================

import sys
import os
sys.path.append(os.path.join(os.getcwd(), "lib"))
from modi_hist_extract import modi_hist_extract
import itertools
import random
import numpy as np
from joblib import Parallel, delayed, cpu_count
#%%
csv_file = "../../Data/240927_arale/240927_031156_miplindoor_na_375279_5m_thres1000_caliOn.csv"
fil_dt = modi_hist_extract(csv_file, cali='on', isbin='no')
#%%
fil_dt.show()
#%%
pdix = fil_dt.find_peak([490,550])
fil_dt.fix_data(511)
fil_dt.show()
#%%
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 가우시안 함수 정의
def gaussian(x, a, b, c):
    return a * np.exp(-((x - b) ** 2) / (2 * c ** 2))
x_data = np.arange(0,1000)
y_data = fil_dt.hist
# 특정 위치(중심값 2)를 기준으로 데이터 선택
center = 511
window = 40  # 범위 (2 ± 2)
mask = (x_data >= center - window) & (x_data <= center + window)
x_subset = x_data[mask]
y_subset = y_data[mask]

# 피팅 수행
popt, _ = curve_fit(gaussian, x_subset, y_subset, p0=[4, center, 1])
a_fit, b_fit, c_fit = popt
print(f"Fitted parameters: a={a_fit}, b={b_fit}, c={c_fit}")

# 피팅 결과 시각화
plt.plot(x_data, y_data, label="Original Data")#), #s=10, color="blue")
plt.plot(x_subset, y_subset, label="Peak window")#, s=20, color="orange")
plt.plot(x_data, gaussian(x_data, *popt), label="Gaussian Fit", color="red")
plt.axvline(x=center, color='green', linestyle='--', label="Center")
plt.legend()
plt.show()
#%%
peak_ = max(gaussian(x_data, *popt))
peak_
#%%
idx = np.where(abs(gaussian(x_data, *popt)-(peak_/2)) <= 10)
idx

#%%
(543-491)/1000 * 100


#%%
plt.scatter(np.arange(0,1000), fil_dt.hist, label='Data', color='gray')
plt.xlabel('Energy (keV)')
plt.ylabel('Count')
plt.title('Na-22')
plt.show()







#%% xz -------------------------------------------------------------------------------------
import lzma
import pickle
import numpy as np
import matplotlib.pyplot as plt

dt_path = f"../../Data/230000_forTensorflow_xz/"
xz_ = np.array(['Ba133', 'Cs137', 'Na22', 'Background'])
idx = 2
RA = "na"

final_hist = np.zeros([1, 1000])
for time in range(0,930):
    xzfile_path = os.path.join(dt_path, RA, f'{xz_[idx]}_{time}.xz')    
    with lzma.open(xzfile_path, 'rb') as f:
        temp = pickle.load(f)
        final_hist[0, :] += temp.reshape(1,1000)[0,:]
#%%
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "lib"))
from modwt_pkg import *

#%%
plt.plot(final_hist[0], label='Data', color='gray')
plt.xlabel('Energy (keV)')
plt.ylabel('Count')
plt.title('Na-22')
plt.axhline(y=1770)
plt.axvline(x=479)
plt.axvline(x=568)
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 가우시안 함수 정의
def gaussian(x, a, b, c):
    return a * np.exp(-((x - b) ** 2) / (2 * c ** 2))
x_data = np.arange(0,1000)
y_data = final_hist[0]

# 특정 위치(중심값 2)를 기준으로 데이터 선택
center = 511
window = 40  # 범위 (2 ± 2)
mask = (x_data >= center - window) & (x_data <= center + window)
x_subset = x_data[mask]
y_subset = y_data[mask]

# 피팅 수행
popt, _ = curve_fit(gaussian, x_subset, y_subset, p0=[4, center, 1])
a_fit, b_fit, c_fit = popt
print(f"Fitted parameters: a={a_fit}, b={b_fit}, c={c_fit}")

# 피팅 결과 시각화
plt.plot(x_data, y_data, label="Original Data")#), #s=10, color="blue")
plt.plot(x_subset, y_subset, label="Peak window")#, s=20, color="orange")
plt.plot(x_data, gaussian(x_data, *popt), label="Gaussian Fit", color="red")
plt.axvline(x=center, color='green', linestyle='--', label="Center")
plt.legend()
plt.show()

#%%
from scipy.ndimage import median_filter
final_hist_fil = median_filter(final_hist,size=2)
plt.plot(final_hist_fil[0])
#%%
final_hist_fil[0,:50] = 0
plt.plot(final_hist_fil[0])
#%%
peak_ = max(gaussian(x_data, *popt)[300:700])
peak_
#%%
idx = np.where(abs(gaussian(x_data, *popt)-(peak_/2)) <= 20)
idx
#%%
(568-479)/1000*100















#%% plot AI data------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
filename = '241113/241113_count5000down_3series_merged_12000_all_orispec_normed'
model_data = np.load(f"./1.preprocess_data/{filename}.npz")

x_train = model_data["x_train"]#[:,:,:2]
y_train = model_data["y_train"]

x_val = model_data["x_val"]#[:,:,:2]
y_val = model_data["y_val"]

x_test = model_data["x_test"]
y_test = model_data["y_test"]

x_train = x_train[:,:,:]
y_train = y_train[:,0,:]
x_val = x_val[:,:,:]
y_val = y_val[:,0,:]
x_test = x_test[:,:,:]
y_test = y_test[:,0,:]

#%%
def normalize(array):
    return (array-min(array)) / (array.max() - array.min())
a = x_train[0,:,:]
a[:,0] = normalize(a[:,0])
#%%
plt.plot(a)



#%%
20*(0.6**(3/2))



















#%%
# ========================================================================================================================================================================================
# 평가 지표
# ========================================================================================================================================================================================
#%% 평가지표 뽑을 것들
import pandas as pd
import copy

# dataframe 기초 양식 생성
columns = [str(i*2000+1000) for i in range(0,8)]
init_dict = {col: {"Detection Rate":None, "Precision":None, "Sensitivity":None, "Specificity":None, "F1 Score":None}
             for col in columns}

# source 별로 하나씩 데이터 생성
sources = ["ba", "cs", "na", "co57", "th", "ra", "am"]
final_dt = pd.DataFrame(init_dict)

# 모든 dataframe 하나의 dictionary 로 묶기
overall_pd = {}

for source in sources:
    overall_pd[f"{source}"] = copy.deepcopy(final_dt)

eval_cols = ["Detection Rate", "Sensitivity", "Specificity", "Precision", "F1 Score"]

overall_pd
#%% 241202 귀찮게 하나하나 confusion matrix 까야할 때.
import pandas as pd
import os 

# 불러올 accuracy 결과 폴더 최상단.
base = f"./2.model/241215_effinet_combi/"
dt_folders = os.listdir(base)
dt_folders.sort()

try:
    dt_folders.remove("pre_data")
except:
    dt_folders.remove("wrong_data")
dt_folders
#%%
new_dt_folders = []

for i in dt_folders:
    if '1129' not in i:
        print(i)
        new_dt_folders.append(i)

new_dt_folders
#%%

#filename_isold = 1

# 4중첩 for문이라니....
# 각 폴더별 반복, 241121 결과만.
for dt_folder in new_dt_folders:
    inner_folder = os.listdir(os.path.join(base, dt_folder, "1.0,1.0,1.0/predict/"))
    for filename in inner_folder:
        # 오류로 학습안되어있는 폴더 존재하기 때문에(txt 파일이 있어야 학습이 정상적으로 이루어진 것)
        if not filename.endswith(".csv"):
            continue
        
        df = pd.read_csv(os.path.join(base, dt_folder, "1.0,1.0,1.0/predict/", filename))
        print(filename[:40], "===============================")
        print("precision : ", df.iloc[7].mean())
        print("Sensitivity : ", df.iloc[5].mean())
        print("specificity : ", df.iloc[6].mean())
        print("f1 score : ", df.iloc[8].mean())
        
        fmci = dt_folder.find("min") # file_min_count_idx
        
        ''' filename 에서 찾으려고 했는데 그냥 디렉토리 name 에서 찾아야할듯
        if filename_isold:
            if filename[fmci+3+3] == '_': # min 500 부분 글자 개수 때문에 예외처리
                file_min_count = filename[fmci+3:fmci+3+3]
            else:
                file_min_count = filename[fmci+3:fmci+3+4]
        else:
            file_min_count = str(round(int(filename[fmci+3:fmci+3+5]),-3))
        '''
        file_min_count = str(round(int(dt_folder[fmci+3:fmci+3+5]),-2)) # min 시작 index 3개 이후부터 숫자, 총 5글자가 숫자
        
        # df 의 평가지표 순서는 차례대로 Sen, Spe, Pre, F1 임.    
        for source in sources:
            for eval_idx in range(5): # eval_cols 개수만큼 반복
                overall_pd[source][file_min_count][eval_cols[eval_idx]] = round(df[source][eval_idx+4], 2) # eval_idx 부터 +4 가 accuracy ~ 지표들 (df 상 row)

overall_pd

#%% 241202 귀찮게 하나하나 confusion matrix 까야할 때. - EfficientNet

#filename_isold = 1

# 4중첩 for문이라니....
# 각 폴더별 반복, 241121 결과만.
for dt_folder in new_dt_folders:
    inner_folder = os.listdir(os.path.join(base, dt_folder, "blocks_arg6/1.0,1.0,1.0/predict/"))
    for filename in inner_folder:
        # 오류로 학습안되어있는 폴더 존재하기 때문에(txt 파일이 있어야 학습이 정상적으로 이루어진 것)
        if not filename.endswith(".csv"):
            continue
        
        df = pd.read_csv(os.path.join(base, dt_folder, "blocks_arg6/1.0,1.0,1.0/predict/", filename))
        print(filename[:40], "===============================")
        print("precision : ", df.iloc[7].mean())
        print("Sensitivity : ", df.iloc[5].mean())
        print("specificity : ", df.iloc[6].mean())
        print("f1 score : ", df.iloc[8].mean())
        
        fmci = dt_folder.find("min") # file_min_count_idx
        
        ''' filename 에서 찾으려고 했는데 그냥 디렉토리 name 에서 찾아야할듯
        if filename_isold:
            if filename[fmci+3+3] == '_': # min 500 부분 글자 개수 때문에 예외처리
                file_min_count = filename[fmci+3:fmci+3+3]
            else:
                file_min_count = filename[fmci+3:fmci+3+4]
        else:
            file_min_count = str(round(int(filename[fmci+3:fmci+3+5]),-3))
        '''
        file_min_count = str(round(int(dt_folder[fmci+3:fmci+3+5]),-2)) # min 시작 index 3개 이후부터 숫자, 총 5글자가 숫자
        
        # df 의 평가지표 순서는 차례대로 Sen, Spe, Pre, F1 임.    
        for source in sources:
            for eval_idx in range(5): # eval_cols 개수만큼 반복
                overall_pd[source][file_min_count][eval_cols[eval_idx]] = round(df[source][eval_idx+4], 2) # eval_idx 부터 +4 가 accuracy ~ 지표들 (df 상 row)

overall_pd

# =======================================================================
#%% 각 선원별 pre, sen, spe, f1 --------------------------------------
import matplotlib.pyplot as plt

columns = columns[:-1]
sources = ['ba', 'cs', 'na' ,'th', 'am']

plt.figure(figsize=(10,6))
for source in sources:
    plt.plot(overall_pd[source].loc["Detection Rate"].to_numpy()[:-1], label=f"{source}")
plt.xticks([i for i in range(len(columns))], columns, fontsize=10)
plt.xlabel("Minimum Counts of Data", fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel("", fontsize=15)
plt.ylim(0.80, 1.02)
plt.title("Detection Rate", fontsize=18)
plt.legend(fontsize=15, loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()

#%% ----------------------------------------
plt.figure(figsize=(10,6))
for source in sources:
    plt.plot(overall_pd[source].loc["Precision"].to_numpy(), label=f"{source}")
plt.xticks([i for i in range(len(columns))], columns, fontsize=10)
plt.xlabel("Minimum Counts of Data", fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel("[TP / (TP+FP)]", fontsize=15)
plt.ylim(0.50, 1.05)
plt.title("Precision", fontsize=18)
plt.legend(fontsize=15, loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()

#%% --------------------------------------
plt.figure(figsize=(10,6))
for source in sources:
    plt.plot(overall_pd[source].loc["Sensitivity"].to_numpy(), label=f"{source}")
plt.xticks([i for i in range(len(columns))], columns, fontsize=10)
plt.xlabel("Minimum Counts of Data", fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel("[TP / (TP+FN)]", fontsize=15)
plt.ylim(0.50, 1.05)
plt.title("Sensitivity", fontsize=18) #recall
plt.legend(fontsize=15, loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()

#%% --------------------------------------
plt.figure(figsize=(10,6))
for source in sources:
    plt.plot(overall_pd[source].loc["Specificity"].to_numpy(), label=f"{source}")
plt.xticks([i for i in range(len(columns))], columns, fontsize=10)
plt.xlabel("Minimum Counts of Data", fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel("[TN / (FP+TN)]", fontsize=15)
plt.ylim(0.50, 1.05)
plt.title("Specificity", fontsize=18)
plt.legend(fontsize=15, loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()

#%% --------------------------------------
plt.figure(figsize=(10,6))
for source in sources:
    plt.plot(overall_pd[source].loc["F1 Score"].to_numpy(), label=f"{source}")
plt.xticks([i for i in range(len(columns))], columns, fontsize=10)
plt.xlabel("Minimum Counts of Data", fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel("2*(Pre*sen)/(Pre+Sen)", fontsize=15)
plt.ylim(0.50, 1.05)
plt.title("F1 Score", fontsize=18)
plt.legend(fontsize=15, loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()



#%%












#%% ----------------------------------------------------------------------------------------------------------------------------
# try 0 ~ 9 반복학습에 대한 결과
# ------------------------------------------------------------------------------------------------------------------------------
#%% 평가지표 뽑을 것들
import pandas as pd
import copy

# dataframe 기초 양식 생성
columns = [f"try{i}" for i in range(0,10)]
init_dict = {col: {"Detection Rate":None, "Precision":None, "Sensitivity":None, "Specificity":None, "F1 Score":None}
             for col in columns}

# source 별로 하나씩 데이터 생성
sources = ["ba", "cs", "na", "co57", "th", "ra", "am"]
final_dt = pd.DataFrame(init_dict)

# 모든 dataframe 하나의 dictionary 로 묶기
overall_pd = {}

for source in sources:
    overall_pd[f"{source}"] = copy.deepcopy(final_dt)

eval_cols = ["Detection Rate", "Sensitivity", "Specificity", "Precision", "F1 Score"]

overall_pd
#%% 241202 귀찮게 하나하나 confusion matrix 까야할 때. - DenseNet
import pandas as pd
import os 

# 불러올 accuracy 결과 폴더 최상단.
base = f"./2.model/241121data_densenet1d_combi/final_seedfix812"
dt_folders = os.listdir(base)
dt_folders.sort()
#%%
new_dt_folders = []

for i in dt_folders:
    if '1129' not in i:
        print(i)
        new_dt_folders.append(i)

new_dt_folders
#%%

filename_isold = 1

# 4중첩 for문이라니....
# 각 폴더별 반복, 241121 결과만.
for dt_folder in [new_dt_folders[9]]:
    inner_folder = os.listdir(os.path.join(base, dt_folder, "block2,4,8,4/"))
    inner_folder.remove("1.0,1.0,1.0")
    inner_folder.sort()
    for eachtry in inner_folder:
        eachtry_folder = os.path.join(base, dt_folder, "block2,4,8,4", eachtry, "1.0,1.0,1.0", 'predict')
        for filename in os.listdir(eachtry_folder):
        
            # 오류로 학습안되어있는 폴더 존재하기 때문에(txt 파일이 있어야 학습이 정상적으로 이루어진 것)
            if not filename.endswith(".csv"):
                continue
            
            df = pd.read_csv(os.path.join(eachtry_folder, filename))
            print(filename[:40], "===============================")
            print("precision : ", df.iloc[7].mean())
            print("Sensitivity : ", df.iloc[5].mean())
            print("specificity : ", df.iloc[6].mean())
            print("f1 score : ", df.iloc[8].mean())
            
            fmci = dt_folder.find("min") # file_min_count_idx
            
            ''' filename 에서 찾으려고 했는데 그냥 디렉토리 name 에서 찾아야할듯
            if filename_isold:
                if filename[fmci+3+3] == '_': # min 500 부분 글자 개수 때문에 예외처리
                    file_min_count = filename[fmci+3:fmci+3+3]
                else:
                    file_min_count = filename[fmci+3:fmci+3+4]
            else:
                file_min_count = str(round(int(filename[fmci+3:fmci+3+5]),-3))
            '''
            #file_min_count = str(round(int(dt_folder[fmci+3:fmci+3+5]),-2)) # min 시작 index 3개 이후부터 숫자, 총 5글자가 숫자
            
            # df 의 평가지표 순서는 차례대로 Sen, Spe, Pre, F1 임.    
            for source in sources:
                for eval_idx in range(5): # eval_cols 개수만큼 반복
                    overall_pd[source][eachtry][eval_cols[eval_idx]] = round(df[source][eval_idx+4], 2) # eval_idx 부터 +4 가 accuracy ~ 지표들 (df 상 row)

overall_pd




# =======================================================================
#%% 각 선원별 pre, sen, spe, f1 --------------------------------------
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
for source in sources:
    plt.plot(overall_pd[source].loc["Detection Rate"].to_numpy(), label=f"{source}")
plt.xticks([i for i in range(len(columns))], columns, fontsize=10)
plt.xlabel("Minimum Counts of Data", fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel("", fontsize=15)
plt.ylim(0.70, 1.05)
plt.title("Detection Rate", fontsize=18)
plt.legend(fontsize=15, loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()

#%% ----------------------------------------
plt.figure(figsize=(10,6))
for source in sources:
    plt.plot(overall_pd[source].loc["Precision"].to_numpy(), label=f"{source}")
plt.xticks([i for i in range(len(columns))], columns, fontsize=10)
plt.xlabel("Minimum Counts of Data", fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel("[TP / (TP+FP)]", fontsize=15)
plt.ylim(0.50, 1.05)
plt.title("Precision", fontsize=18)
plt.legend(fontsize=15, loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()

#%% --------------------------------------
plt.figure(figsize=(10,6))
for source in sources:
    plt.plot(overall_pd[source].loc["Sensitivity"].to_numpy(), label=f"{source}")
plt.xticks([i for i in range(len(columns))], columns, fontsize=10)
plt.xlabel("Minimum Counts of Data", fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel("[TP / (TP+FN)]", fontsize=15)
plt.ylim(0.4, 1.05)
plt.title("Sensitivity", fontsize=18) #recall
plt.legend(fontsize=15, loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()

#%% --------------------------------------
plt.figure(figsize=(10,6))
for source in sources:
    plt.plot(overall_pd[source].loc["Specificity"].to_numpy(), label=f"{source}")
plt.xticks([i for i in range(len(columns))], columns, fontsize=10)
plt.xlabel("Minimum Counts of Data", fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel("[TN / (FP+TN)]", fontsize=15)
plt.ylim(0.40, 1.05)
plt.title("Specificity", fontsize=18)
plt.legend(fontsize=15, loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()

#%% --------------------------------------
plt.figure(figsize=(10,6))
for source in sources:
    plt.plot(overall_pd[source].loc["F1 Score"].to_numpy(), label=f"{source}")
plt.xticks([i for i in range(len(columns))], columns, fontsize=10)
plt.xlabel("Minimum Counts of Data", fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel("2*(Pre*sen)/(Pre+Sen)", fontsize=15)
plt.ylim(0.60, 1.05)
plt.title("F1 Score", fontsize=18)
plt.legend(fontsize=15, loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()

















































#%% ====================================================================================================================================================================================
# Accuracy 부분에 관한 비교 - DenseNet
# ============================================================================================================================================================
import pandas as pd
import os 
import matplotlib.pyplot as plt

# 불러올 accuracy 결과 폴더 최상단.
base = f"./2.model/241215data_densenet1d_combi/"
dt_folders = os.listdir(base)
dt_folders.sort()

# 특수 처리
dt_folders.remove('pre_data')
#dt_folders.remove('wrong_data')

new_dt_folders = []

for i in dt_folders:
    if '1129' not in i:
        print(i)
        new_dt_folders.append(i)


new_dt_folders
#%%
results = []

for folder in new_dt_folders:
    inner_folder = os.path.join(base, folder, "1.0,1.0,1.0")
    for filename in os.listdir(inner_folder):
        if filename.endswith('.txt'):
            txt_path = os.path.join(inner_folder, filename)
            with open(txt_path,'r') as fff:
                lines = fff.readlines()
            acc = list(lines[10].split())[3][:-1]
            results.append(round(float(acc), 2))


#%%
xticks_ = [i for i in range(0,8)]
xticks_dt = [1000*i for i in range(1,16,2)]
plt.figure(figsize=(10,3))
plt.plot(results, marker='o', label="DenseNet")
plt.xticks(xticks_, xticks_dt, fontsize=12)
plt.xlabel("Minimum Counts", fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel("Detection Rate(%)", fontsize=15)
plt.ylim(80, 100.1)
plt.title("", fontsize=18)
plt.legend(fontsize=15)
plt.tight_layout()


#%% ------------------------------------------------------------------------------------
# EfficientNet 부분.
import pandas as pd
import os 
import matplotlib.pyplot as plt

# 불러올 accuracy 결과 폴더 최상단.
base2 = f"./2.model/241215_effinet_combi/"
dt_folders2 = os.listdir(base2)
dt_folders2.sort()

# 특수 처리
dt_folders2.remove('pre_data')

new_dt_folders2 = []

for i in dt_folders2:
    if '1129' not in i:
        print(i)
        new_dt_folders2.append(i)
        
#%%
results2 = []

for folder in new_dt_folders2:
    inner_folder = os.path.join(base2, folder, "blocks_arg6", "1.0,1.0,1.0")
    for filename in os.listdir(inner_folder):
        if filename.endswith('.txt'):
            txt_path = os.path.join(inner_folder, filename)
            with open(txt_path,'r') as fff:
                lines = fff.readlines()
            acc = list(lines[13].split())[3][:-1]
            results2.append(round(float(acc), 2))

#%%
xticks_ = [i for i in range(0,8)]
xticks_dt = [1000*i for i in range(1,9)]
plt.figure(figsize=(10,3))
plt.plot(results2, marker='o', label="EfficientNet")
plt.xticks(xticks_, xticks_dt, fontsize=12)
plt.xlabel("Minimum Counts", fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel("Detection Rate(%)", fontsize=15)
plt.ylim(95, 100.1)
plt.title("", fontsize=18)
plt.legend(fontsize=15)
plt.tight_layout()



#%% ------------------------------------------------------------------
# 함꼐 비교.

xticks_ = [i for i in range(0,7)]
xticks_dt = [1000*i for i in range(1,14,2)]
plt.figure(figsize=(10,3))
plt.plot(results[:-1], marker='o', label="DenseNet")
plt.plot(results2[:-1], marker='o', label="EfficientNet")

for i in range(len(results[:-1])):
    plt.annotate(
        f"({results[i]})",  # 표시할 텍스트
        (xticks_[i], results[i]-2),        # 텍스트 위치
        textcoords="offset points",  # 텍스트 위치 기준
        xytext=(5, 5),       # 데이터 포인트로부터 오프셋 (x, y)
        ha="center",         # 텍스트 정렬 (horizontal alignment)
        color='blue'
    )

for i in range(len(results2[:-1])):
    plt.annotate(
        f"({results2[i]})",  # 표시할 텍스트
        (xticks_[i], results2[i]+0.3),        # 텍스트 위치
        textcoords="offset points",  # 텍스트 위치 기준
        xytext=(5, 5),       # 데이터 포인트로부터 오프셋 (x, y)
        ha="center",         # 텍스트 정렬 (horizontal alignment)
        color='orange'
    )


plt.xticks(xticks_, xticks_dt, fontsize=12)
plt.xlabel("Minimum Counts of Datasets", fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel("Detection Rate(%)", fontsize=15)
plt.ylim(85, 102)
plt.title("", fontsize=18)
plt.legend(fontsize=15)
plt.tight_layout()
































#%% -------------------------------------------------------------
#  try 10 번에 관한 평균 accuracy 구하기.
# ---------------------------------------------------------------

import pandas as pd
import os 
import matplotlib.pyplot as plt

# 불러올 accuracy 결과 폴더 최상단.
base = f"./2.model/241121data_densenet1d_combi/final_seedfix812"
dt_folders = os.listdir(base)
dt_folders.sort()

new_dt_folders = []

for i in dt_folders:
    if '1129' not in i:
        print(i)
        new_dt_folders.append(i)

new_dt_folders

#%%

final_results = []
real_results = []

temp_stop = 0


for dt_folder in new_dt_folders:  # data 폴더에 대해 반복
    each_results = []
    trylist = os.listdir(os.path.join(base, dt_folder, "block2,4,8,4"))
    trylist.remove("1.0,1.0,1.0")
    trylist.sort()
    
    for eachtry in trylist: # try 0~9 에 관한 반복
        infiles = os.listdir(os.path.join(base, dt_folder, "block2,4,8,4", eachtry, "1.0,1.0,1.0"))
        for resultfiles in infiles: # 최종 결과 폴더 안의 파일들
            if resultfiles.endswith('.txt'):
                txt_path = os.path.join(base, dt_folder, "block2,4,8,4", eachtry, "1.0,1.0,1.0", resultfiles)
                with open(txt_path,'r') as fff:
                    lines = fff.readlines()
                acc = list(lines[10].split())[3][:-1]
                each_results.append(round(float(acc), 2))
    real_results.append(each_results)
    final_results.append(round(sum(each_results)/10, 2))
    
    
    if temp_stop == 11:
        break
    
    temp_stop += 1
    

final_results

#%%
xticks_ = [i for i in range(0,10)]
xticks_dt = [f"try{i}" for i in range(1,11)]
plt.figure(figsize=(10,3))
plt.plot(real_results[9], marker='o', label="DenseNet")
plt.xticks(xticks_, xticks_dt, fontsize=12)
plt.xlabel("Minimum Counts", fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel("Detection Rate(%)", fontsize=15)
plt.ylim(60, 100)
plt.title("", fontsize=18)
plt.legend(fontsize=15)
plt.tight_layout()


#%%
xticks_ = [i for i in range(0,15)]
xticks_dt = ["500"]+[str(i*1000+1000) for i in range(0,14)]
plt.figure(figsize=(10,3))
plt.plot(final_results, marker='o', label="DenseNet")
plt.xticks(xticks_, xticks_dt, fontsize=12)
plt.xlabel("Minimum Counts", fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel("Detection Rate(%)", fontsize=15)
plt.ylim(60, 100)
plt.title("", fontsize=18)
plt.legend(fontsize=15)
plt.tight_layout()





















































#%% ====================================================================================================================================================================================
# 배율로 비교 ~ Accuracy 비교 그래프 | width, depth, resol 간.       
# ============================================================================================================================================================
import os

# 현재 디렉토리의 폴더 리스트 출력

coeff = 'resol'
# densenet
#dirname = f'241121data_densenet1d_combi/241121_set15000_min500_max1000_norm_all/block2,4,8,4/{coeff}'

# efficient
dirname = f'241121_effinet_combi/241121_set15000_min500_max1000_norm_all/blocks_arg6/{coeff}'

folders = [f.name for f in os.scandir(f'./2.model/{dirname}') if f.is_dir()]
folders.sort()

#%% =====> DenseNet 에서 빼오기!
import pandas as pd

finaldirec = f"./2.model/{dirname}"

df = pd.DataFrame(columns=['coefs', 'acc', 'FLOPs', 'params'])

for folder in folders:
    if 'phi' in folder:  # phi 제외
        continue
    finalfile = os.path.join(finaldirec, folder)
    for filename in os.listdir(finalfile):
        if filename.endswith('.txt'):
            txt_path = os.path.join(finalfile, filename)
            with open(txt_path,'r') as fff:
                lines = fff.readlines()
            params = list(lines[3].split())[2]    
            flops = list(lines[5].split())[2]
            acc = list(lines[10].split())[3][:-1]
            coef = folder
            
            new_row = {'coefs' : folder,
                       'acc' : round(float(acc),2),
                       'FLOPs' : int(flops),
                       'params' : int(params)}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            
#%% =====> EfficientNet 에서 빼오기!
import pandas as pd

finaldirec = f"./2.model/{dirname}"

df = pd.DataFrame(columns=['coefs', 'acc', 'FLOPs', 'params'])

for folder in folders:
    if 'phi' in folder:  # phi 제외
        continue
    finalfile = os.path.join(finaldirec, folder)
    for filename in os.listdir(finalfile):
        if filename.endswith('.txt'):
            txt_path = os.path.join(finalfile, filename)
            with open(txt_path,'r') as fff:
                lines = fff.readlines()
            params = list(lines[3].split())[2]    
            flops = list(lines[5].split())[2]
            acc = list(lines[13].split())[3][:-1]
            coef = folder
            
            new_row = {'coefs' : folder,
                       'acc' : round(float(acc),2),
                       'FLOPs' : int(flops),
                       'params' : int(params)}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            
            
#%%
df = df.sort_values(by='FLOPs', ascending=True)
df

#%%
df_y = df['acc'].to_numpy()
df_x = df['FLOPs'].to_numpy()

#%%
#df_xtick = [round(i/1_000_000) for i in df_x]
#df_x_label = [str(round(i/1_000_000)) for i in df_x]
#df_xtick

df_xtick = [round(0.5+0.1*i,1) for i in range(0,16)]
df_xtick


#%%
import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(12,3))
plt.plot(df_xtick, df_y, marker='^', markersize=10, linewidth=3)
plt.axvline(x=df_xtick[np.argmax(df_y)], color='red', linestyle='--', alpha=0.8)
plt.axhline(y=max(df_y), color='red', linestyle='--', label=f"{max(df_y)}", alpha=0.8)
plt.title(f"Compound Scaling - {coeff}", fontsize=15)
plt.xlabel("coefficient scaling", fontsize=15)
plt.xticks(df_xtick,df_xtick, fontsize=15)
plt.ylabel("Accuracy", fontsize=15)
plt.yticks([75,80,85], fontsize=15)
plt.tight_layout()
plt.gcf().autofmt_xdate()
plt.legend(fontsize=15)




#%% ===========================================================================
# FLOPS 비교 ~ Accuracy 비교 그래프 | phi coeff 로
import os

# 현재 디렉토리의 폴더 리스트 출력

coeff = 'width'
# densenet
dirname = f'241121data_densenet1d_combi/241121_set15000_min500_max1000_norm_all/block2,4,8,4/{coeff}'

# efficient
#dirname = ''

folders = [f.name for f in os.scandir(f'./2.model/{dirname}') if f.is_dir()]
folders.sort()

#%%
import pandas as pd

finaldirec = f"./2.model/{dirname}"

df = pd.DataFrame(columns=['coefs', 'acc', 'FLOPs', 'params'])

for folder in folders:
    if 'phi' in folder:  # phi 제외
        continue
    finalfile = os.path.join(finaldirec, folder)
    for filename in os.listdir(finalfile):
        if filename.endswith('.txt'):
            txt_path = os.path.join(finalfile, filename)
            with open(txt_path,'r') as fff:
                lines = fff.readlines()
            params = list(lines[3].split())[2]    
            flops = list(lines[5].split())[2]
            acc = list(lines[10].split())[3][:-1]
            coef = folder
            
            new_row = {'coefs' : folder,
                       'acc' : round(float(acc),2),
                       'FLOPs' : int(flops),
                       'params' : int(params)}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            
            
#%%
df = df.sort_values(by='FLOPs', ascending=True)
df

#%%
df_y = df['acc'].to_numpy()
df_x = df['FLOPs'].to_numpy()

#%%
df_xtick = [round(i/1_000_000) for i in df_x]
df_x_label = [str(round(i/1_000_000)) for i in df_x]
df_xtick

#%%
df_xtick_new = [i*5+5 for i in range(1,14)]
df_xtick_new

#%%
import matplotlib.pyplot as plt

plt.figure(figsize=(12,3))
plt.plot(df_xtick, df_y, marker='^', markersize=10, linewidth=3)
#plt.axhline(y=82, color='red', linestyle='--', label="y=78", alpha=0.8)
#plt.axhline(y=78, color='black', linestyle='-.', label="y=78", alpha=0.8)
plt.axhline(y=max(df_y), color='red', linestyle='--', label=f"{max(df_y)}", alpha=0.8)
plt.title(f"Compound Scaling - {coeff}", fontsize=15)
plt.xlabel("FLOPs (x1_000_000)", fontsize=15)
plt.xticks(df_xtick_new,df_xtick_new, fontsize=15)
plt.ylabel("Accuracy", fontsize=15)
plt.yticks([75,80,85], fontsize=15)
plt.tight_layout()
plt.gcf().autofmt_xdate()
plt.legend(fontsize=15)















#%%----------------------------------------------------------------------
#%% coefs 조합중 최고 acc 찾기
import os

# 현재 디렉토리의 폴더 리스트 출력
dirname2 = '241113_densenet1d_combi/241113_count5000down_3series_merged_12000_all_orispec_normed/block2,4,8,4/2.2,2.9,0.2'
folders2 = [f.name for f in os.scandir(f'./2.model/{dirname2}') if f.is_dir()]
folders2

#%%
import pandas as pd

finaldirec2 = f"./2.model/{dirname2}"

df2 = pd.DataFrame(columns=['coefs', 'acc', 'FLOPs', 'params'])

for folder in folders2:
    if 'phi' in folder:  # phi 제외
        continue
    finalfile = os.path.join(finaldirec2, folder)
    for filename in os.listdir(finalfile):
        if filename.endswith('.txt'):
            txt_path = os.path.join(finalfile, filename)
            with open(txt_path,'r') as fff:
                lines = fff.readlines()
            params = list(lines[3].split())[2]    
            flops = list(lines[5].split())[2]
            acc = list(lines[10].split())[3][:-1]
            coef = folder
            
            new_row = {'coefs':folder,
                       'acc':float(acc),
                       'FLOPs':int(flops),
                       'params':int(params)}
            df2 = pd.concat([df2, pd.DataFrame([new_row])], ignore_index=True)
#%%
df2["Score"] = df2['acc']/(35906146/df2['FLOPs'])
df2
#%%
df2 = df2.sort_values(by='FLOPs', ascending=True)
df2
#%%
df2_y = df2['acc'].to_numpy()
df2_x = df2['FLOPs'].to_numpy()

df2_xtick = [round(i/1000000) for i in df2_x]
df2_x_label = [str(round(i/1000000)) for i in df2_x]












df_xticks_evenidx = [df_xtick[i] for i in range(0,len(df_xtick),2)]
df_y_evenidx = [df_y[i] for i in range(0,len(df_y),2)]







#%%
import matplotlib.pyplot as plt

plt.figure(figsize=(12,3))
#plt.plot(df_xtick[:-2], df_y[:-2], marker='^', markersize=10, linewidth=3)
plt.plot(df_xtick[:-1], df_y[:-1], marker='^', markersize=10, linewidth=3, alpha=0.7)
plt.plot(df2_xtick[:15], df2_y[:15],marker='^', markersize=10, linewidth=3, alpha=0.7)
#plt.axhline(y=85.1, color='red', linestyle='--', label="y=78", alpha=0.8)
plt.axhline(y=78, color='black', linestyle='-.', label="y=78", alpha=0.8)
plt.axvline(x=9.6,  color='blue', linestyle='--', label="Default", alpha=0.8)
plt.axvline(x=20.8,  color='orange', linestyle='--', label="Default", alpha=1)

plt.title("Compound Scaling")
plt.xlabel("FLOPs (x1_000_000)")
plt.xticks(df_xtick_new,df_xtick_new)
plt.ylabel("Accuracy")
plt.yticks([77,82,87])
plt.tight_layout()
plt.gcf().autofmt_xdate()
# %%
sum(df2_y[1:14])/13
sum(df_y[21:-2])/11




























#%% ========================================================================================================================================================================================
# ROC graph - sensitivity, 등 지표까지 모두..
# ========================================================================================================================================================================================

#%% 평가지표 뽑을 것들
import pandas as pd
import copy

# dataframe 기초 양식 생성
columns = [str(i*2000+1000) for i in range(0,8)]
init_dict = {col: {"Detection Rate":None, "Precision":None, "Sensitivity":None, "Specificity":None, "F1 Score":None}
             for col in columns}

# source 별로 하나씩 데이터 생성
sources = ["ba", "cs", "na", "co57", "th", "ra", "am"]
final_dt = pd.DataFrame(init_dict)

# 모든 dataframe 하나의 dictionary 로 묶기
overall_pd = {}

for source in sources:
    overall_pd[f"{source}"] = copy.deepcopy(final_dt)

eval_cols = ["Detection Rate", "Sensitivity", "Specificity", "Precision", "F1 Score"]

overall_pd
#%% 241202 귀찮게 하나하나 confusion matrix 까야할 때.
import pandas as pd
import os 

# 불러올 accuracy 결과 폴더 최상단.
base = f"./2.model/241215data_densenet1d_nonpoisson/"
dt_folders = os.listdir(base)
dt_folders.sort()

dt_folders.remove("wrong_data")
dt_folders
#%%
new_dt_folders = []

for i in dt_folders:
    if '1129' not in i:
        print(i)
        new_dt_folders.append(i)

new_dt_folders
#%%

#filename_isold = 1

# 4중첩 for문이라니....
# 각 폴더별 반복, 241121 결과만.
for dt_folder in new_dt_folders:
    inner_folder = os.listdir(os.path.join(base, dt_folder, "1.0,1.0,1.0/predict/"))
    for filename in inner_folder:
        # 오류로 학습안되어있는 폴더 존재하기 때문에(txt 파일이 있어야 학습이 정상적으로 이루어진 것)
        if not filename.endswith(".csv"):
            continue
        
        df = pd.read_csv(os.path.join(base, dt_folder, "1.0,1.0,1.0/predict/", filename))
        print(filename[:40], "===============================")
        print("precision : ", df.iloc[7].mean())
        print("Sensitivity : ", df.iloc[5].mean())
        print("specificity : ", df.iloc[6].mean())
        print("f1 score : ", df.iloc[8].mean())
        
        fmci = dt_folder.find("min") # file_min_count_idx
        
        ''' filename 에서 찾으려고 했는데 그냥 디렉토리 name 에서 찾아야할듯
        if filename_isold:
            if filename[fmci+3+3] == '_': # min 500 부분 글자 개수 때문에 예외처리
                file_min_count = filename[fmci+3:fmci+3+3]
            else:
                file_min_count = filename[fmci+3:fmci+3+4]
        else:
            file_min_count = str(round(int(filename[fmci+3:fmci+3+5]),-3))
        '''
        file_min_count = str(round(int(dt_folder[fmci+3:fmci+3+5]),-2)) # min 시작 index 3개 이후부터 숫자, 총 5글자가 숫자
        
        # df 의 평가지표 순서는 차례대로 Sen, Spe, Pre, F1 임.    
        for source in sources:
            for eval_idx in range(5): # eval_cols 개수만큼 반복
                overall_pd[source][file_min_count][eval_cols[eval_idx]] = round(df[source][eval_idx+4], 2) # eval_idx 부터 +4 가 accuracy ~ 지표들 (df 상 row)

overall_pd










#%% 6.AUC Curve ===========================================================================================
import numpy as np
import matplotlib.pyplot as plt

base = f"./2.model/241215_efficient_nonpoisson/"
dt_folders = os.listdir(base)
dt_folders.sort()

dt_folders
#%%
# densenet
inner_folder = os.listdir(os.path.join(base, dt_folders[0], "1.0,1.0,1.0/predict/"))
pred_data_path = os.path.join(os.path.join(base, dt_folders[0], "1.0,1.0,1.0/predict/"), inner_folder[0])

# efficientnet
inner_folder = os.listdir(os.path.join(base, dt_folders[0], "blocks_arg6/1.0,1.0,1.0/predict/"))
pred_data_path = os.path.join(os.path.join(base, dt_folders[0], "blocks_arg6/1.0,1.0,1.0/predict/"), inner_folder[0])


pred = np.load(pred_data_path)

y_test = np.load("./1.preprocess_data/241215_set10000_min01040_max01999_all.npz")["y_test"][:,0,:]




#%%
data_num = pred.shape[0]
roc_len = 20  # thres 간격 개수
src_num = 8
roc_values = np.zeros([src_num, 6, roc_len]) # 행 : source, 열 : thres
# row 의 6은 순서대로 : TP, TN, FP, FN, TPR, FPR

thres_col = 0
for thres in np.arange(0, 1, 0.05):
    TF_pred = pred > thres  
    TF_y = y_test > thres  # 한 행당 [T,F,F,F] 이런식

    for src_dim in range(src_num): # source 에 대한 반복
        for cnt_row in range(data_num): # data 개수에 대한 반복
            
            if TF_pred[cnt_row, src_dim] == TF_y[cnt_row, src_dim]:
                if TF_pred[cnt_row, src_dim] == 1:
                    roc_values[src_dim, 0, thres_col] += 1   # TP : pos를 정확히 예측
                else:
                    roc_values[src_dim, 1, thres_col] += 1   # TN : neg를 정확히 예측
            else:
                if TF_pred[cnt_row, src_dim] == 1:   # FP : 가짜 True
                    roc_values[src_dim, 2, thres_col] += 1
                else:   # FN 
                    roc_values[src_dim, 3, thres_col] += 1
    
    thres_col += 1

# TPR
roc_values[:,4,:] = roc_values[:,0,:] / (roc_values[:,0,:] + roc_values[:,3,:] + 0.00001)

# FPR
roc_values[:,5,:] = roc_values[:,2,:] / (roc_values[:,2,:] + roc_values[:,1,:] + 0.00001)

#%% 6-2. Plot ROC ! ===========================================================================================
import matplotlib.pyplot as plt

fig, ax = plt.subplots(8, 1, figsize=(3,20))

temp = np.linspace(0,1,roc_len)

source_name = ["Ba-133", "Cs-137", "Na-22", "background", "co57", "Th-232", "ra", "Am-241"]
using_source_idx = [0,1,2,5,7]

for i in range(8):
    area = round(abs(np.trapz(np.append(roc_values[i,4,:],0), np.append(roc_values[i,5,:],0))),3)
    ax[i].plot(np.append(roc_values[i,5,:],0), np.append(roc_values[i,4,:],0),
               linewidth='2', color='red', marker='>')
    ax[i].plot(temp, temp, linestyle=':', linewidth='1', color='blue')
    #ax[i].grid(True)s
    ax[i].set_title(f"{source_name[i]}, AUC : {area}")
    ax[i].set_xlabel("False Positive Rate")
    ax[i].set_ylabel("True Positive Rate")
    #ax[i].text(0.8,0.8, f"AUC : {area}", color='red', fontsize=20, 
    #           bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
    #ax[i].margins(x=0,y=0)

plt.subplots_adjust(wspace=15)
plt.tight_layout()
plt.show()















#%% ========================================================================================================================================================================================
# Source original spectrum 출력
# ========================================================================================================================================================================================

import sys
import os

sys.path.append(os.path.join(os.getcwd(), "lib"))
print(sys.path)

from modi_hist_extract import modi_hist_extract
import itertools
import random
import numpy as np
from joblib import Parallel, delayed, cpu_count
import lzma
import pickle
import numpy as np
import matplotlib.pyplot as plt
#from phaseA_csv2dataset_v003 import xz_hist_make

#%%
RA = 'ba'
final_direc = os.path.join(dt_path, RA)

os.listdir(final_direc)
#%%
final_hist = np.zeros([1, 1000])
for file in os.listdir(final_direc):
    file_path = os.path.join(final_direc, file)
    with lzma.open(file_path, 'rb') as f:
        temp = pickle.load(f)
        final_hist[0, :] += temp.reshape(1,1000)[0,:]
        #print("Debug : ", sum(final_hist[0]))


#%%
def normalize(array):
    return (array-min(array)) / (array.max() - array.min())

final_plot = normalize(final_hist[0])
#%%

plt.plot(final_plot, color='blue')

y_max = max(final_plot[300:500])
x_max = np.argmax(final_plot)
#plt.figure(figsize=(12,3))
plt.annotate(
    f'Peak : 356 KeV',  # 표시할 텍스트
    xy=(x_max, y_max),                  # 화살표가 가리키는 좌표
    xytext=(x_max + 100, y_max-0.1),      # 텍스트 위치
    arrowprops=dict(facecolor='red', arrowstyle='->'),  # 화살표 스타일
    fontsize=15,                        # 글씨 크기
    color='red'                         # 글씨 색상
)
plt.title(f"Gamma Energy Spectrum of Ba-133", fontsize=17)
plt.xlabel("Energy bin(keV)", fontsize=15)
plt.ylabel("Counts (A.U.)", fontsize=15)
plt.tight_layout()










#%% dataset 2 --------------------------------------------------------------
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "lib"))
from modi_hist_extract import modi_hist_extract
import numpy as np

datadir = '../../Data/integrated_modified/'
data_date = '240906'
csv_path = os.path.join(datadir, data_date, 'am')

os.listdir(csv_path)
#%%
# 고른 source 부분들 합치기
final_hist = np.zeros([1,1000])

csv_file_list = os.listdir(csv_path)
csv_file_name = os.path.join(csv_path, csv_file_list[0])
csv_file = f"{csv_file_name}"
fil_dt = modi_hist_extract(csv_file)  # filtered data

#%%
def normalize(array):
    return (array-min(array)) / (array.max() - array.min())

#%%
final_plot = fil_dt.hist

plt.plot(normalize(final_plot), color='red')

y_max = max(final_plot[400:700])
x_max = np.argmax(final_plot)
#plt.figure(figsize=(12,3))
plt.annotate(
    f'Peak : 356 KeV',  # 표시할 텍스트
    xy=(x_max, y_max),                  # 화살표가 가리키는 좌표
    xytext=(x_max + 100, y_max-0.1),      # 텍스트 위치
    arrowprops=dict(facecolor='red', arrowstyle='->'),  # 화살표 스타일
    fontsize=15,                        # 글씨 크기
    color='red'                         # 글씨 색상
)
plt.title(f"Gamma Energy Spectrum of Am-241", fontsize=17)
plt.xlabel("Energy bin(keV)", fontsize=15)
plt.ylabel("Counts (A.U.)", fontsize=15)
plt.tight_layout()


