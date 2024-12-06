#%%

import numpy as np
import sys
import os
import matplotlib.pyplot as plt
sys.path.append('./')

from lib.modi_hist_extract import modi_hist_extract
from lib.modwt_pkg import *
#%%
#dt = np.load("./0.dataset/241113_250to300sec_8source_5000.npz",)
dt = np.load("./0.dataset/241113_multiple240627_250to300sec_8source_5000.npz",)
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
columns = ["500"]+[str(i*1000+1000) for i in range(0,5)]
init_dict = {col: {"Precision":None, "Sensitivity":None, "Specificity":None, "F1 Score":None}
             for col in columns}

# source 별로 하나씩 데이터 생성
sources = ["ba", "cs", "na", "co57", "th", "ra", "am"]
final_dt = pd.DataFrame(init_dict)

# 모든 dataframe 하나의 dictionary 로 묶기
overall_pd = {}

for source in sources:
    overall_pd[f"{source}"] = copy.deepcopy(final_dt)

eval_cols = ["Sensitivity", "Specificity", "Precision", "F1 Score"]

overall_pd
#%% 241202 귀찮게 하나하나 confusion matrix 까야할 때. - DenseNet
import pandas as pd
import os 

# 불러올 accuracy 결과 폴더 최상단.
base = f"./2.model/241121data_densenet1d_combi/"
dt_folders = os.listdir(base)
dt_folders.sort()

# 4중첩 for문이라니....
# 각 폴더별 반복, 241121 결과만.
for dt_folder in dt_folders[:6]:
    inner_folder = os.listdir(os.path.join(base, dt_folder, "block2,4,8,4/1.0,1.0,1.0/predict/"))
    for filename in inner_folder:

        # 오류로 학습안되어있는 폴더 존재하기 때문에(txt 파일이 있어야 학습이 정상적으로 이루어진 것)
        try:
            
            df = pd.read_csv(os.path.join(base, dt_folder, "block2,4,8,4/1.0,1.0,1.0/predict/", filename))
            print(filename[:40], "===============================")
            print("precision : ", df.iloc[7].mean())
            print("Sensitivity : ", df.iloc[5].mean())
            print("specificity : ", df.iloc[6].mean())
            print("f1 score : ", df.iloc[8].mean())
            
            fmci = filename.find("min") # file_min_count_idx
            if filename[fmci+3+3] == '_': # min 500 부분 글자 개수 때문에 예외처리
                file_min_count = filename[fmci+3:fmci+3+3]
            else:
                file_min_count = filename[fmci+3:fmci+3+4]
            
            # df 의 평가지표 순서는 차례대로 Sen, Spe, Pre, F1 임.    
            for source in sources:
                for eval_idx in range(4): # eval_cols 개수만큼 반복
                    overall_pd[source][file_min_count][eval_cols[eval_idx]] = round(df[source][eval_idx+5], 2)
        except:
            pass

overall_pd

#%% 241202 귀찮게 하나하나 confusion matrix 까야할 때. - EfficientNet
import pandas as pd
import os 
base = f"./2.model/241121_effinet_combi/"
dt_folders = os.listdir(base)
dt_folders.sort()

for dt_folder in dt_folders[:6]:
    inner_folder = os.listdir(os.path.join(base, dt_folder, "blocks_arg6/1.0,1.0,1.0/predict/"))
    for filename in inner_folder:
        try:
            df = pd.read_csv(os.path.join(base, dt_folder, "blocks_arg6/1.0,1.0,1.0/predict/", filename))
            print(filename[:40], "===============================")
            print("precision : ", df.iloc[7].mean())
            print("Sensitivity : ", df.iloc[5].mean())
            print("specificity : ", df.iloc[6].mean())
            print("f1 score : ", df.iloc[8].mean())
            
        except:
            pass

#%% 241129 500-6000 data counts 비교 ==================================================================
import matplotlib.pyplot as plt
x = [76.9, 86.6, 90.9, 89.6, 70.4, 78.6]
x1 = [78.8, 87, 89.9, 94.5, 93.9, 96.1]
plt.plot(x, marker='o', label="DenseNet")
plt.plot(x1, marker='o', label="EfficientNet")
plt.xticks([0,1,2,3,4,5],["500","1000","2000","3000","4000","5000"], fontsize=15)
plt.xlabel("Minimum Counts of Data", fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel("Detection Rate(%)", fontsize=15)
plt.ylim(50, 100)
plt.title("", fontsize=18)
plt.legend(fontsize=15)


#%% Macro Precision -----------------------------
import matplotlib.pyplot as plt
d1 = [0.91, 0.96, 0.97, 0.96, 0.94, 0.96]
e1 = [0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
plt.plot(d1, marker='o', label="DenseNet")
plt.plot(e1, marker='o', label="EfficientNet")
plt.xticks([0,1,2,3,4,5],["500","1000","2000","3000","4000","5000"], fontsize=15)
plt.xlabel("Minimum Counts of Data", fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel("[TP / (TP+FP)]", fontsize=15)
plt.ylim(0.5, 1.05)
plt.title("Macro Precision", fontsize=18)
plt.legend(fontsize=15)

#%% Macro  Sensitivity | recall -----------------------------
import matplotlib.pyplot as plt
d1 = [0.88, 0.94, 0.96, 0.96, 0.83, 0.92]
e1 = [0.88, 0.93, 0.96, 0.98, 0.97, 0.98]
plt.plot(d1, marker='o', label="DenseNet")
plt.plot(e1, marker='o', label="EfficientNet")
plt.xticks([0,1,2,3,4,5],["500","1000","2000","3000","4000","5000"], fontsize=15)
plt.xlabel("Minimum Counts of Data", fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel("[TP / (TP+FN)]", fontsize=15)
plt.ylim(0.5, 1.05)
plt.title("Macro Sensitivity", fontsize=18)
plt.legend(fontsize=15)

#%% Macro  Specificity -----------------------------
import matplotlib.pyplot as plt
d1 = [0.97, 0.98, 0.99, 0.98, 0.92, 0.95]
e1 = [0.98, 0.98, 0.99, 0.99, 0.99, 0.99]
plt.plot(d1, marker='o', label="DenseNet")
plt.plot(e1, marker='o', label="EfficientNet")
plt.xticks([0,1,2,3,4,5],["500","1000","2000","3000","4000","5000"], fontsize=15)
plt.xlabel("Minimum Counts of Data", fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel("[TN / (FP+TN)]", fontsize=15)
plt.ylim(0.5, 1.05)
plt.title("Macro Specificity", fontsize=18)
plt.legend(fontsize=15)

#%% Macro  F1 score -----------------------------
import matplotlib.pyplot as plt
d1 = [0.89, 0.95, 0.96, 0.96, 0.86, 0.93]
e1 = [0.91, 0.94, 0.96, 0.98, 0.97, 0.98]
plt.plot(d1, marker='o', label="DenseNet")
plt.plot(e1, marker='o', label="EfficientNet")
plt.xticks([0,1,2,3,4,5],["500","1000","2000","3000","4000","5000"], fontsize=15)
plt.xlabel("Minimum Counts of Data", fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel("F1 Score", fontsize=15)
plt.ylim(0.5, 1.05)
plt.title("Macro F1 Score", fontsize=18)
plt.legend(fontsize=15)

# =======================================================================
#%% 각 선원별 pre, sen, spe, f1 --------------------------------------
import matplotlib.pyplot as plt

plt.figure()
for source in sources:
    plt.plot(overall_pd[source].loc["Precision"].to_numpy(), label=f"{source}")
plt.xticks([0,1,2,3,4,5],["500","1000","2000","3000","4000","5000"], fontsize=15)
plt.xlabel("Minimum Counts of Data", fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel("[TP / (TP+FP)]", fontsize=15)
plt.ylim(0.60, 1.05)
plt.title("Precision", fontsize=18)
plt.legend(fontsize=15, loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()

#%% --------------------------------------
plt.figure()
for source in sources:
    plt.plot(overall_pd[source].loc["Sensitivity"].to_numpy(), label=f"{source}")
plt.xticks([0,1,2,3,4,5],["500","1000","2000","3000","4000","5000"], fontsize=15)
plt.xlabel("Minimum Counts of Data", fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel("[TP / (TP+FN)]", fontsize=15)
plt.ylim(0.4, 1.05)
plt.title("Sensitivity", fontsize=18) #recall
plt.legend(fontsize=15, loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()

#%% --------------------------------------
plt.figure()
for source in sources:
    plt.plot(overall_pd[source].loc["Specificity"].to_numpy(), label=f"{source}")
plt.xticks([0,1,2,3,4,5],["500","1000","2000","3000","4000","5000"], fontsize=15)
plt.xlabel("Minimum Counts of Data", fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel("[TN / (FP+TN)]", fontsize=15)
plt.ylim(0.40, 1.05)
plt.title("Specificity", fontsize=18)
plt.legend(fontsize=15, loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()

#%% --------------------------------------
plt.figure()
for source in sources:
    plt.plot(overall_pd[source].loc["F1 Score"].to_numpy(), label=f"{source}")
plt.xticks([0,1,2,3,4,5],["500","1000","2000","3000","4000","5000"], fontsize=15)
plt.xlabel("Minimum Counts of Data", fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel("2*(Pre*sen)/(Pre+Sen)", fontsize=15)
plt.ylim(0.60, 1.05)
plt.title("F1 Score", fontsize=18)
plt.legend(fontsize=15, loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()














#%% 241202 count 별 비교 in single source ==================================================================
import matplotlib.pyplot as plt
x = [99.9, 99.9, 100, 100, 100, 100]
x1 = [99.2, 99.9, 98, 99.9, 100, 100]
plt.plot(x, marker='o', label="DenseNet")
plt.plot(x1, marker='o', label="EfficientNet")
plt.xticks([0,1,2,3,4,5],["500","1000","2000","3000","4000","5000"], fontsize=15)
plt.xlabel("Minimum Counts of Data", fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel("Detection Rate(%)", fontsize=15)
plt.ylim(50, 100)
plt.title("", fontsize=18)
plt.legend(fontsize=15)






#%% ====================================================================================================================================================================================
# 모든 것을 동일하게 했을 때의 결과 비교. 
# ============================================================================================================================================================
import pandas as pd
import os 
import matplotlib.pyplot as plt

# 불러올 accuracy 결과 폴더 최상단.
base = f"./2.model/241121data_densenet1d_combi/"
dt_folders = os.listdir(base)
dt_folders.sort()

actual_folder = os.path.join(base, dt_folders[3], "block2,4,8,4", "default_retry_seedfix")
actual_folder
results = []

for folder in os.listdir(actual_folder):
    inner_folder = os.path.join(actual_folder, folder, "1.0,1.0,1.0")
    for filename in os.listdir(inner_folder):
        if filename.endswith('.txt'):
            txt_path = os.path.join(inner_folder, filename)
            with open(txt_path,'r') as fff:
                lines = fff.readlines()
            acc = list(lines[10].split())[3][:-1]
            results.append(round(float(acc), 2))

xticks_ = [i for i in range(1,11)]
plt.plot(results, marker='o', label="DenseNet")
plt.xticks(xticks_, xticks_, fontsize=15)
plt.xlabel("trial", fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel("Detection Rate(%)", fontsize=15)
plt.ylim(70, 100)
plt.title("", fontsize=18)
plt.legend(fontsize=15)








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



