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

import os
import pickle
import numpy as np
from joblib import Parallel, delayed, cpu_count
import lzma
import matplotlib.pyplot as plt
import pandas as pd
from modwt_pkg import *
from modi_hist_extract import modi_hist_extract
from poisson_deviance import my_pos_dev
import copy


'''
====================================================================================
'''

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
'''


def process_filtering_1data(i_):
    global pre_data
    
    # ut = pre_data["x_back"][i_, :, :]  # -> 처음 dataset 구성에서 같은 시간동안 가져온 background 만 한 것. -> 그러나 background 는 긴 시간동안 받은 파형을 토대로 이용할 예정이기 때문에 주석처리
    ut_hat = (ut)/sum(abs(ut[0]))    
    
    ori_spectrum = pre_data["x_ori"][i_, :, :]
    mixed_spectrum = copy.deepcopy(ori_spectrum)                # shape (1,1000)
    filtered_ori = noisefiltering(mixed_spectrum,10,0)
    
    mixed_spectrum = mixed_spectrum / sum(mixed_spectrum[0])  # l1 normalize 로 교체 -  240923
    filtered1 = noisefiltering(mixed_spectrum,0,0)
    derivatives1 = derivative_signal(filtered1)
    filtered2 = noisefiltering2(derivatives1,0,0)
    pos_dev = my_pos_dev(mixed_spectrum[0,:], ut_hat[0,:])
    pos_dev = pos_dev[np.newaxis,:]   
    pos_dev_fil = noisefiltering(pos_dev,0,0)   
    
    x_temp = np.zeros((1, 1, 1000, 3))
    x_temp[0, :, :, 0] = filtered_ori
    x_temp[0, :, :, 1] = filtered2
    x_temp[0, :, :, 2] = pos_dev_fil

    return x_temp


'''
# # 0.dataset에 있는 npy로 최종 model input용 dataset 생성.
# def process_dataset_csv(start:int, end:int):
#     num_cores = cpu_count()
#     tasks = range(start, end)
#     results = Parallel(n_jobs=num_cores, verbose=10)(delayed(process_filtering_1data)(ii) for ii in tasks)

#     return results
'''



def process_all(train_n, val_n, test_n):
    global pre_data
    num_cores = cpu_count()
    
    # train / val / test  flag 설정
    if train_n > 0 : train_flag = 1
    else: train_flag = 0
    if val_n > 0 : vali_flag = 1
    else: vali_flag = 0
    if test_n > 0 : test_flag = 1
    else : test_flag = 0
    
    #each_core = num_cores // (train_flag+vali_flag+test_flag)
    each_core = num_cores
    
    # 각기 시작점 설정
    train_s = 0
    train_e = train_n
    
    val_s = train_n
    val_e = val_s + val_n
    
    test_s = val_e
    test_e = test_s + test_n

    print(train_s, train_e)
    # train
    if train_flag:
        results1 = Parallel(n_jobs=each_core, backend="threading", verbose=10)(delayed(process_filtering_1data)(i) for i in range(train_s, train_e))
    else:
        results1 = None
    
    # validation
    if vali_flag:
        results2 = Parallel(n_jobs=each_core, backend="threading", verbose=10)(delayed(process_filtering_1data)(ii) for ii in range(val_s, val_e))
    else:
        results2 = None
    
    # test
    if test_flag:
        results3 = Parallel(n_jobs=each_core,  backend="threading", verbose=10)(delayed(process_filtering_1data)(iii) for iii in range(test_s, test_e))
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
        y_train = pre_data['y'][train_s:train_e]
    else:
        y_train = []
    
    if vali_flag:
        for x_comp in results2:
            x_val.extend(x_comp)
        x_val = np.vstack(x_val)
        y_val = pre_data['y'][val_s:val_e]
    else:
        y_val = []
    
    if test_flag:
        for x_comp in results3:
            x_test.extend(x_comp)
        x_test = np.vstack(x_test)
        y_test = pre_data['y'][test_s:test_e]
    else:
        y_test = []

    
    print("Work Done!")
    return x_train, y_train, x_val, y_val, x_test, y_test


'''
<<< method loaded >>>
'''

#%% 1. "0.dataset"에서 data load ===========================================================
import numpy as np
#filename = "240806_4source_3000_xzfile"
#pre_data = np.load(f"./0.dataset/{filename}.npy")   # (datanum, 1, 1000)
#pre_data_y = np.load(f"./1.preprocess_data/{filename}_y.npy")  # [[4개] x datanum]
#filename = "./0.dataset/240923_10to20sec_8source_10000.npz"
#filename = "./0.dataset/240923_multiple240627_10to20sec_8source_1000.npz"
#filename = "./0.dataset/240923_10to20_3source_10000_xzfile.npz"

# 241113
#filen = "241121_dataset2_set5000_min500_max1000.npz"
#filen = "241121_multiple240627_dataset3_set5000_min500_max1000.npz"

'''
filenames = [
    "241129_dataset1_set5000_min500_max999_xzfile_combi1.npz",
    "241129_dataset1_set5000_min1037_max1998_xzfile_combi1.npz",
    "241129_dataset1_set5000_min2003_max3000_xzfile_combi1.npz",
    "241129_dataset1_set5000_min3000_max3999_xzfile_combi1.npz",
    "241129_dataset1_set5000_min4001_max4994_xzfile_combi1.npz",
    "241129_dataset1_set5000_min5014_max5999_xzfile_combi1.npz"
]

filenames = [
    "241129_dataset2_set10000_min500_max1000_combi1.npz",
    "241129_dataset2_set10000_min1000_max2000_combi1.npz",
    "241129_dataset2_set10000_min2000_max3000_combi1.npz",
    "241129_dataset2_set10000_min3000_max4000_combi1.npz",
    "241129_dataset2_set10000_min4000_max5000_combi1.npz",
    "241129_dataset2_set10000_min5000_max6000_combi1.npz"
]

filenames = [
    "241204_dataset2_set5000_min6000_max7000_combi3.npz",
    "241204_dataset2_set5000_min7000_max8000_combi3.npz",
    "241204_dataset2_set5000_min8000_max9000_combi3.npz",
    "241204_dataset2_set5000_min9000_max10000_combi3.npz",
    "241204_dataset2_set5000_min10000_max11000_combi3.npz",
    "241204_dataset2_set5000_min11000_max12000_combi3.npz",
    "241204_dataset2_set5000_min12000_max13000_combi3.npz",
    "241204_dataset2_set5000_min13000_max14000_combi3.npz",
    "241204_dataset2_set5000_min14000_max15000_combi3.npz"
]

filenames = [
    "241204_multiple240627_dataset3_set5000_min6000_max7000.npz",
    "241204_multiple240627_dataset3_set5000_min7000_max8000.npz",
    "241204_multiple240627_dataset3_set5000_min8000_max9000.npz",
    "241204_multiple240627_dataset3_set5000_min9000_max10000.npz",
    "241204_multiple240627_dataset3_set5000_min10000_max11000.npz",
    "241204_multiple240627_dataset3_set5000_min11000_max12000.npz",
    "241204_multiple240627_dataset3_set5000_min12000_max13000.npz",
    "241204_multiple240627_dataset3_set5000_min13000_max14000.npz",
    "241204_multiple240627_dataset3_set5000_min14000_max15000.npz"
]
'''
'''
filenames = [
    #"241204_dataset1_set5000_min6000_max7000_xzfile.npz",
    #"241204_dataset1_set5000_min7000_max8000_xzfile.npz",
    "241204_dataset1_set5000_min8000_max9000_xzfile.npz",
    "241204_dataset1_set5000_min9000_max9999_xzfile.npz",
    "241204_dataset1_set5000_min10001_max10999_xzfile.npz",
    "241204_dataset1_set5000_min11001_max11998_xzfile.npz",
    "241204_dataset1_set5000_min12000_max13000_xzfile.npz",
    "241204_dataset1_set5000_min13000_max13998_xzfile.npz",
    "241204_dataset1_set5000_min14001_max15000_xzfile.npz"
]
'''
'''
filenames = [
    '241211_dataset2_set5000_min1000_max2000_combi3_src5.npz',
    '241211_dataset2_set5000_min3000_max4000_combi3_src5.npz',
    '241211_dataset2_set5000_min5000_max6000_combi3_src5.npz',
    '241211_dataset2_set5000_min7000_max8000_combi3_src5.npz',
    '241211_dataset2_set5000_min9000_max10000_combi3_src5.npz',
    '241211_dataset2_set5000_min11000_max12000_combi3_src5.npz',
    '241211_dataset2_set5000_min13000_max14000_combi3_src5.npz',
    '241211_dataset2_set5000_min15000_max16000_combi3_src5.npz'
]
'''
'''
filenames = [
    "241211_dataset1_set5000_min15000_max16000_xzfile.npz"
]
'''
filenames = [
    '241216_dataset2_set5000_min1000_max2000_combi3.npz',
    '241216_dataset2_set5000_min3000_max4000_combi3.npz',
    '241216_dataset2_set5000_min5000_max6000_combi3.npz',
    '241216_dataset2_set5000_min7000_max8000_combi3.npz',
    '241216_dataset2_set5000_min9000_max10000_combi3.npz',
    '241216_dataset2_set5000_min11000_max12000_combi3.npz',
    '241216_dataset2_set5000_min13000_max14000_combi3.npz',
    '241216_dataset2_set5000_min15000_max16000_combi3.npz'
]

train_n = 3000
val_n = 1000
test_n = 1000


# --- BACKGROUND 지정. 중요! (수집된 BACKGROUND 전체를 지정해야함.) ----
# xz data 는 파형 잘 나와있어서 그냥 background 사용 가능

# dataset 3
#ut_data_name = "../../Data/integrated_modified/240627_t700/background/240627_t700.csv"

# dataset 2
ut_data_name = "../../Data/integrated_modified/240603/background/240603.csv"

# dataset 1
#ut_data_name = "../../Data/230000_forTensorflow_xz/background/"

isxz = 0  # <--------- flag 설정 잊지 않기

if isxz:
    ut = np.zeros([1,1000])
    for ut_filename in os.listdir(ut_data_name):
        xz_back_file = os.path.join(ut_data_name, ut_filename)
        with lzma.open(xz_back_file, 'rb') as f:
            temp = pickle.load(f)
        ut[0, :] += temp.reshape(1,1000)[0,:]  

else:
    ut_dt = modi_hist_extract(ut_data_name)
    ut = ut_dt.hist
    ut = ut[np.newaxis,:]  # (1,1000)



# Start process all!
for filen in filenames:
    try:
        del pre_data
    except:
        pass
    
    try:
        del x_train, y_train, x_val, y_val, x_test, y_test
    except:
        pass
    
    filename = f"./0.dataset/{filen}"

    pre_data = np.load(f"{filename}")
    print(pre_data.files)

    print("최대 count 수 : ", max(np.sum(pre_data["x_ori"][:,0,:],axis=1)))
    print("최소 count 수 : ", min(np.sum(pre_data["x_ori"][:,0,:],axis=1)))

    print("<<< data loaded >>>")
    


    x_train, y_train, x_val, y_val, x_test, y_test = process_all(train_n, val_n, test_n)    # 2.4분만에 완료!

    if train_n:
        print("x_train : ", x_train.shape)
        print("y_train : ", y_train.shape)
    if val_n:
        print("x_val : ", x_val.shape)
        print("y_val : ", y_val.shape)
    if test_n:
        print("x_test : ", x_test.shape)
        print("y_test : ", y_test.shape)


    if not train_n:
        x_train=None
        y_train=None
    if not val_n:
        x_val=None
        y_val=None
    if not test_n:
        x_test=None
        y_test=None

    np.savez(f"./1.preprocess_data/{filen}_fullbackground.npz", 
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test)


# %%
