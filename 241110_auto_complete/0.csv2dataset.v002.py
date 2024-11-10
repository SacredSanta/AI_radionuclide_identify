'''
최종 수정 : 2024.09.21.
사용자 : 서동휘

<수정 내용> 

<24.09.19>
- Ax + b 의 형태로 energy correction 을 진행.
  검출기 단에서 energy correction 을 해야하지만.. 일단 이대로 진행.
  data 보정해서 -> integrated 폴더에 각각 source 별로 정리..? 
  각 experiment 마다 보정 계수 이용해서 일괄적으로 처리하는 것으로 충분할듯.

<24.08.06>
- 정의된 class를 lib의 modi_hist_extract 로 이전.
그러나 csv_hist_modify 는 사용하지 않고 -> csv_hist_modify_old 로 명명
modi_hist_extract로 모두 사용하면 됨.
- gaussian fit 관련된 불필요한 내용 삭제
- <<<chapter 1>>> modi csv 저장부분 수정

<처음>
검출기에서 얻은 csv 데이터파일을 numpy로 전환
energy 위치 안 맞는 부분을 보정해준 후,
조건을 지정하여 그 부분의 data만 저장.
'''

#%% 1. modi csv를 통해 정해진 규칙을 기준으로 data filter 후 train or test data로 생성  import sys
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "lib"))
from modi_hist_extract import modi_hist_extract
import itertools
import random
import numpy as np
from joblib import Parallel, delayed, cpu_count

source = np.array(['ba', 'cs', 'na', 'background', 'co57', 'th', 'ra', 'am'])

# (0) init
total_set = 10000  #! <----------------------이부분 수정해서 이용 !!
index = [i for i in range(len(source))]   # source list에서 뽑을 index
#time_range = [10, 20] # <-------------------- 개인 변경

datadir = '../../Data/integrated_modified/'
data_date = ['240603', '240906']  #'240905' , '240627_t700'
max_hist_count = 5000

back_cutting = 1  # background 를 spectrum 에서 단순 제거를 할지 flag

time_ranges = [[10,20],[20,30],[30,40],[40,50],[50,60]]

# (1) start
def work_1(ii, time_range):

    y = np.zeros([1, len(source)])

    # 우선 어느 날짜 data 로 뽑을지 정하기.
    exp_data = random.choice(data_date)
    dt_path = os.path.join(datadir,exp_data)
    dt_path_source = os.listdir(dt_path)
    dt_path_source.remove('background')
    
    # 개수가 최대 3개를 넘어가지 않게 하기
    if len(dt_path_source) < 3:
        howmany = random.choice(range(1, len(dt_path_source)+1))
    else:
        howmany = random.choice(range(1, 4))

    # 개수만큼의 조합을 구하고, 그 중 하나뽑기 + y(정답)도 지정
    combi = list(itertools.combinations(dt_path_source, howmany))
    final_select = random.choice(combi)
    for i in final_select:
        idx = np.where(source == i)[0][0]
        y[0, idx] = 1

    # (2) 특정 시간 구간에 대한 구현. 
    total_time = 0  # 5초 이내로 작동
    while total_time < time_range[0] or total_time > time_range[1]:  #! <----------------------이부분 수정해서 이용 !!
        starttime = random.choice(range(0, 300)) + random.random()
        endtime = random.choice(range(int(starttime), 301)) + random.random()
        total_time = endtime - starttime

    # (3) data 추출
    # 최종적으로 여러가지가 합쳐진 spectrum 초기화
    
    # background hist 계산
    its_background = np.zeros([1,1000])
    
    csv_back_path = os.path.join(dt_path, 'background', exp_data)
    csv_back_file = f"{csv_back_path}.csv"
    fil_back_dt = modi_hist_extract(csv_back_file)
    fil_back_dt.filtered_counts(max_hist_count, starttime, endtime)
    its_background[0] = fil_back_dt.filtered_hist

    # 고른 source 부분들 합치기
    final_hist = np.zeros([1,1000])
    for src in final_select:  # 뽑힌 source idx에서 반복
        csv_path = os.path.join(dt_path, src)
        csv_file_list = os.listdir(csv_path)
        csv_file_sel = random.choice(csv_file_list)
        csv_file_name = os.path.join(csv_path, csv_file_sel)
        csv_file = f"{csv_file_name}"
        fil_dt = modi_hist_extract(csv_file)  # filtered data
        fil_dt.filtered_counts(max_hist_count, starttime, endtime)
        final_hist[0] += fil_dt.filtered_hist
        
    cutted_hist = np.zeros([1, 1000])
    if back_cutting:
        cutted_hist[0] = final_hist[0] - (its_background[0] * len(final_select))
        cutted_hist[0][cutted_hist[0]<0] = 0 # 음수로 넘어가버리면 안되니 0으로 정정
        
    return final_hist, y, its_background, cutted_hist

for time_ran in time_ranges:
    try:
        del dataset, dataset_y, dataset_back, cutted_hist, filename
    except:
        pass
    
    results = Parallel(n_jobs=cpu_count(), verbose=10)(delayed(work_1)(task, time_ran) for task in range(total_set))

    # 결과 받고 numpy로 처리해주기
    dataset, dataset_y, dataset_back, cutted_hist = zip(*results)
    dataset = np.array(dataset)
    dataset_y = np.array(dataset_y)
    dataset_back = np.array(dataset_back)
    cutted_hist = np.array(cutted_hist)

    date = "240923"
    filename = f"{date}_{time_ran[0]}to{time_ran[1]}sec_{len(source)}source_{total_set}"
    np.savez(f"./0.dataset/{filename}.npz", x_ori=dataset, y=dataset_y, x_back=dataset_back, x_cutted=cutted_hist)
# np.save(f"./0.dataset/{filename}.npy", dataset)

# # y는 이미 처리되어 있으니까 1단계로 넘김.
# np.save(f"./1.preprocess_data/{filename}_y.npy", dataset_y)





#%%


#%% ====================================================================================
# 2. data 자체가 multi source 에서 온 경우.
# ======================================================================================
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "lib"))
from modi_hist_extract import modi_hist_extract
import itertools
import random
import numpy as np
from joblib import Parallel, delayed, cpu_count

source = np.array(['ba', 'cs', 'na', 'background', 'co57', 'th', 'ra', 'am'])

# (0) init
total_set = 10000  #! <----------------------이부분 수정해서 이용 !!
index = [i for i in range(len(source))]   # source list에서 뽑을 index

datadir = '../../Data/integrated_modified/'
data_date = ['240627_t700']
max_hist_count = 10000

back_cutting = 0  # background 를 spectrum 에서 단순 제거를 할지 flag

time_ranges = [[10,20]]#,[20,30],[30,40],[40,50],[50,60]]

# (1) start
def work_1(ii, time_range):

    y = np.zeros([1, len(source)])

    # 우선 어느 날짜 data 로 뽑을지 정하기.
    exp_data = random.choice(data_date)
    dt_path = os.path.join(datadir,exp_data)
    dt_path_source = os.listdir(dt_path)
    dt_path_source.remove('background')


    # 그 중 하나뽑기 + y(정답)도 지정
    final_select = random.choice(dt_path_source)
    for i in final_select.split('.'):   # . 으로 구분된 source 들을 flag on
        idx = np.where(source == i)[0][0]
        y[0, idx] = 1

    # (2) 특정 시간 구간에 대한 구현. 
    total_time = 0  # 5초 이내로 작동
    while total_time < time_range[0] or total_time > time_range[1]:  #! <----------------------이부분 수정해서 이용 !!
        starttime = random.choice(range(0, 300)) + random.random()
        endtime = random.choice(range(int(starttime), 301)) + random.random()
        total_time = endtime - starttime

    # (3) data 추출
    # 최종적으로 여러가지가 합쳐진 spectrum 초기화
    
    # background hist 계산
    its_background = np.zeros([1,1000])

    csv_back_path = os.path.join(dt_path, 'background', exp_data)
    csv_back_file = f"{csv_back_path}.csv"
    fil_back_dt = modi_hist_extract(csv_back_file)
    fil_back_dt.filtered_counts(max_hist_count, starttime, endtime)
    its_background[0] = fil_back_dt.filtered_hist

    # 고른 source 부분들 합치기
    final_hist = np.zeros([1,1000])
    csv_path = os.path.join(dt_path, final_select)
    csv_file_list = os.listdir(csv_path)
    csv_file_sel = random.choice(csv_file_list)
    csv_file_name = os.path.join(csv_path, csv_file_sel)
    csv_file = f"{csv_file_name}"
    fil_dt = modi_hist_extract(csv_file)  # filtered data

    fil_dt.filtered_counts(max_hist_count, starttime, endtime)
    final_hist[0] += fil_dt.filtered_hist

 
    cutted_hist = np.zeros([1, 1000])
    if back_cutting:
        cutted_hist[0] = final_hist[0] - (its_background[0] * len(final_select))
        cutted_hist[0][cutted_hist[0]<0] = 0 # 음수로 넘어가버리면 안되니 0으로 정정
        
    return final_hist, y, its_background, cutted_hist




for time_ran in time_ranges:
    try:
        del dataset, dataset_y, dataset_back, cutted_hist, filename
    except:
        pass
    
    results = Parallel(n_jobs=cpu_count(), verbose=10)(delayed(work_1)(task, time_ran) for task in range(total_set))

    # 결과 받고 numpy로 처리해주기
    dataset, dataset_y, dataset_back, cutted_hist = zip(*results)
    dataset = np.array(dataset)
    dataset_y = np.array(dataset_y)
    dataset_back = np.array(dataset_back)
    cutted_hist = np.array(cutted_hist)

    date = "240923_multiple240627"
    filename = f"{date}_{time_ran[0]}to{time_ran[1]}sec_{len(source)}source_{total_set}"
    np.savez(f"./0.dataset/{filename}.npz", x_ori=dataset, y=dataset_y, x_back=dataset_back, x_cutted=cutted_hist)
# np.save(f"./0.dataset/{filename}.npy", dataset)

# # y는 이미 처리되어 있으니까 1단계로 넘김.
# np.save(f"./1.preprocess_data/{filename}_y.npy", dataset_y)


















#%% ==================================================================================
# xz data 에서 추출하기.
# =====================================================================================
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

dt_path = f"../../Data/230000_forTensorflow_xz/"

source = np.array(['ba', 'cs', 'na', 'background', 'co57', 'th', 'ra', 'am'])
xz_ = np.array(['Ba133', 'Cs137', 'Na22', 'Background'])

# (0) init
total_set = 10000  #! <----------------------이부분 수정해서 이용 !!
index = [0, 1, 2]   # source list에서 뽑을 index
time_range = [5, 10]
# <-------------------- 개인 변경

xz_back_path = os.path.join(dt_path, 'background')
xz_back_files = os.listdir(xz_back_path)

idx_files = []

for i in xz_back_files:
    data_num = i.split('_')[1][:-3]
    idx_files.append(int(data_num))
idx_files.sort()
print(max(idx_files))

back_cutting = 1  # background 를 spectrum 에서 단순 제거를 할지 flag


time_ranges_xz = [[10,20],[20,30],[30,40]]
#%%
def xz_hist_make(ii, time_ran):
    # (1) source 뭐 뽑을지 선택
    # 우선 어떤 source가 섞일 것인지 정해놓고 가야할듯.
    howmany = random.choice([1,2,3])  # 한가지 or 두가지 선원 뽑을건지 선택
    combi = list(itertools.combinations(index, howmany))
    
    final_select = random.choice(combi)   
    
    # final select를 통해 y도 같이 생성
    y = np.zeros([1, len(source)])
    for i in final_select:
        y[0, i] = 1
        
    
    # (2) 특정 시간 구간에 대한 구현. 
    dur_time = random.choice(range(time_ran[0], time_ran[1]))
    start_time = random.choice(range(0, max(idx_files)-dur_time)) # index 최대값 넘지않게, time duration 만큼 시작 시간 선택
    total_time = [start_time+i for i in range(0,dur_time)]
    
    # final_hist 초기화
    its_background = np.zeros([1, 1000])
    xz_back_path = os.path.join(dt_path, 'background')
    for time in total_time:
        xz_back_file = os.path.join(xz_back_path, f"Background_{time}.xz")
        with lzma.open(xz_back_file, 'rb') as f:
            temp = pickle.load(f)
        its_background[0, :] += temp.reshape(1,1000)[0,:]  
    
    
    # source 조합에 맞춰 반복해서 spectrum 더해주기
    final_hist = np.zeros([1, 1000])
    for idx in final_select:
        RA = source[idx]
        for time in total_time:
            xzfile_path = os.path.join(dt_path, RA, f'{xz_[idx]}_{time}.xz')    
            with lzma.open(xzfile_path, 'rb') as f:
                temp = pickle.load(f)
                final_hist[0, :] += temp.reshape(1,1000)[0,:]

    cutted_hist = np.zeros([1, 1000])
    if back_cutting:
        cutted_hist[0] = final_hist[0] - (its_background[0] * len(final_select))
        cutted_hist[0][cutted_hist[0]<0] = 0 # 음수로 넘어가버리면 안되니 0으로 정정
    
    return final_hist, y, its_background, cutted_hist


for time_ran in time_ranges_xz:
    
    try:
        del dataset, dataset_y, dataset_back, cutted_hist, filename
    except:
        pass
    results = Parallel(n_jobs=cpu_count(), verbose=10)(delayed(xz_hist_make)(task,time_ran) for task in range(total_set))

# 결과 받고 numpy로 처리해주기
    dataset, dataset_y, dataset_back, cutted_hist = zip(*results)
    dataset = np.array(dataset)
    dataset_y = np.array(dataset_y)
    dataset_back = np.array(dataset_back)
    cutted_hist = np.array(cutted_hist)

    #save generated data =========================================================
    date = "240923"
    filename = f"{date}_{time_ran[0]}to{time_ran[1]}_{len(index)}source_{total_set}_xzfile"
    np.savez(f"./0.dataset/{filename}.npz", x_ori=dataset, y=dataset_y, x_back=dataset_back, x_cutted=cutted_hist)
# np.save(f"./0.dataset/{filename}.npy", dataset)















" ======================  End Line ======================= "





