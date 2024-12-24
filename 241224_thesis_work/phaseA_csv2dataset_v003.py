#%%
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

#%% 1. modi csv를 통해 정해진 규칙을 기준으로 data filter 후 train or test data로 생성  import sys [dataset 2]
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
total_set = 5000  #! <----------------------이부분 수정해서 이용 !!
index = [i for i in range(len(source))]   # source list에서 뽑을 index
#time_range = [10, 20] # <-------------------- 개인 변경

datadir = '../../Data/integrated_modified/'
data_date = ['240603', '240906']  #'240905' , '240627_t700'

max_combi = 3 # combination 종류 최대 개수

back_cutting = 1  # background 를 spectrum 에서 단순 제거를 할지 flag

time_ranges = [[1,300]]#,[10,20],[20,30],[30,40],[40,50],[50,60]]
min_max_set = [[1000,2000], [3000,4000], [5000,6000], [7000,8000], [9000,10000],
                [11000,12000], [13000,14000], [15000,16000]]
#min_max_set = [[7000,8000]]
# min_count = 10000
# max_count = 11000

# (1) start
def work_1(ii, time_range, min_count, max_count):

    y = np.zeros([1, len(source)])

    # 우선 어느 날짜 data 로 뽑을지 정하기.
    exp_data = random.choice(data_date)
    dt_path = os.path.join(datadir,exp_data)
    dt_path_source = os.listdir(dt_path)
    dt_path_source.remove('background')
    # if exp_data == '240603':
    #     dt_path_source.remove('co57')
    #     dt_path_source.remove('ra')
    
    # 개수가 최대 3개를 넘어가지 않게 하기
    if len(dt_path_source) < max_combi + 1:
        howmany = random.choice(range(1, len(dt_path_source)+1))
    else:
        howmany = random.choice(range(1, max_combi + 1))

    # 개수만큼의 조합을 구하고, 그 중 하나뽑기 + y(정답)도 지정
    combi = list(itertools.combinations(dt_path_source, howmany))
    final_select = random.choice(combi)
    for i in final_select:
        idx = np.where(source == i)[0][0]
        y[0, idx] = 1

    # (2) 특정 시간 구간에 대한 구현. 
    '''
    total_time = 0  # 5초 이내로 작동
    while total_time < time_range[0] or total_time > time_range[1]:  #! <----------------------이부분 수정해서 이용 !!
        starttime = random.choice(range(0, 300)) + random.random()
        endtime = random.choice(range(int(starttime), 301)) + random.random()
        total_time = endtime - starttime
    '''
    # (3) data 추출
    # 최종적으로 여러가지가 합쳐진 spectrum 초기화
    
    # background hist 계산 - 1.preprocessing 에서 전체 background 합산하기 때문에 생략
    '''
    # its_background = np.zeros([1,1000])
    
    # max_hist_count = random.choice(range(200,1000))
    
    # csv_back_path = os.path.join(dt_path, 'background', exp_data)
    # csv_back_file = f"{csv_back_path}.csv"
    # fil_back_dt = modi_hist_extract(csv_back_file)
    # fil_back_dt.filtered_counts(max_hist_count, starttime, endtime)
    # its_background[0] = fil_back_dt.filtered_hist
    '''


    # 고른 source 부분들 합치기
    final_hist = np.zeros([1,1000])
    for src in final_select:  # 뽑힌 source idx에서 반복
        max_hist_count = int(random.choice(range(min_count, max_count)) / len(final_select))
        csv_path = os.path.join(dt_path, src)
        csv_file_list = os.listdir(csv_path)
        csv_file_sel = random.choice(csv_file_list)
        csv_file_name = os.path.join(csv_path, csv_file_sel)
        csv_file = f"{csv_file_name}"
        fil_dt = modi_hist_extract(csv_file)  # filtered data
        fil_dt.filtered_counts(max_hist_count, time_range[0], time_range[1])
        final_hist[0] += fil_dt.filtered_hist
        
    # source 에 맞는 background 를 커팅해야할지는 다시 생각해봐야함.
    '''
    cutted_hist = np.zeros([1, 1000])
    if back_cutting:
        cutted_hist[0] = final_hist[0] - (its_background[0] * len(final_select))
        cutted_hist[0][cutted_hist[0]<0] = 0 # 음수로 넘어가버리면 안되니 0으로 정정
    '''      
    return final_hist, y #, its_background, cutted_hist

for min_max in min_max_set:
    try:
        del dataset, dataset_y, filename, #dataset_back, cutted_hist, 
    except:
        pass
    min_count = min_max[0]
    max_count = min_max[1]
    
    results = Parallel(n_jobs=cpu_count(), verbose=10)(delayed(work_1)(task, time_ranges[0], min_count, max_count) for task in range(total_set))

    # 결과 받고 numpy로 처리해주기
    dataset, dataset_y = zip(*results) #dataset_back, cutted_hist = zip(*results)
    dataset = np.array(dataset)
    dataset_y = np.array(dataset_y)
    #dataset_back = np.array(dataset_back)
    #cutted_hist = np.array(cutted_hist)

    date = "241216"
    filename = f"{date}_dataset2_set{total_set}_min{min_count}_max{max_count}_combi{max_combi}"
    np.savez(f"./0.dataset/{filename}.npz", x_ori=dataset, y=dataset_y)#, x_back=dataset_back, x_cutted=cutted_hist)
# np.save(f"./0.dataset/{filename}.npy", dataset)

# # y는 이미 처리되어 있으니까 1단계로 넘김.
# np.save(f"./1.preprocess_data/{filename}_y.npy", dataset_y)






#%% ====================================================================================
# 2. data 자체가 multi source 에서 온 경우. [dataset 3]
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
total_set = 5000  #! <----------------------이부분 수정해서 이용 !!
index = [i for i in range(len(source))]   # source list에서 뽑을 index

datadir = '../../Data/integrated_modified/'
data_date = ['240627_t700']

back_cutting = 0  # background 를 spectrum 에서 단순 제거를 할지 flag

time_ranges = [[1,250]]#,[20,30],[30,40],[40,50],[50,60]]
min_max_set = [[6000,7000], [7000,8000], [8000,9000], [9000,10000], [10000,11000],
               [11000,12000], [12000,13000], [13000,14000], [14000,15000]]

# (1) start
def work_1(ii, time_range, min_count, max_count):

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
    '''
    total_time = 0  # 5초 이내로 작동
    while total_time < time_range[0] or total_time > time_range[1]:  #! <----------------------이부분 수정해서 이용 !!
        starttime = random.choice(range(0, 300)) + random.random()
        endtime = random.choice(range(int(starttime), 301)) + random.random()
        total_time = endtime - starttime
    '''
    
    # (3) data 추출
    # 최종적으로 여러가지가 합쳐진 spectrum 초기화
    
    # background hist 계산
    '''
    its_background = np.zeros([1,1000])

    max_hist_count = random.choice(range(500,1500))

    csv_back_path = os.path.join(dt_path, 'background', exp_data)
    csv_back_file = f"{csv_back_path}.csv"
    fil_back_dt = modi_hist_extract(csv_back_file)
    fil_back_dt.filtered_counts(max_hist_count, starttime, endtime)
    its_background[0] = fil_back_dt.filtered_hist
    '''
    
    # 고른 source 부분들 합치기
    final_hist = np.zeros([1,1000])
    max_hist_count = int(random.choice(range(min_count, max_count)))
    csv_path = os.path.join(dt_path, final_select)
    csv_file_list = os.listdir(csv_path)
    csv_file_sel = random.choice(csv_file_list)
    csv_file_name = os.path.join(csv_path, csv_file_sel)
    csv_file = f"{csv_file_name}"
    fil_dt = modi_hist_extract(csv_file)  # filtered data
    fil_dt.filtered_counts(max_hist_count, time_range[0], time_range[1])
    final_hist[0] += fil_dt.filtered_hist

    '''
    cutted_hist = np.zeros([1, 1000])
    if back_cutting:
        cutted_hist[0] = final_hist[0] - (its_background[0] * len(final_select))
        cutted_hist[0][cutted_hist[0]<0] = 0 # 음수로 넘어가버리면 안되니 0으로 정정
    '''    
    return final_hist, y #, its_background, cutted_hist

for min_max in min_max_set:
    try:
        del dataset, dataset_y, filename, #dataset_back, cutted_hist, 
    except:
        pass
    min_count = min_max[0]
    max_count = min_max[1]
    
    results = Parallel(n_jobs=cpu_count(), verbose=10)(delayed(work_1)(task, time_ranges[0], min_count, max_count) for task in range(total_set))

    # 결과 받고 numpy로 처리해주기
    dataset, dataset_y = zip(*results)   # dataset_back, cutted_hist
    dataset = np.array(dataset)
    dataset_y = np.array(dataset_y)
    #dataset_back = np.array(dataset_back)
    #cutted_hist = np.array(cutted_hist)

    date = "241204_multiple240627"
    filename = f"{date}_dataset3_set{total_set}_min{min_count}_max{max_count}"
    np.savez(f"./0.dataset/{filename}.npz", x_ori=dataset, y=dataset_y) #, x_back=dataset_back, x_cutted=cutted_hist)
# np.save(f"./0.dataset/{filename}.npy", dataset)

# # y는 이미 처리되어 있으니까 1단계로 넘김.
# np.save(f"./1.preprocess_data/{filename}_y.npy", dataset_y)


















#%% ==================================================================================
# xz data 에서 추출하기. [dataset 1]
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
total_set = 5000  #! <----------------------이부분 수정해서 이용 !!
index = [0, 1, 2]   # source list에서 뽑을 index
# <-------------------- 개인 변경

xz_back_path = os.path.join(dt_path, 'background')
xz_back_files = os.listdir(xz_back_path)

idx_files = []

for i in xz_back_files:
    data_num = i.split('_')[1][:-3]
    idx_files.append(int(data_num))
idx_files.sort()
print("background 파일 개수 : ", max(idx_files))

back_cutting = 0  # background 를 spectrum 에서 단순 제거를 할지 flag

# min_max_set = [[6000,7000], [7000,8000], [8000,9000], [9000,10000], [10000,11000],
#                [11000,12000], [12000,13000], [13000,14000], [14000,15000]]

min_max_set = [[15000,16000]]

time_ranges_xz = [[1,15]]#,[10,20],[20,30],[30,40]]


#%% -------------------------------------------------------------------------------------------------------------------
def xz_hist_make(ii, time_ran, min_count, max_count):
    # (1) source 뭐 뽑을지 선택
    # 우선 어떤 source가 섞일 것인지 정해놓고 가야할듯.
    howmany = random.choice([1,2,3])  # 한가지 or 두가지 선원 뽑을건지 선택
    combi = list(itertools.combinations(index, howmany))
    
    final_select = random.choice(combi)   
    
    # final select를 통해 y도 같이 생성
    y = np.zeros([1, len(source)])
    for i in final_select:
        y[0, i] = 1
    
    final_hist = np.zeros([1, 1000])
    
    while sum(final_hist[0]) < min_count or sum(final_hist[0]) > max_count:
    # (2) 특정 시간 구간에 대한 구현. 
        dur_time = random.choice(range(time_ran[0], time_ran[1])) #! <--------- 이거 수정필요
        start_time = random.choice(range(0, max(idx_files)-dur_time)) # index 최대값 넘지않게, time duration 만큼 시작 시간 선택
        total_time = [start_time+i for i in range(0,dur_time)] # xz 파일은 초마다 파일 넘버로 저장되어 있기 때문.
        
        # final_hist의 background 부분 초기화
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
        #print("Debug : ", sum(final_hist[0]))

    cutted_hist = np.zeros([1, 1000])
    if back_cutting:
        cutted_hist[0] = final_hist[0] - (its_background[0] * len(final_select))
        cutted_hist[0][cutted_hist[0]<0] = 0 # 음수로 넘어가버리면 안되니 0으로 정정
    
    return final_hist, y #, its_background, cutted_hist

#%% -------------------------------------------------------------------------------------------------------------------
for min_max in min_max_set:
    min_count = min_max[0]
    max_count = min_max[1]
    try:
        del dataset, dataset_y, filename # dataset_back, cutted_hist, 
    except:
        pass
    
    results = Parallel(n_jobs=cpu_count(), verbose=10)(delayed(xz_hist_make)(task, time_ranges_xz[0], min_count, max_count) for task in range(total_set))

# 결과 받고 numpy로 처리해주기
    dataset, dataset_y = zip(*results)  # dataset_back, cutted_hist
    dataset = np.array(dataset)
    dataset_y = np.array(dataset_y)
    # dataset_back = np.array(dataset_back)
    # cutted_hist = np.array(cutted_hist)

    ddt = dataset[:,0,:]
    ddt_sum = np.sum(ddt,axis=1)
    
    print(ddt_sum)
    #save generated data =========================================================
    date = "241211"
    filename = f"{date}_dataset1_set{total_set}_min{int(min(ddt_sum))}_max{int(max(ddt_sum))}_xzfile"
    np.savez(f"./0.dataset/{filename}.npz", x_ori=dataset, y=dataset_y) #x_back=dataset_back, x_cutted=cutted_hist)
# np.save(f"./0.dataset/{filename}.npy", dataset)













#%% ==================================================================================
# xz data 에서 추출하기. [dataset 1] - for low count @@@@@
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
total_set = 5000  #! <----------------------이부분 수정해서 이용 !!
index = [0, 1, 2]   # source list에서 뽑을 index
# <-------------------- 개인 변경

xz_back_path = os.path.join(dt_path, 'background')
xz_back_files = os.listdir(xz_back_path)

idx_files = []

for i in xz_back_files:
    data_num = i.split('_')[1][:-3]
    idx_files.append(int(data_num))
idx_files.sort()
print("파일 개수 : ", max(idx_files))
#
back_cutting = 0  # background 를 spectrum 에서 단순 제거를 할지 flag
#
min_count = 500
max_count = 1000

#%%
def xz_hist_make_lowcount(ii):
    # (1) source 뭐 뽑을지 선택
    # 우선 어떤 source가 섞일 것인지 정해놓고 가야할듯.
    howmany = random.choice([1,2,3])  # 한가지 or 두가지 선원 뽑을건지 선택
    combi = list(itertools.combinations(index, howmany))
    
    final_select = random.choice(combi)   
    
    # final select를 통해 y도 같이 생성
    y = np.zeros([1, len(source)])
    for i in final_select:
        y[0, i] = 1
    
    final_hist = np.zeros([1, 1000])
    
    for idx in final_select:
        RA = source[idx]
        fname = f"{RA}_all.npz"
        all_data = np.load(f"{dt_path}/{fname}")
        count_ = int(random.choice(range(min_count, max_count))/len(final_select))
        for i in range(count_):
            e_idx = random.choices(np.arange(0,1000), weights=all_data["dt"][0])
            final_hist[0,e_idx] += 1               
    
    return final_hist, y #, its_background, cutted_hist


#%%
results = Parallel(n_jobs=cpu_count(), verbose=10)(delayed(xz_hist_make_lowcount)(task) for task in range(total_set))

# 결과 받고 numpy로 처리해주기
dataset, dataset_y = zip(*results)  # dataset_back, cutted_hist
dataset = np.array(dataset)
dataset_y = np.array(dataset_y)
# dataset_back = np.array(dataset_back)
# cutted_hist = np.array(cutted_hist)

ddt = dataset[:,0,:]
ddt_sum = np.sum(ddt,axis=1)

#save generated data =========================================================
date = "241129"
filename = f"{date}_dataset1_set{total_set}_min{int(min(ddt_sum))}_max{int(max(ddt_sum))}_xzfile"
np.savez(f"./0.dataset/{filename}.npz", x_ori=dataset, y=dataset_y) #x_back=dataset_back, x_cutted=cutted_hist)
# np.save(f"./0.dataset/{filename}.npy", dataset)







# ------------------------------------------------------------
#%% xz data 모두 합쳐버리기 (완료된 사항) 
import os
import lzma
import pickle

final_hist = np.zeros([1, 1000])

dtpath = "../../Data/230000_forTensorflow_xz/background"

for i in os.listdir(dtpath):
    xzfile_path = os.path.join(dtpath, i)
    with lzma.open(xzfile_path, 'rb') as f:
        temp = pickle.load(f)
        final_hist[0, :] += temp.reshape(1,1000)[0,:]

np.savez("../../Data/230000_forTensorflow_xz/back_all.npz", dt=final_hist)




















#%% ==================================================================================
# bin 자체에서 추출하기 [dataset 4]
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


datadir = "../../Data/230000_unknown"
#dataname = "3.5t_8mm_ba133_1M_1.bin"
dataname = "7t_8mm_ba133_100cm_5.bin"
#%%
fil_dt = modi_hist_extract(os.path.join(datadir,dataname), cali='off', isbin='yes')

plt.plot(fil_dt.hist)
#%%
'''
import struct
import os

filepath = os.path.join(datadir, dataname)

bytes_count = [8,8,4,4,4,4, 
               8,8,4,4,4,4]
bytes_type = ['<2f','<2i','<i','<i','<i','<i',
              '<2f','<2i','<i','<i','<i','<i']
cols = ['time_stamp', 'event_number', 'ch1', 'ch2', 'ch3', 'ch4']
temp_col = ['1',  '2',  '3',  '4',  '5',  '6', 
            '11','22', '33', '44', '55', '66']


def bin_data_stream_old(nn, filepath):
    with open(f"{filepath}", 'rb') as file:
        file.seek(nn*32)
        new_row = {}
        
        # 각 컬럼에 대해서 넣어주기
        for ii in range(len(temp_col)):
            val_hx = file.read(bytes_count[ii])
            val = struct.unpack(bytes_type[ii], val_hx)[0]
            new_row[temp_col[ii]] = val
    return new_row

result = bin_data_stream_old(0, filepath)
print(result)

'''


























" ======================  End Line ======================= "
#%% data의 count 상태 확인하기
ddt = dataset[:,0,:]
ddt_sum = np.sum(ddt,axis=1)
#%%
count_hist = np.zeros((int(max_count/100)))
for i in range(len(ddt_sum)):
    count_hist[int(ddt_sum[i]/100)] += 1
#%%
import matplotlib.pyplot as plt
plt.bar(range(len(count_hist)), count_hist)





# %%
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "lib"))
from modi_hist_extract import modi_hist_extract
import itertools
import random
import numpy as np
from joblib import Parallel, delayed, cpu_count

datadir = '../../Data/240603_nucare/ori'
csv_dir = os.path.join(datadir,"background_close_10min.csv")
dt = modi_hist_extract(csv_dir)

#%%
bq = (sum(dt.hist)-(1*300))/300
bq



#%%
ci = bq/(3.7*(10)**(10))
ci # co57 87931 # cs 1071

#%%
coeff = 1/(2.8*(10**(-8)) / 0.000030)
coeff
# %%
