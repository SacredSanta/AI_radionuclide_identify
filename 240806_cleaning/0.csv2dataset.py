'''
최종 수정 : 2024.08.05.
사용자 : 서동휘

<수정 내용> 

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


'''
<<< chapter 1 >>>
''' 
#%% 0. modi 된 csv 저장용
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "lib"))
print(sys.path)

from modi_hist_extract import modi_hist_extract

# 하드코딩으로 스펙트럼 위치 보정 구현

src_list = ['ba133', 'cs137', 'na22', 'background', 'co57', 'th232', 'ra226']
distance_list = ["close", "35cm", "67cm", "103cm"]
real_peak = [356, 662, 511, '-', 122, 238, 352, 60]

src = "background"  #! -> 바꿔서 사용!
distance = distance_list[0]  #! -> 바꿔서 사용!
data_title = "240603_nucare"
direc = "../../Data/" #! -> 바꿔서 사용!
thres = "500"
#filename = f"{data_title}/ori/thres{thres}/{src}_{distance}_5min.csv" #! -> 바꿔서 사용!
filename = f"{data_title}/ori/{src}_{distance}_5min.csv"

# csv 있는 위치
datadir = os.path.join(direc, filename)
print(datadir)


#%% 1. make histogram ------------------------------------------------------------
#* datadir = os.path.join(direc, "background_indoor.csv")  # background data
test = modi_hist_extract(datadir)

import matplotlib.pyplot as plt

plt.plot(test.hist)
x_ticks = [i*50 for i in range(0,20)]
x_labels = [50*i for i in range(0,20)]
plt.xticks(ticks=x_ticks, labels=x_labels, fontsize=5)

#%% scale data -------------------------------------------------------------------------------

# 만약 peak 위치 찾기가 어렵다면 scale 값 고정해서 변경
fixed = 1

if fixed:
    test.scale_factor = 1.8  #! -> 바꿔서 사용!
    test.fix_data_with_scalefactor(test.scale_factor)

else:
    test.find_peak([100, 200]) #! -> 바꿔서 사용!
    test.fix_data(real_peak[5])  #! -> 바꿔서 사용!
    
test.show()


#%% extract and save data -------------------------------------------------------------------------------
cols = ["event_numbers", " total_energy (keV)",  " time_stamp (sec)", "ene"]
new_dt = test.dt[cols]

save_dir = f"../../Data/240603_nucare/newmodi/"
save_filename = f"{src}_{distance}_5min.csv"

new_dt.to_csv(save_dir + save_filename)

#*new_dt.to_csv(f"../../Data/240603/modi/background_indoor.csv") # background save

if fixed: fixedmsg="fixed"
else: fixedmsg=''
with open(save_dir + "coef.txt", "a") as f:
    f.write("\n"+save_filename+" : "+str({round(test.scale_factor,2)})+fixedmsg)

















'''
<<< chapter 2 >>>
'''
#%% 3. source 선택  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
import itertools

# mix single set version
mix_version = 5

# version 1 : Ba,Na,Cs 3가지
if mix_version == 1:
    mixing_list = {'Ba':['Ba133'],
                'Na':['Na22'],
                'Cs':['Cs137'],
                'BaCs':['Ba133','Cs137'],
                'BaNa':['Ba133','Na22'], 
                'CsNa':['Cs137','Na22'],
                'BaNaCs':['Ba133','Na22','Cs137'],
                'BG':['Background']
            }

    mixed_source_name = list(mixing_list.keys())
    mixed_list = list(mixing_list.values())

# version 2 : 11가지   
# Ba133, Cs137, Am241, Co60, Ga67, I131, Ra226, Tc99m, Th232, Tl201, Na22 
# 0      1      2      3     4     5     6      7      8      9      10     
elif mix_version == 2:
    source_seed = [i for i in range(0,11)]
    src_combi = []
    max_combi_len = 3 #len(source_seed)+1

    for i in range(1, max_combi_len+1):
        for combi in itertools.combinations(source_seed, i):
            src_combi.append(list(combi))

    print(src_combi)
    
# version 3: 11가지
# 내가 쓰기 쉽게 조금 다르게 변형
elif mix_version == 3:
    source = ['ba133', 'cs137', 'na22', 'background']
    
# nucare 전용 | source label 이 7개    
elif mix_version == 4:
    #source = ['Ba133', 'Cs137', 'Na22', 'Am241', 'Co60', 'Ga67', 'I131', 'Ra226', 'Tc99m', 'Th232', 'Tl201', 'background']
    source = ['ba133', 'cs137', 'na22', 'background', 'co57', 'th232', 'ra226']
    # am241, co60, CRM 빠짐

# xz data 전용 | source label 이 7이어야 할때
elif mix_version == 5:
    source = ['Ba133', 'Cs137', 'Na22', 'Background', '_', '_', '_']




#%% 4. modi csv를 통해 정해진 규칙을 기준으로 data filter 후 train or test data로 생성  import sys
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "lib"))
print(sys.path)

from modi_hist_extract import modi_hist_extract


import itertools
import random
import numpy as np
from joblib import Parallel, delayed, cpu_count

# (0) init
total_set = 5000  #! <----------------------이부분 수정해서 이용 !!
index = [i for i in range(len(source))]   # source list에서 뽑을 index
time_range = [30, 300] # <-------------------- 개인 변경

def work_1(ii):
    #if ii % 100 == 0: print(ii, " data proceed..")
    
    # (1) source 뭐 뽑을지 선택
    # 우선 어떤 source가 섞일 것인지 정해놓고 가야할듯.
    howmany = random.choice([1,2])  # 한가지 or 두가지 선원 뽑을건지 선택
    combi = list(itertools.combinations(index, howmany))
    final_select = random.choice(combi)   # -> y    ex : (0, 2),  뽑힌 source들
    # final select를 통해 y도 같이 생성
    y = np.zeros([1,len(index)])
    for i in final_select:
        y[0, i] = 1
    #dataset_y.append(y)

    # (2) 특정 시간 구간에 대한 구현. 
    total_time = 0
    while total_time < time_range[0] or total_time > time_range[1]:  #! <----------------------이부분 수정해서 이용 !!
        starttime = random.choice(range(0, 300)) + random.random()
        endtime = random.choice(range(int(starttime), 301)) + random.random()
        total_time = endtime - starttime

    # (3) data 추출
    # 최종적으로 여러가지가 합쳐진 spectrum 초기화
    final_hist = np.zeros([1,1000])

    for idx in final_select:  # 뽑힌 source idx에서 반복
        source_ = source[idx]
        distance = random.choice(['close', '35cm'])
        csv_file = f"../../Data/240603_nucare/newmodi/{source_}_{distance}_5min.csv"
        fil_dt = modi_hist_extract(csv_file)  # filtered data
        fil_dt.filtered(starttime, endtime)
        final_hist[0] += fil_dt.filtered_hist
        
    #dataset[ii,:,:] = final_hist
    return final_hist, y

results = Parallel(n_jobs=cpu_count(), verbose=10)(delayed(work_1)(task) for task in range(total_set))

# 결과 받고 numpy로 처리해주기
dataset, dataset_y = zip(*results)
dataset = np.array(dataset)
dataset_y = np.array(dataset_y)




# %% 5. save generated data
date = "240806"
filename = f"{date}_{time_range[0]}to{time_range[1]}sec_{len(index)}source_{total_set}"
np.save(f"./0.dataset/{filename}.npy", dataset)

# y는 이미 처리되어 있으니까 1단계로 넘김.
np.save(f"./1.preprocess_data/{filename}_y.npy", dataset_y)

















'''
<<< chapter 2-2 >>>
'''

#%% xz data 에서 추출하기.
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

dtpath = f"../../Data/230000_forTensorflow_xz/"


# (0) init
total_set = 3000  #! <----------------------이부분 수정해서 이용 !!
index = [0, 1, 2, 3]   # source list에서 뽑을 index
# <-------------------- 개인 변경

def work_1(ii):
    #if ii % 100 == 0: print(ii, " data proceed..")
    
    # (1) source 뭐 뽑을지 선택
    # 우선 어떤 source가 섞일 것인지 정해놓고 가야할듯.
    howmany = random.choice([1,2])  # 한가지 or 두가지 선원 뽑을건지 선택
    combi = list(itertools.combinations(index, howmany))
    
    # -> y    ex : (0, 2),  뽑힌 source들
    final_select = random.choice(combi)   
    
    # final select를 통해 y도 같이 생성
    y = np.zeros([1, len(source)])
    for i in final_select:
        y[0, i] = 1

   
    # final_hist 초기화
    final_hist = np.zeros([1, 1000])
    
    # source 조합에 맞춰 반복해서 spectrum 더해주기
    for idx in final_select:
        RA = source[idx]
        num = np.random.randint(1, 600, size=1)
        xzfile_path = dtpath+"/{}/{}_{}.xz".format(RA, RA, num[0])
        
        # num 이 없는 파일 지정할 수도 있어서..
        while not os.path.isfile(xzfile_path):
            num = np.random.randint(1, 600, size=1)
            xzfile_path = dtpath+"/{}/{}_{}.xz".format(RA, RA, num[0])
            
        with lzma.open(xzfile_path, 'rb') as f:
            temp = pickle.load(f)
            
        final_hist += temp.reshape(1,1000)[0,:]

    return final_hist, y

results = Parallel(n_jobs=cpu_count(), verbose=10)(delayed(work_1)(task) for task in range(total_set))

# 결과 받고 numpy로 처리해주기
dataset, dataset_y = zip(*results)
dataset = np.array(dataset)
dataset_y = np.array(dataset_y)


# %% save generated data =========================================================
date = "240806"
filename = f"{date}_{len(index)}source_{total_set}_xzfile"
np.save(f"./0.dataset/{filename}.npy", dataset)

# y는 이미 처리되어 있으니까 1단계로 넘김.
np.save(f"./1.preprocess_data/{filename}_y.npy", dataset_y)




" ======================  End Of Line ======================= "