'''
검출기에서 얻은 csv 데이터파일을 numpy로 전환
energy 위치 안 맞는 부분을 보정해준 후,
조건을 지정하여 그 부분의 data만 저장.
'''


#%% init
class csv_hist_modify():
    def __init__(self, csvfile:str):
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        
        self.dt = pd.read_csv(csvfile)
        self.col = list(self.dt.keys())  # column 저장하기
        self.addcol = ["ene", "modi_ene"]  # 추가적으로 필요한 column은 여기서 추가하기.
        
        # ene 추가
        self.dt[self.addcol[0]] = round(self.dt[self.col[1]]/1000) # keV로 전환
        n, bins, _ = plt.hist(self.dt[self.addcol[0]],
                              bins=int(self.dt[self.addcol[0]].max()),
                              color='red', 
                              range=(0, self.dt[self.addcol[0]].max()))
        plt.xlim(0,1001)  # 보여지는 범위 고정
        self.histvalues = n   # 실제 histogram 그리는 값들
        self.histvalues_bin = bins  # 값이 존재하는 idx 부분 저장
    
    # 특정 구간에서 실제 이 source의 peak라고 예상되는 부분의 구간에서 peak idx 추출    
    def find_peak_idx(self, range_:range)->int:
        max_val = self.histvalues[range_].max() # 구간의 peak 값 확인
        idx = 0
        for i in range(len(self.histvalues)):
            if self.histvalues[i] == max_val:
                idx = i
        self.peakidx = idx
        print("peak location updated to : ", idx)
        return idx
    
    # 처음 spectrum show
    def plot_ori_spec(self):
        import matplotlib.pyplot as plt
        # self.col[1] 이 total_energy
        plt.hist(self.dt[self.addcol[0]], 
                 bins=int(self.dt[self.addcol[0]].max()), 
                 color='red', 
                 range=(0, self.dt[self.addcol[0]].max()))
    
    # peak idx 를 통해 spectrum 위치보정
    def modi_spec(self, realpeak:int)->list:
        import matplotlib.pyplot as plt
        # real peak 실제 선원의 peak 위치(값)
        coef = realpeak / self.peakidx  # 전반적으로 옮겨주기 위한 곱할 값
        self.coef = coef  
        #self.coef = 1.7 # 애매한 histogram이면 약 얼추 1.7~1.9 정도로 곱해주면 되는듯
        self.dt[self.addcol[1]] = round(self.dt[self.addcol[0]]*self.coef)
        n, bins, _ = plt.hist(self.dt[self.addcol[1]],
                              bins=int(self.dt[self.addcol[1]].max()), # bin을 max로 해야, max값까지 하나씩 잘 쪼개냄.
                              color='purple', 
                              range=(0, self.dt[self.addcol[1]].max()))
        self.modi_hist = n[0:1001] 
        
    # 변경된 spectrum show
    def modi_spec_show(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(self.modi_hist)
        ax.set_xlim(0,1001)
        
        
    # 무슨 method 있는지
    def show(self):
        print("show : method show")
        print("plot_ori_spec : plotting original spectrum")
        print("modi_spec : modify and plot spectrum")



#%%
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "lib"))
print(sys.path)

from modi_hist_extract import modi_hist_extract


#%% ================================================= 
# make data 
# ===================================================

#%% 2. data 추출을 위해, original data(csv) 파일을 새롭게 저장 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
import os
# 하드코딩으로 스펙트럼 위치 보정 구현

src_list = ['ba133', 'cs137', 'na22', 'background', 'co57', 'th232', 'ra226']
distance_list = ["close", "35cm", "67cm", "103cm"]

src = src_list[0]  #! -> 바꿔서 사용!
distance = distance_list[0]  #! -> 바꿔서 사용!
direc = "../../Data/240603/ori"
filename = f"{src}_{distance}_5min.csv"

# csv 있는 위치
datadir = os.path.join(direc, filename)
#* datadir = os.path.join(direc, "background_indoor.csv")  # background data
test = csv_hist_modify(datadir)
#%% process
real_peak = [356, 662, 511, '-', 122, 238, 351]

test.find_peak_idx(range(180,250)) #! -> 바꿔서 사용!
test.modi_spec(real_peak[0])  #! -> 바꿔서 사용!
test.modi_spec_show()

#%% extract and save data
cols = [" time_stamp (sec)", "ene", "modi_ene"]
new_dt = test.dt[cols]

new_dt.to_csv(f"../../Data/240603/modi/{src}_{distance}.csv")
#*new_dt.to_csv(f"../../Data/240603/modi/background_indoor.csv") # background save















#%% 3. source 선택  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
import itertools

# mix single set version
mix_version = 4

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
    
elif mix_version == 4:
    #source = ['Ba133', 'Cs137', 'Na22', 'Am241', 'Co60', 'Ga67', 'I131', 'Ra226', 'Tc99m', 'Th232', 'Tl201', 'background']
    source = ['ba133', 'cs137', 'na22', 'background', 'co57', 'th232', 'ra226']
    # am241, co60, CRM 빠짐






#%% 4. modi csv를 통해 정해진 규칙을 기준으로 data filter 후 train or test data로 생성  

import itertools
import random
import numpy as np
from joblib import Parallel, delayed, cpu_count

# (0) init
total_set = 20000  #! <----------------------이부분 수정해서 이용 !!
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
        csv_file = f"../../Data/240603/modi/{source_}_{distance}.csv"
        fil_dt = modi_hist_extract(csv_file)  # filtered data
        fil_dt.filtered(starttime, endtime)
        final_hist[0] += fil_dt.histvalues[0:1000]
        
    #dataset[ii,:,:] = final_hist
    return final_hist, y

results = Parallel(n_jobs=cpu_count(), verbose=10)(delayed(work_1)(task) for task in range(total_set))

# 결과 받고 numpy로 처리해주기
dataset, dataset_y = zip(*results)
dataset = np.array(dataset)
dataset_y = np.array(dataset_y)




# %% 5. save generated data
date = "240624"
filename = f"{date}_{time_range[0]}to{time_range[1]}sec_{len(index)}source_{total_set}"
np.save(f"./0.dataset/{filename}.npy", dataset)

# y는 이미 처리되어 있으니까 1단계로 넘김.
np.save(f"./1.preprocess_data/{filename}_y.npy", dataset_y)


# %%
