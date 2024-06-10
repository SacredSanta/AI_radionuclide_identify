#%% init
class csv_hist_modify():
    def __init__(self, csvfile:str):
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        
        self.dt = pd.read_csv(csvfile)
        self.col = list(self.dt.keys())  # column 저장하기
        self.addcol = ["ene", "modi_ene"]  # 추가적으로 필요한 column은 여기서 추가하기.
        
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
        self.dt[self.addcol[1]] = round(self.dt[self.addcol[0]]*coef)
        n, bins, _ = plt.hist(self.dt[self.addcol[1]],
                              bins=int(self.dt[self.addcol[1]].max()),
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




# 위에서 modifed한 csv에서 시간에 따라서 추출
class modi_hist_extract():
    def __init__(self, csvfile:str):
        import pandas as pd
        import numpy as np
        
        self.dt = pd.read_csv(csvfile)
        self.col = list(self.dt.keys())  # column 저장하기
       
    # 시작시간, 끝나는 시간    
    def filtered(self, starttime, endtime):
        import matplotlib.pyplot as plt
        
        self.filtered_dt = self.dt[(self.dt[" time_stamp (sec)"] >= starttime) 
                                    & (self.dt[" time_stamp (sec)"] <= endtime)]
        
        n, bins, _ = plt.hist(self.filtered_dt["modi_ene"],
                              bins=1000,
                              color='purple', 
                              range=(0, 1001))
        plt.xlim(0,1001)  # 보여지는 범위 고정
        self.histvalues = n   # np.array - 1,1000 : 실제 histogram 그리는 값들
        self.histvalues_bin = bins  # 값이 존재하는 idx 부분 저장 (사실 필요없음)
    


#%% ================================================= 
# make data 
# ===================================================





#%% 2. source 선택  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
import itertools

# mix single set version
mix_version = 3

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
    #source = ['Ba133', 'Cs137', 'Am241', 'Co60', 'Ga67', 'I131', 'Ra226', 'Tc99m', 'Th232', 'Tl201', 'Na22']









#%% 3. data 추출을 위한 csv 파일 모두 새롭게 저장 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
import os
# 하드코딩으로 스펙트럼 위치 보정 구현

src_list = ["ba133", "cs137", "na22"]
distance_list = ["close", "35cm", "67cm", "103cm"]

src = src_list[2]  #! -> 바꿔서 사용!
distance = distance_list[3]  #! -> 바꿔서 사용!
direc = "../../Data/240603/ori"
filename = f"{src}_{distance}_5min.csv"

# csv 있는 위치
datadir = os.path.join(direc, filename)
#* datadir = os.path.join(direc, "background_indoor.csv")  # background data
test = csv_hist_modify(datadir)
#%% process
real_peak = [356, 662, 511]

test.find_peak_idx(range(220,350)) #! -> 바꿔서 사용!
test.modi_spec(real_peak[2])  #! -> 바꿔서 사용!
test.modi_spec_show()

#%% extract and save data
cols = [" time_stamp (sec)", "ene", "modi_ene"]
new_dt = test.dt[cols]

new_dt.to_csv(f"../../Data/240603/modi/{src}_{distance}.csv")
#*new_dt.to_csv(f"../../Data/240603/modi/background_indoor.csv") # background save











#%% 4. modi csv를 통해 정해진 규칙을 기준으로 data filter 후 train or test data로 생성  

total_set = 10

# 우선 어떤 source가 섞일 것인지 정해놓고 가야할듯.
import itertools
import random

howmany = random.choice([1,2])  # 한가지 or 두가지 선원 뽑을건지 선택
index = [0, 1, 2]   # source list에서 뽑을 index
combi = list(itertools.combinations(index, howmany))
final_select = random.choice(combi)   # -> y    ex : (0, 2)


#%% (1) 특정 시간 구간에 대한 구현. 
import random

total_time = 0
while total_time < 5 or total_time > 10:
    starttime = random.choice(range(0, 300)) + random.random()
    endtime = random.choice(range(int(starttime), 301)) + random.random()
    total_time = endtime - starttime

# %% (2) data 추출
import numpy as np

# 최종적으로 여러가지가 합쳐진 spectrum
final_hist = np.zeros([1000])

for idx in final_select:
    source_ = source[idx]
    csv_file = f"../../Data/240603/modi/{source_}_close.csv"
    fil_dt = modi_hist_extract(csv_file)  # filtered data
    fil_dt.filtered(starttime, endtime)
    final_hist += fil_dt.histvalues