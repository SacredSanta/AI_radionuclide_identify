'''
검출기를 통해 얻은 data(csv)를
x (energy), y (time) , value (값-누적)으로 2D array로 저장하기.
'''

#%% init
#?---- -------------------------------------------------------------------------------------
# csv에서 시간에 따라서 추출
class modi_hist_extract():
    def __init__(self, csvfile:str):
        import pandas as pd
        import numpy as np
        
        self.dt = pd.read_csv(csvfile)
        self.col = list(self.dt.keys())  # column 저장하기
       
    # 시작시간, 끝나는 시간    
    def filtered(self, starttime, endtime):        
        self.filtered_dt = None
        self.filtered_dt = self.dt[(self.dt[" time_stamp (sec)"] >= starttime) 
                                    & (self.dt[" time_stamp (sec)"] < endtime)]
        
        # max값 관련 오류있다면..
        try :
            counts, bin = np.histogram(self.filtered_dt["modi_ene"], bins=int(self.filtered_dt["modi_ene"].max()+1))
            self.histvalues = counts[0:1001] # np.array - 1,1000 : 실제 histogram 그리는 값들
            #self.histvalues_bin = bin[0:1001]  # 값이 존재하는 idx 부분 저장 (사실 필요없음)
            
            # 구간의 최대값이 1000보다 작은 경우 histogram channel이 1000개 이하로 생성될 수 있다.
            if self.filtered_dt["modi_ene"].max() < 1001:
                new_array = np.zeros(1001)
                new_array[:counts.shape[0]] = counts
                self.histvalues = new_array
                
        except :
            print("None value")
            print(self.filtered_dt["modi_ene"].max())
            print(starttime)
            print(endtime)
            self.histvalues = np.zeros(1001)
            return          
        



#?---- -------------------------------------------------------------------------------------
#%% ================================================= 
# make data 
# ===================================================

#%% 3. source 선택  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
import itertools

# mix single set version
mix_version = 3

# version 1 : Ba,Na,Cs 3가지
'''
[legacy]
# if mix_version == 1:
#     mixing_list = {'Ba':['Ba133'],
#                 'Na':['Na22'],
#                 'Cs':['Cs137'],
#                 'BaCs':['Ba133','Cs137'],
#                 'BaNa':['Ba133','Na22'], 
#                 'CsNa':['Cs137','Na22'],
#                 'BaNaCs':['Ba133','Na22','Cs137'],
#                 'BG':['Background']
#             }

#     mixed_source_name = list(mixing_list.keys())
#     mixed_list = list(mixing_list.values())
'''
if mix_version == 1:
    # Ba133, Na22, Cs137, Ba133Cs137, Ba133Na22, Cs137Na22, Ba133Na22Cs137
    pass




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
total_set = 1  #! <----------------------이부분 수정해서 이용 !!
index = [i for i in range(len(source))]   # source list에서 뽑을 index

def work_1(ii):
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
    while total_time < 5 or total_time > 10:  #! <----------------------이부분 수정해서 이용 !!
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
        final_hist[0] += fil_dt.histvalues
        
    #dataset[ii,:,:] = final_hist
    return final_hist, y

results = Parallel(n_jobs=cpu_count(), verbose=10)(delayed(work_1)(task) for task in range(total_set))

# 결과 받고 numpy로 처리해주기
dataset, dataset_y = zip(*results)
dataset = np.array(dataset)
dataset_y = np.array(dataset_y)




# %% 5. save generated data

filename = "5to15sec_7_source_close_35cm_12000"
np.save(f"./0.dataset/{filename}.npy", dataset)

# y는 이미 처리되어 있으니까 1단계로 넘김.
np.save(f"./1.preprocess_data/{filename}_y.npy", dataset_y)














#?---- -------------------------------------------------------------------------------------
# %% proto of make 2D image (only one!) --------------------------------------------------
import numpy as np
from joblib import Parallel, delayed, cpu_count


index = [i for i in range(len(source))]   # source list에서 뽑을 index

source_ = 'na22'
distance = 'close'    

rowsize = 1000 # 이미지의 row 개수

starttime = 0
finaltime = 50 # data의 마지막 시간   
interval = (finaltime-starttime) / (rowsize-1)

# data 불러오기
csv_file = f"../../Data/240603/modi/{source_}_{distance}.csv"
fil_dt = modi_hist_extract(csv_file)  # filtered data

# endtime 별 해당하는 histogram row 1개씩 뽑기
accumulate = 1

# debug 용
debug_counts = []

endtime_values = np.linspace(starttime+interval, finaltime, rowsize, endpoint=True)


def onestack_histimage(startidx, endidx):
    global endtime_values
         
    # filter할 time 구간지정
    previous_time = endtime_values[startidx]
    if endidx > 999:
        return np.zeros(1001)
    endtime = endtime_values[endidx]
    
    # 누적상황을 보여주고 싶으면 filter time 구간 처음은 무조건 0
    if accumulate: previous_time = 0
    
    # filter 진행
    fil_dt.filtered(previous_time, endtime)
    
    return fil_dt.histvalues
    
# starttime ~ finaltime 사이를 rowsize 간격으로 나누어서 누적상태로 각 row에 저장.

results = Parallel(n_jobs=cpu_count(), verbose=10)(delayed(onestack_histimage)(i, i+1) for i in range(len(endtime_values)))
results = np.array(results)

#%% (Debug) check shapes of results 
for i in range(len(results)):
    if results[i].shape != (1001,):
        print(i)
#%% (Debug) check sum of each results
debug_sum = []
for i in range(len(results)):
    debug_sum.append(sum(results[i]))
plt.plot(debug_sum)
        
#%% show image
import matplotlib.pyplot as plt
plt.imshow(results, cmap='turbo', vmin=0, vmax=100)

# figure settings
plt.xlabel("Energy bin")
plt.ylabel("Time(s)")
pixelsize = 100
y_ticks = [i*pixelsize for i in range(0,10)]
y_labels = [(finaltime/1000)*pixelsize*i for i in range(0,10)]
plt.yticks(ticks=y_ticks, labels=y_labels)

if accumulate :
    text = "Accumulate On"
else:
    text = "Accumulate Off"
plt.text(1050, 50, text, fontsize=12, color='blue')



#%% (Debug) show each row of results
for i in range(10):
    plt.subplot(10,1,i+1)
    plt.plot(results[415+i,:])
















#?---- -------------------------------------------------------------------------------------
# %% -================================================
# make data
# ======================================================
#%% 1. source choice version 
# version 3: 11가지
mix_version = 3   # <----------------변형해서 사용!

if mix_version == 3:
    source = ['ba133', 'cs137', 'na22', 'background']
    
elif mix_version == 4:
    #source = ['Ba133', 'Cs137', 'Na22', 'Am241', 'Co60', 'Ga67', 'I131', 'Ra226', 'Tc99m', 'Th232', 'Tl201', 'background']
    source = ['ba133', 'cs137', 'na22', 'background', 'co57', 'th232', 'ra226']
    # am241, co60, CRM 빠짐

#%% 2.==================================================================
import numpy as np
from joblib import Parallel, delayed, cpu_count
import itertools
import random


total_data_count = 1000 # <-------------------- 개인 변경
rowsize = 1000 # 이미지의 row 개수  # <-------------------- 개인 변경
accumulate = 1 # <-------------------- 개인 변경
time_range = [1, 100] # <-------------------- 개인 변경


index = [i for i in range(len(source))]   # source list에서 뽑을 index


def onestack_histimage(startidx, endidx, fil_dt):
    global endtime_values
         
    # filter할 time 구간지정
    previous_time = endtime_values[startidx]
    if endidx > 999:
        return np.zeros(1001)
    endtime = endtime_values[endidx]
    
    # 누적상황을 보여주고 싶으면 filter time 구간 처음은 무조건 0
    if accumulate: previous_time = 0
    
    # filter 진행
    fil_dt.filtered(previous_time, endtime)
    
    return fil_dt.histvalues

def make2d_dataset(count):
    # (1) source 관련 선택
    final_image = np.zeros([1000, 1000])
    howmany = random.choice([1,2]) # 선원 조합 1개까지, 2개까지 선택
    combi = list(itertools.combinations(index, howmany)) # index 조합
    final_select = random.choice(combi) # index 조합 중 하나 선택

    y = np.zeros([1, len(index)]) # index 통해서 y도 초기화
    for i in final_select:
        y[0, i] = 1

    # (2) 특정 시간 추출
    total_time = 0
    while total_time < time_range[0] or total_time > time_range[1]:  # time 길이 체크
        starttime = random.choice(range(0, 300)) + random.random()  # start random 뽑기
        finaltime = random.choice(range(int(starttime), 301)) + random.random() # end random 뽑기
        total_time = finaltime - starttime # 총 시간
    time_interval = total_time / (rowsize-1)
    endtime_values = np.linspace(starttime+time_interval, finaltime, rowsize, endpoint=True)

    # (3) data 추출
    # (3-1) 선원들에 대해서 반복
    for idx in final_select:
        source_ = source[idx]
        distance = random.choice(['close', '35cm'])
        csv_file = f"../../Data/240603/modi/{source_}_{distance}.csv"
        fil_dt = modi_hist_extract(csv_file)  # filtered data

        results = Parallel(n_jobs=12, verbose=1)(delayed(onestack_histimage)(i, i+1, fil_dt) for i in range(len(endtime_values)))
        results = np.array(results)
        final_image += results[:,:1000]
        
    return final_image, y # -> 1개의 1000x1000 image 생성완료
    
final_image = np.zeros([total_data_count, 1000, 1000])
final_image_y = np.zeros([total_data_count, 1, len(index)])

for count in range(total_data_count):
    final_image[count,:,:], final_image_y[count,:,:] = make2d_dataset(count)
print("Work Done! No error...")
    

# %% 결과 잘 되었는지 확인
import matplotlib.pyplot as plt
plt.imshow(final_image[1], cmap='turbo')



# %% 5. save generated data
filename = f"{time_range[0]}to{time_range[1]}sec_{len(index)}source_{total_data_count}_acc{accumulate}"
print(filename)
np.save(f"./0.dataset/{filename}.npy", final_image)

# y는 이미 처리되어 있으니까 1단계로 넘김.
np.save(f"./1.preprocess_data/{filename}_y.npy", final_image_y)


# %%
