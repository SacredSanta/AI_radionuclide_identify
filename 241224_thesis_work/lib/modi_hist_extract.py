'''
최종 수정 : 2024.09.21.
사용자 : 서동휘

<수정 내용> 

<2024.08.05.>
- hist 관련 np.histogram으로 이용설정.
- 기존 코드의 순서 : csv_hist_modify 를 통해 original로 얻은 csv 파일을 수정하여 modified_csv 를 획득
               -> modi_hist_extract 를 통해 이용
- 변경된 코드 : modi_hist_extract 만 이용

<2024.07>
modi 관련 삭제

<처음>
검출기에서 얻은 csv 데이터파일을 numpy로 전환
energy 위치 안 맞는 부분을 보정해준 후,
조건을 지정하여 그 부분의 data만 저장.
'''

#%% csv_hist_modify (구버전)  (DEPRECATED!!)
'''
class csv_hist_modify_old():
    def __init__(self, filename:str):
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        
        self.dt = pd.read_csv(filename)
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
        print(coef)
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
        
        
        
        
        
        
        
def my_hist(dt:pd.DataFrame, bins=1000):
    result = np.zeros([bins+1])
    arr = np.array(dt)
    for dat in arr:
        if int(dat) < 1001:
            result[int(dat)] += 1
        else:
            continue
    return result
    
    
    
    
# 선형 방식의 energy correction (DEPRECATED!)   
def cal_coefs(target_peak:list, sig_dt_peaks_idx:list):
        coef_a = (target_peak[0] - target_peak[1]) / (sig_dt_peaks_idx[0] - sig_dt_peaks_idx[1])
        coef_b = target_peak[0] - coef_a*sig_dt_peaks_idx[0]
        return coef_a, coef_b


'''



#%% modi_hist_extract -----------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import struct 
import pandas as pd 
from joblib import Parallel, delayed, cpu_count   
import os
from scipy.optimize import nnls

# bin file 뜯기위한 info
bytes_count = [8,8,4,4,4,4]
bytes_type = ['<2f','<2i','<i','<i','<i','<i']
cols = ['time_stamp', 'event_number', 'ch1', 'ch2', 'ch3', 'ch4']

# joblib 통해서 bin 파일의 count 하나씩 뜯는 함수
def bin_data_stream(nn, filepath):
    with open(f"{filepath}", 'rb') as file:
        file.seek(nn*32) # 32 단위마다 data 있으니까.. - 병렬 처리 위해서 nn 이용
        new_row = {}
            
        # 각 컬럼에 대해서 넣어주기
        for ii in range(6):
            val_hx = file.read(bytes_count[ii])
            val = struct.unpack(bytes_type[ii], val_hx)[0]
            new_row[cols[ii]] = val
                 
    return new_row
                
                   
def bin_data_stream_old(nn, filepath):
    with open(f"{filepath}", 'rb') as file:
        file.seek(nn*32)
        new_row = {}
               
        # 각 컬럼에 대해서 넣어주기
        for ii in range(6):
            val_hx = file.read(bytes_count[ii])
            val = struct.unpack(bytes_type[ii], val_hx)[0]
            new_row[cols[ii]] = val
    return
 


# spectrum data(csv)를 class 객체로 생성            
class modi_hist_extract():
    def __init__(self, filename:str, cali='off', isbin='no'):
        
        # calibration 되어 있으면, 1000 eV 나눠줄 필요 없음.
        if cali=='on':
            self.divid_ = 1
        else:
            self.divid_ = 1000

        # file 종류별로 뜯기  - arale 24.02 gammaspectroscopy version data
        if isbin == 'no':   # csv 형태   
            self.dt = pd.read_csv(filename)
            self.col = list(self.dt.keys())  # column 저장하기
        elif isbin == 'yes':  # bin 형태
            num_cores = cpu_count()
            filesize = int(os.path.getsize(filename)/32)           
            results = Parallel(n_jobs=num_cores, verbose=10)(delayed(bin_data_stream)(num, filename) for num in range(filesize))
            self.dt = pd.DataFrame(results)
            self.dt[" total_energy (keV)"] = self.dt["ch1"]+self.dt["ch2"]+self.dt["ch3"]+self.dt["ch4"] 
        elif isbin == 'old':  # 구버전 형태 data
            num_cores = cpu_count()
            filesize = int(os.path.getsize(filename)/32)           
            
        
        else:  # 예외
            print("wrong input")
            return    
    
        self.dt["ene"] = np.round(np.array(self.dt[" total_energy (keV)"] / self.divid_))
        self.hist = np.histogram(np.array(self.dt["ene"]), bins=[i for i in range(1001)])[0]
        self.peakidx = None
        self.filename = filename
    
    def __call__(self):
        print("how ? : modi_hist_extract(self, filename, cali='off, isbin='no')")
        print("method")
        print("- show()")
        print("- show_fil()")
        print("- find_peak(searchrange)")
        print("- fix_data(target_peak)")
        print("- fix_data_with_scalefactor(coef_a, coef_b)")
        print("- filtered(strat_time, end_time)")
        print("- filtered_counts(count_, starttime, endtime)")
        print("- show_filtered_dt()")
    
    # ----------------------------------------------
    def show(self):
        print(self.filename)
        plt.figure()
        plt.plot(self.hist)
        plt.xlabel("Energy bin")
        plt.ylabel("Counts")
        pixelsize = 50
        x_ticks = [i*pixelsize for i in range(0, int(1000/pixelsize))]
        x_labels = [int((1000/1000)*pixelsize*i) for i in range(0, int(1000/pixelsize))]
        plt.xticks(ticks=x_ticks, labels=x_labels, fontsize=7)
        
    def show_fil(self):
        if not hasattr(self, 'filtered_hist'):
            print("There's no filtered data.")
            return 
        plt.figure()
        plt.plot(self.filtered_hist)
        plt.xlabel("Energy bin")
        plt.ylabel("Counts")
        pixelsize = 50
        x_ticks = [i*pixelsize for i in range(0, int(1000/pixelsize))]
        x_labels = [int((1000/1000)*pixelsize*i) for i in range(0, int(1000/pixelsize))]
        plt.xticks(ticks=x_ticks, labels=x_labels, fontsize=7)

    
    # peak 위치 탐색
    def find_peak(self, searchrange:list):
        # 구간의 peak 값 확인
        max_idx = np.argmax(self.hist[searchrange[0]:searchrange[1]+1]) 
        print(max_idx)
        
        # for i in range(searchrange[0], searchrange[1]+1):
        #     if self.hist[i] == max_val:
        #         idx = i
        #         break
        self.peakidx = searchrange[0] + max_idx
        print("Peak index : ", self.peakidx)
    
    
    # peak 위치 보정
    def fix_data(self, target_peak:int):
        if self.peakidx == None:
            print("You should find the peak idx first.")
            return
        
        print("peak location will updated to : {} -> {}".format(self.peakidx, target_peak))
        
        self.scale_factor = target_peak / self.peakidx
        
        self.dt[" total_energy (keV)"] *= self.scale_factor
        self.dt["ene"] = np.round(np.array(self.dt[" total_energy (keV)"] / self.divid_))
        self.hist = np.histogram(np.array(self.dt["ene"]), bins=[i for i in range(1001)])[0]
        
        
    # 다른 데이터에서 얻은 scale factor 기준을 이용하여 데이터를 수정할 때.    
    def fix_data_with_scalefactor(self, coef_a:float, coef_b:float):  # aX+b 방식 (두개의 선원에 관한 스펙트럼으로 구하기)
        #self.dt["ch1"] = self.dt["ch1"]*coef_a + coef_b  # 바꾸려다가 column name 을 모두 바궈야하는 불상사가 생겨
        #self.dt["ch2"] = self.dt["ch2"]*coef_a + coef_b
        #self.dt["ch3"] = self.dt["ch3"]*coef_a + coef_b
        #self.dt["ch4"] = self.dt["ch4"]*coef_a + coef_b
        
        # coef_b 의 단순 연산으로 인한 예외 처리를 위한 index
        # total energy 에서는 /1000 이전이니까 *1000 으로 보정해야됨
        # temp_idx = self.dt.loc[self.dt[" total_energy (keV)"] > (-coef_b)*1000].index 
        # !100000 언저리부분이 다 없어져 버림
        # # self.dt.loc[self.dt[" total_energy (keV)"] > (-coef_b)*1000, " total_energy (keV)"] = self.dt.loc[self.dt[" total_energy (keV)"] > (-coef_b)*1000, " total_energy (keV)"]*coef_a + coef_b*self.divid_  
        
        self.dt[" total_energy (keV)"] = self.dt[" total_energy (keV)"]*coef_a + coef_b*self.divid_
        self.dt["ene"] = np.round(np.array(self.dt[" total_energy (keV)"] / self.divid_))
        self.hist = np.histogram(np.array(self.dt["ene"]), bins=[i for i in range(1001)])[0]
       
       
    # 시작시간, 끝나는 시간    
    def filtered(self, starttime, endtime):      
        import numpy as np  
        if hasattr(self,'filtered_dt'):
            del self.filtered_dt
        self.filtered_dt = self.dt[(self.dt[" time_stamp (sec)"] >= starttime) 
                                    & (self.dt[" time_stamp (sec)"] < endtime)]
        
        # # max값 관련 오류있다면..
        # try :
        #     counts, bin = np.histogram(self.filtered_dt["modi_ene"], bins=int(self.filtered_dt["modi_ene"].max()+1))
        #     self.histvalues = counts[0:1001] # np.array - 1,1000 : 실제 histogram 그리는 값들
        #     #self.histvalues_bin = bin[0:1001]  # 값이 존재하는 idx 부분 저장 (사실 필요없음)
            
        #     # 구간의 최대값이 1000보다 작은 경우 histogram channel이 1000개 이하로 생성될 수 있다.
        #     if self.filtered_dt["modi_ene"].max() < 1001:
        #         new_array = np.zeros(1001)
        #         new_array[:counts.shape[0]] = counts
        #         self.histvalues = new_array
                
        # except :
        #     #print("None value")
        #     #print(self.filtered_dt["modi_ene"].max())
        #     #print(starttime)
        #     #print(endtime)
        #     self.histvalues = np.zeros(1001)
        #     return        
        self.filtered_hist = np.histogram(np.array(self.filtered_dt["ene"]), bins=[i for i in range(1001)])[0]
    
    def filtered_counts(self, count_, starttime, endtime): # count 만큼 뽑아내기
        import numpy as np
        
        # 이미 filtered_dt 있으면 attritube 삭제
        if hasattr(self,'filtered_dt'):
            del self.filtered_dt
        
        # 범위 시간 조정
        fil_dt = self.dt[(self.dt[" time_stamp (sec)"] >= starttime) & (self.dt[" time_stamp (sec)"] < endtime)]
        
        # count 지정 안했다면,
        if count_ == 0: 
            self.filtered_dt = fil_dt
        # count 지정 했다면,
        else:
            # 시간을 통해 filter 된 얘들 index 값
            idx_range = fil_dt.index
            
            # 예상하는 count 가 범위에 다 존재하지 않을 수 있으니..
            if count_ < len(idx_range):
                idx_ = np.random.choice(idx_range, count_)
            else:
                idx_ = np.array(idx_range)

            self.filtered_dt = fil_dt.loc[idx_]
            
        self.filtered_hist = np.histogram(np.array(self.filtered_dt["ene"]), bins=[i for i in range(1001)])[0]
                        
        
    def show_filtered_dt(self):
        print(self.filename)
        plt.figure()
        plt.plot(self.filtered_hist)
        plt.xlabel("Energy bin")
        plt.ylabel("Counts")
        pixelsize = 50
        x_ticks = [i*pixelsize for i in range(0, int(1000/pixelsize))]
        x_labels = [int((1000/1000)*pixelsize*i) for i in range(0, int(1000/pixelsize))]
        plt.xticks(ticks=x_ticks, labels=x_labels, fontsize=7)
        
    
     
        


# NNLS 이용한 aX+b 해 찾기 
def cal_coefs(target_peak:list, cur_peak_idx:list):
    A = np.vstack([cur_peak_idx, np.ones(len(cur_peak_idx))]).T
    solution, _ = nnls(A, target_peak)
    return solution
   
        
        
        
        
   
        
        
        
        
        
        
        
        
        
# %% ===============================================================================
# 스펙트럼 보정을 여기에서 진행하였음..
# =======================================================================        


import os 

if __name__ == '__main__':
    direc = "../../../Data/"
    foldername = "240627_nucare/ori/thres700/"
    #foldername = "240906_arale_indoor_det2"

    # 지정 폴더 파일 이름들
    files = os.listdir(os.path.join(direc, foldername))

    for i in range(len(files)):
        print(i, '|',  files[i])
    print("--")
    
    b_on = 0
        

#%%
    # file 2개 지정해서 주소로 저장
    datadir_a = os.path.join(direc, foldername, files[25])   # 13
    if b_on: datadir_b = os.path.join(direc, foldername, files[2])
    print(datadir_a)

    # make csv 각각
    sig_dt_a = modi_hist_extract(datadir_a, cali='off', isbin='no')
    if b_on: sig_dt_b = modi_hist_extract(datadir_b, cali='off', isbin='yes')
    sig_dt_a.show()
#%% 1. check peak ========================================================================
    sig_dt_a.show()

    # 각각의 peak 를 찾아줌 - 눈대중으로 위치 파악해서
    sig_dt_a.find_peak([350,500]) # -> a
    if b_on: sig_dt_b.find_peak([150,200])

#%% ========================================================================
    print(sig_dt_a.peakidx)
    if b_on: print(sig_dt_b.peakidx)

#%% 2. calculate coef ========================================================================
    '''
    # 실제 peak 여야하는 곳 입력
    target_peak = []  # 큰거부터 작성
    target_peak.append(662)
    if b_on: target_peak.append(356)

    
    # aX+b 방정식 통해서 a,b 계산
    if b_on: 
        coef_a = (target_peak[0] - target_peak[1]) / (sig_dt_a.peakidx - sig_dt_b.peakidx)
        coef_b = target_peak[0] - coef_a*sig_dt_a.peakidx
        print("coef_a : ", coef_a, '\n', "coef_b : ", coef_b)
    else:
        coef_a = target_peak[0] / sig_dt_a.peakidx
        print("coef_a : ", coef_a)
    '''
    target_peak = [662, 356]
    cur_peak = [335, 201]
    
    coefs = cal_coefs(target_peak, cur_peak)
    
#%% 3. fix_data with coef ========================================================================

    #! 딱 한번만 실행!!!!

    # 이전에 계산된 coef 쓸거라면
    direct_type = 1
    if direct_type:
        coefs = [0,0]
        coefs[0] = 1.825 #1.92186128 #2.1702127659574466 
        coefs[1] = 0

    # coef 통해서 모든 energy 값 보정
    sig_dt_a.fix_data_with_scalefactor(coefs[0], coefs[1])
    if b_on: sig_dt_b.fix_data_with_scalefactor(coef_a, coef_b)

#%% 4. check ========================================================================
    sig_dt_a.show()
    if b_on: sig_dt_b.show()

#%% 4-2. remodify ( 재수정용! )=====================================================================

    # 각각의 peak 를 찾아줌 - 눈대중으로 위치 파악해서
    sig_dt_a.find_peak([350,600]) # -> a
    target_peak = [356]
    
    coef_a = target_peak[0] / sig_dt_a.peakidx
    sig_dt_a.fix_data_with_scalefactor(coef_a, coef_b=0)
#%%
    sig_dt_a.show()
    sig_dt_a.find_peak([510, 600])



#%% 5. save as modified csv ========================================================
    savedir = "../../../Data/integrated_modified/"

    # source 위치 변경해서 잘 저장하기
    sig_dt_a.dt.to_csv(savedir+"/baco57/240627_t700.csv")
    if b_on: sig_dt_b.dt.to_csv(savedir+"/ba/240906.csv")