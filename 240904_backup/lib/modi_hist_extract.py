'''
최종 수정 : 2024.08.05.
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
class csv_hist_modify_old():
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





#%% modi_hist_extract -----------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
def my_hist(dt:pd.DataFrame, bins=1000):
    result = np.zeros([bins+1])
    arr = np.array(dt)
    for dat in arr:
        if int(dat) < 1001:
            result[int(dat)] += 1
        else:
            continue
    return result
'''
   
            
class modi_hist_extract():
    def __init__(self, csvfile:str):        
        self.dt = pd.read_csv(csvfile)
        self.col = list(self.dt.keys())  # column 저장하기
        self.dt["ene"] = np.round(np.array(self.dt[" total_energy (keV)"] / 1000))
        self.hist = np.histogram(np.array(self.dt["ene"]), bins=[i for i in range(1001)])[0]
        self.peakidx = None
    
    def show(self):
        plt.figure()
        plt.plot(self.hist)
        plt.xlabel("Energy bin")
        plt.ylabel("eV")
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
        plt.ylabel("eV")
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
        self.dt["ene"] = np.round(np.array(self.dt[" total_energy (keV)"] / 1000))
        self.hist = np.histogram(np.array(self.dt["ene"]), bins=[i for i in range(1001)])[0]
        
        
    # 다른 데이터에서 얻은 scale factor 기준을 이용하여 데이터를 수정할 때.    
    def fix_data_with_scalefactor(self, scale_factor:float):
        self.dt[" total_energy (keV)"] *= scale_factor
        self.dt["ene"] = np.round(np.array(self.dt[" total_energy (keV)"] / 1000))
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
        
        
   
   
        
#%%        
if __name__ == '__main__':
    source_ = 'background'
    distance = 'close' 
    distance2 = 'close'
    #csv_file = f"../../../Data/240603_nucare/ori/{source_}_{distance}_5min.csv"
    csv_file = f"../../../Data/240627_nucare/ori/thres700/{source_}_{distance}_10min.csv"
    #csv_file = f"../../../Data/240627_nucare/ori/thres700/{source_}_{distance}.{distance2}_5min.csv"
    
    sig_dt = modi_hist_extract(csv_file)
    sig_dt.show()
    print("total count : ",sum(sig_dt.hist))
    #sig_dt.find_peak([150,250])
    #sig_dt.filtered(5,5.1)
    #sig_dt.show_fil()
    
    # sig_dt.

# %%
