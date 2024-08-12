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
        plt.figure(figsize=(8,10))
        plt.plot(self.hist)
        plt.xlabel("Energy bin")
        plt.ylabel("eV")
        pixelsize = 50
        x_ticks = [i*pixelsize for i in range(0, int(1000/pixelsize))]
        x_labels = [(1000/1000)*pixelsize*i for i in range(0, int(1000/pixelsize))]
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
        
        