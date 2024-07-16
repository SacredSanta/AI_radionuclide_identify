#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def my_hist(dt:pd.DataFrame, bins=1000):
    result = np.zeros([bins+1])
    arr = np.array(dt)
    for dat in arr:
        if type(dat) == np.float64 :
            result[int(dat)] += 1
        else:
            result[dat] += 1
    return result

            
class modi_hist_extract():
    def __init__(self, csvfile:str):        
        self.dt = pd.read_csv(csvfile)
        self.col = list(self.dt.keys())  # column 저장하기
        self.dt["ene"] = np.round(np.array(self.dt[" total_energy (keV)"] / 1000))
        self.hist = my_hist(np.array(self.dt["ene"]))
       
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
        self.filtered_hist = my_hist(np.array(self.filtered_dt["ene"]))
        
        
        
        
if __name__ == '__main__':
    source_ = 'ba133'
    distance = 'close' 
    csv_file = f"../../../Data/240603_nucare/ori/{source_}_{distance}_5min.csv"

    sig_dt = modi_hist_extract(csv_file)




# %%
