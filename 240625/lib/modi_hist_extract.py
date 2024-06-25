class modi_hist_extract():
    def __init__(self, csvfile:str):
        import pandas as pd

        
        self.dt = pd.read_csv(csvfile)
        self.col = list(self.dt.keys())  # column 저장하기
       
    # 시작시간, 끝나는 시간    
    def filtered(self, starttime, endtime):      
        import numpy as np  
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
            #print("None value")
            #print(self.filtered_dt["modi_ene"].max())
            #print(starttime)
            #print(endtime)
            self.histvalues = np.zeros(1001)
            return        