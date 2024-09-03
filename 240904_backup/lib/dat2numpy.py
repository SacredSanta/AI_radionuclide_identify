'''
최종 수정 : 2024.07.15.
사용자 : 서동휘

<수정 내용> 

<처음>
gate simulation 을 통해 얻은 .dat 파일을 csv 로 넘기는 파일
'''

#%%
import numpy as np
import pandas as pd

debug = 0

def gate_data_conversion(dpath):
        
    with open(dpath, 'r') as f:
        for line in f:
            dtlist = list(line.split())[:-2]
            if debug:print("--- gate original data ---"); print(float(dtlist[-8]), len(dtlist))
            '''
            dtlist[2] source id
            dtlist[3] x source
            dtlist[4] y source
            dtlist[5] z source
                        
            dtlist[-9] time
            dtlist[-8] energy
            dtlist[-7] x position of detector
            dtlist[-6] y position of detector
            dtlist[-5] z position of detector                    
            '''
            idx = int(round(float(dtlist[-8])*1000,0))   # 나중에 이 floor 한 것이 data 신뢰도에 영향을 줄 수 있음.
            if debug: print("idx : ",idx); break
            if not idx>=1000 and not idx<0:
                dt[:,idx] += 1
    return dt



class SinglesData():
    def __init__(self, dt, dtype_list, additional_data=None, *more_data):
        newnum = 10      
        print(type(additional_data))
        try:
            print("additional data inserted")
            self.sourceID = np.concatenate( (dt["sourceID"], newnum*np.ones(len(additional_data["sourceID"]), dtype=int)) )
            self.energy = np.concatenate( (dt["energy"], additional_data["energy"]) )
        except:
            self.sourceID = dt["sourceID"]
            self.energy = dt["energy"]
            
            
            

class filteredData():  # 선원 번호 지정 부분만 빼오기
    def __init__(self, filter_source_idx:dict, data):
        self.energy = data.energy
        self.idx_dict = filter_source_idx
        
    # 고른 source 모두 histogram에 저장    
    # ? .sortall([filter할 sourceID])
    def sortall(self, num:list)->np.array:    
        # histogram 초기화
        self.histogram = np.zeros([1,1000])
        # 여러 source에 대해서 반복
        for srcnum in num:
            # 한 source가 해당되는 id
            idx = self.idx_dict[str(srcnum)]
            
            # 각 idx에 해당하는 energy값 histogram에 더해주기
            for i in idx:
                ene = int(round(float(self.energy[i])*1000,0))
                if ene < 0 or ene > 999:
                    continue
                else:
                    self.histogram[0,ene] += 1
        
        return self.histogram
    
    
    
    # 고른 source 정한 count 까지만 저장
    # ? .sortall([filter할 sourceID], 저장할 총 카운트 수)
    def make_uni(self, components:list, count:int):
        # histogram 초기화
        self.histogram = np.zeros([1,1000], dtype=int)
        
        seed = len(components)
        
        # 각 src마다 count 개수가 달라 idx를 동일하게 보면서 가면 bound 넘음.
        idx = np.zeros([1,seed], dtype=int)
        
        # num -> [0,3,5 ...]  source list
        # seed -> source list의 index
        # idx -> 특정 source에서 energy array 안에서의 index
                
        while int(sum(self.histogram[0]))!=count:
            data_idx = self.idx_dict[str(components[seed-1])][idx[0,seed-1]]
                
            # 각 idx에 해당하는 energy값 histogram에 더해주기
            ene = int(round(float(self.energy[data_idx])*1000,0))
            if ene < 0 or ene > 999:
                continue
            else:
                self.histogram[0,ene] += 1
            
            idx[0,seed-1] += 1   # 특정 source energy array의 다음 index로    
            seed += 1
            if seed > len(components) : seed = 1
        
        return self.histogram
                
        
        
    # 고른 source 중 count 까지 random하게 선별
    def make_rancombi(self, components:list, count:int):
        # histogram 초기화
        self.histogram = np.zeros([1,1000], dtype=int)
        
        while int(sum(self.histogram[0]))!=count:
            for src in components:
                idx_ = np.random.choice(len(self.idx_dict[str(src)])-1)  # 해당 source의 data의 index             
            
                data_idx = self.idx_dict[str(src)][idx_]
                
                ene = int(round(float(self.energy[data_idx])*1000,0))
                if ene < 0 or ene > 999:
                    continue
                else:
                    self.histogram[0,ene] += 1
                
                
                if int(sum(self.histogram[0]))==count : return self.histogram
                
        return self.histogram
        
        
        
    def show(self):   # 히스토그램 plot
        plt.plot(self.histogram[0])
        
        
        
        

# ===========================================================================================
# 24.07.19 최신

# modi_hist_extract 재사용을 위해 연결방식
# raw data -> csv -> pandas(hist_extract)
# csv column - num, total energy, timeStamp
class gate2csv():
    def __init__(self, dpath, datatype='npy'):
        
        # 저장형태에 따라 변환이 조금 다름.
        # 1.numpy 에서 불러올 때
        if datatype == 'npy':
            self.dt = np.load(dpath)
            #Index(['event_numbers', ' total_energy (keV)', ' ch1_energy', ' ch2_energy',
            #      ' ch3_energy', ' ch4_energy', ' time_stamp (sec)', 'ene'],
            #       dtype='object')
            dt_temp = {'event_numbers' : dt['eventID'],
                       ' total_energy (keV)' : dt['energy']*1_000_000,   # 검출기가 단위가 eV라서
                       ' time_stamp (sec)' : dt['time']}

            self.dt_pd = pd.Dataframe(dt_temp)
            
            
        # 2.ascii(dat) 파일에서 불러올 때
        elif datatype == 'dat':
            from joblib import Parallel, delayed, cpu_count
            # column 저장 관련 flag
            self.colflag = [0, 1, 0, 0, 0,
                            0, 0, 0, 0, 0,
                            0, 0, 0, 1, 1,
                            0, 0, 0, 0, 0,
                            0, 0, 0, 0]
            self.col = ['runID', 'eventID', 'sourceID', 'sourceX', 'sourceY',
                        'sourceZ', '7', '8', '9', '10',
                        '11', '12', '13', 'Timestamp', 'Energy',
                        'singleX', 'singleY', 'singleZ', '19', '20',
                        '21', '22', '23', '24']
            
            # ascii 한줄씩 array 바꿔주기
            def ascii_process(line):
                line_arr = np.array(line.split())
                return line_arr

            with open(dpath, 'r') as f:
                results = Parallel(n_jobs=cpu_count(), verbose=10)(delayed(ascii_process)(line) for line in f)
            results = np.array(results) 
            
            # pandas로 처리.
            self.dt = pd.DataFrame(results)   
            target = [1,13,14]
            for i in target:
                self.dt[i] = self.dt[i].astype('float64')
            self.dt.rename(columns={1:'event_numbers', 13:' total_energy (keV)', 14:' time_stamp (sec)'}, inplace=True)
            
            #* 왜 곱셈이 안되나 했더니, string 형태여서 값이 엉망진창되며 memory out 됨.
            self.dt[' total_energy (keV)'] = round(self.dt[' total_energy (keV)'] * 1_000_000, 2)
            self.dt_pd = self.dt.iloc[:, target] 
            
            

        # 예외
        else:
            print("data type error.")
            return 
        
        
        print("instance created. You received the pandas object, so you need to convert it to csv files to use in modi_hist_extract. (dt_pd)")
        
    # 시간에 따른 변환
    def filtered(self, time=None, count=None):
        pass
    




    





# %% ===============================================================
# 24.08.05
# last modified : Dong Hui Seo 
# bin -> csv?

## load segmap
# csv file name
import csv

filename = "../../../Data/230000_unknown/resource/7T_Ba133_SegmentationMap.bin.csv"
# initializing the titles and rows list
segmap = []
with open(filename, 'r') as csvfile:
    # creating a csv reader object
    csvreader = csv.reader(csvfile, delimiter=',')

    # extracting each data row one by one
    for row in csvreader:
        segmap.append([int(x) for x in row])
        
segmap = np.array(segmap)  # 512,512

#%%
## load coefficient
filename = "../../../Data/230000_unknown/resource/7T_corr_coef.csv"
coef = []

with open(filename,'r') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    for row in csvreader:
        coef.append([float(x) for x in row])

coef = np.array(coef)  # 361, 2


#%%
def process1(filename: str,
             segmap: np.ndarray,
             corr_coef: np.ndarray):
    
    # load data
    datafile = "./Data2/ori/{}.bin".format(filename)
    with open(datafile, "rb") as f:
        # move file handle to the end of the file
        f.seek(0,2) 
        maxIdx = round(f.tell())
        
        # move file handle to the start index of the file
        f.seek(0) 
        
        # init results array
        result = np.empty((0,4), int)

        # 1. data의 끝까지 take data || results : [x+, x-, y+, y-]
        while f.tell() < maxIdx:
            sum_Idx = int((np.fromfile(f, dtype='<u4',count=1) - 4) / 16)  # 뒤에 연산식은 왜??
            indata = np.fromfile(f, dtype='<u4', count=4*sum_Idx)  # 실제 data
            indata = np.reshape(indata, (sum_Idx, 4))  # reshape? 
            result = np.append(result, indata, axis=0)  # result 에 이어 붙이기

        # apply segmap
        x = np.reshape(np.divide((result[:,0] - result[:,1]),
                                 (result[:,0] - result[:,1]),
                                 out=np.zeros_like(np.float64(result[:,0])),
                                 where=(result[:,0] - result[:,1])!=0),
                        (len(result),1) )
        y = np.reshape(np.divide((result[:,2] - result[:,3]),
                                 (result[:,2] - result[:,3]),
                                 out=np.zeros_like(np.float64(result[:,2])),
                                 where=(result[:,2] - result[:,3])!=0),
                        (len(result),1))
        en_temp = np.sum(result,axis=1,keepdims=True)

        # positioning
        xidx = np.linspace(-1, 1, 512)
        yidx = np.linspace(-1, 1, 512)
        mXidx = abs(xidx - x)
        cXidx = np.argmin(mXidx)
        mYidx = abs(yidx - y)
        cYidx = np.argmin(mYidx)
        seg_idx = segmap[cYidx,cXidx] - 1
        photonenergy = np.int64(((en_temp >> 5) * corr_coef[seg_idx,0] + corr_coef[seg_idx,1]))

    return photonenergy



dtpath = "../../../Data/221021_frog_indoor_angularRes/3.5t_6mm_ba133_25cm.bin"

#datafile = "./Data2/ori/{}.bin".format(filename)
with open(dtpath, "rb") as f:
    # move file handle to the end of the file
    f.seek(0,2) 
    maxIdx = round(f.tell())
    
    # move file handle to the start index of the file
    f.seek(0) 
    
    # init results array
    result = np.empty((0,4), int)

    # 1. data의 끝까지 take data
    while f.tell() < maxIdx:
        sum_Idx = int((np.fromfile(f, dtype='<u4',count=1) - 4) / 16)  # 뒤에 연산식은 ARALE 설정값인 것으로 알고 있음.
        indata = np.fromfile(f, dtype='<u4', count=4*sum_Idx)  # 실제 data
        indata = np.reshape(indata, (sum_Idx, 4))  # reshape? 
        result = np.append(result, indata, axis=0)  # result 에 이어 붙이기
    
    # 1-2. 이부분 코드의 의미를 모르겠음.. 형태 초기화인가?    
    x = np.reshape(np.divide((result[:,0] - result[:,1]),
                        (result[:,0] - result[:,1]),
                        out=np.zeros_like(np.float64(result[:,0])),
                        where=(result[:,0] - result[:,1])!=0),
            (len(result),1) )
    
    y = np.reshape(np.divide((result[:,2] - result[:,3]),
                                (result[:,2] - result[:,3]),
                                out=np.zeros_like(np.float64(result[:,2])),
                                where=(result[:,2] - result[:,3])!=0),
                    (len(result),1))
    en_temp = np.sum(result,axis=1,keepdims=True)
    
    
    # 2. positioning
    xidx = np.linspace(-1, 1, 512)
    yidx = np.linspace(-1, 1, 512)
    mXidx = abs(xidx - x)
    cXidx = np.argmin(mXidx)  # 511
    mYidx = abs(yidx - y)
    cYidx = np.argmin(mYidx)  # 511

#%%
seg_idx = segmap[cYidx, cXidx] - 1

# photonenergy 연산 규칙도 이유는 모르겠음.
photonenergy = np.int64(((en_temp >> 5) * corr_coef[seg_idx,0] + corr_coef[seg_idx,1]))
    
    
    
#%%
import numpy as np

if __name__ == "__main__":
    dpath = "../../../Data/240527_simulation/Ba133/id1052.dat"    
    
    data = gate2csv(dpath, datatype='dat')
    
    data.dt_pd.to_csv("./test/simuldata.csv", encoding='utf-8')
    
    
    
    
    
    
    
    
    
    
    
    
    
