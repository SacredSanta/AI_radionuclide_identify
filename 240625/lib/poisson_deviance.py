#%%
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# 예제 데이터 생성
np.random.seed(0)
n_samples = 100
X = np.random.poisson(lam=3, size=n_samples)
y = np.random.poisson(lam=2 + 0.5 * X, size=n_samples)

# 데이터 프레임으로 변환
data = pd.DataFrame({'X': X, 'y': y})

# 포아송 회귀 모델 적합
model = smf.poisson('y ~ X', data=data).fit()

# 모델 요약 출력
print(model.summary())

# 로그 가능도 값들
loglik_fitted = model.llf
loglik_saturated = np.sum(sm.families.Poisson().loglike(y, y))

# Deviance 계산
deviance = 2 * (loglik_saturated - loglik_fitted)

print(f"Deviance: {deviance}")
# %%




#%% init
import sys

sys.path.append("/tf/latest_version/3.AI/Tensorflow/Code/donghui_prac/lib")
source = ["ba133", "cs137", "na22", "background"]


# %% proto of make 2D image (only one!) --------------------------------------------------
import numpy as np
from joblib import Parallel, delayed, cpu_count
from modi_hist_extract import modi_hist_extract

index = [i for i in range(len(source))]   # source list에서 뽑을 index

source_ = 'cs137'
distance = 'close'    

rowsize = 1000 # 이미지의 row 개수

starttime = 0
finaltime = 50 # data의 마지막 시간   
interval = (finaltime-starttime) / (rowsize-1)

# data 불러오기
csv_file = f"../../../Data/240603/modi/{source_}_{distance}.csv"
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
x_ticks = [i*pixelsize for i in range(0,10)]
x_labels = [i*100 for i in range(0,10)]

y_ticks = [i*pixelsize for i in range(0,10)]
y_labels = [(finaltime/1000)*pixelsize*i for i in range(0,10)]

plt.xticks(ticks=x_ticks, labels=x_labels)
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



#%%
import matplotlib.pyplot as plt
fil_dt.filtered(10, 50)
plt.plot(fil_dt.histvalues)
sig = fil_dt.histvalues
# %% EWMA
import numpy as np

alpha = 0.02
ewma_values = np.zeros(1000)

def ewma(sig:np.array, alpha:int, idx:int):
    global ewma_values
    if idx==0 : return sig[idx]
    return alpha*sig[idx] + (1-alpha)*ewma_values[idx-1]

for t in range(0,1000):
    ewma_values[t] = ewma(sig, alpha, t)
# %%
plt.plot(ewma_values)
plt.plot(sig, alpha=0.3)





# %%
import math
w = 3
boxcar_values = np.zeros(1000)
wrange = math.floor((w-1)/2)

sig_ = np.pad(sig, pad_width = wrange, mode='constant', constant_values=0)

def boxcar(sig, w, idx):
    global boxcar_values
    return (1/w) * sum([sig_[tt] for tt in range(idx-wrange, idx+wrange+1)])
        
    
    return (1/w)*sig[idx]

for t in range(0,1000):
    
