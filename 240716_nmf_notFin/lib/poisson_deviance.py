
#%% #* =======================================================================
#*                              Load Raw Data!
#* =======================================================================
#%% init ------------------------------------------------------------
import sys

sys.path.append("/tf/latest_version/3.AI/Tensorflow/Code/donghui_prac/lib")
source = ["ba133", "cs137", "na22", "background"]



#%% #* ========================== 1 sig test =================================
#%% sig file --------------------------------------------------------
from modi_hist_extract import *

source_ = 'ba133'
distance = 'close' 
csv_file = f"../../../Data/240603_nucare/ori/{source_}_{distance}_5min.csv"

sig_dt = modi_hist_extract(csv_file)

sig_dt.dt["ene"] = round(sig_dt.dt[" total_energy (keV)"] / 1000)

import matplotlib.pyplot as plt

sig_hist = np.histogram(sig_dt.dt["ene"], bins=500)
plt.plot(sig_hist[0])
plt.xticks(ticks=np.linspace(0,500,1),label=np.linspace(0,1000,5))
#%% background --------------------------------------------------------

source_ = 'background'
distance = 'indoor'
csv_file = f"../../../Data/240603_nucare/ori/{source_}_{distance}.csv"
back_dt = modi_hist_extract(csv_file)

back_dt.dt["ene"] = round(back_dt.dt[" total_energy (keV)"] / 1000)

import matplotlib.pyplot as plt

back_hist = np.histogram(back_dt.dt["ene"], bins=500)
plt.plot(back_hist[0])
plt.xticks(ticks=np.linspace(0,500,1),label=np.linspace(0,1000,5))

#%% save the test file --------------------------------------------------------

pois_test = np.savez('./test/pois_test.npz', sig=sig_dt, back=back_dt)



# * ======================================================================================
# * ======================================================================================
#%% 진짜 시작 === --------------------------------------------------------

# 1개의 t 에 대해서만.
sig_dt.filtered(1, 2)
back_dt.filtered(1, 2)

sig_hist = np.histogram(sig_dt.filtered_dt["ene"], bins=250)
back_hist = np.histogram(back_dt.filtered_dt["ene"], bins=250)

print("total sig : ", sum(sig_hist[0]))
print("total background : ", sum(back_hist[0]))

#%% another for t-1 --------------------------------------------------------
sig_dt.filtered(0, 1)
sig_t_1_hist = np.histogram(sig_dt.filtered_dt["ene"], bins=250)

figureon = 0
if figureon:
    #plt.figure(figsize=(20, 6))
    plt.plot(sig_t_1_hist[0])
    plt.title("bins with 100")

#%% background estimation --------------------------------------------------------
alpha = 0.02
xt = sig_hist[0] # x(t)
xt_1 = sig_t_1_hist[0] # x(t-1)
xt_l1norm = sum(abs(xt)) # x(t) l1norm
xt_1_l1norm = sum(abs(xt_1)) # x(t-1) l1norm
ut_1_hat = 0 # u(t-1) hat

ut_hat = (1 - alpha) * ut_1_hat + alpha * xt_1 / xt_1_l1norm

figureon = 1
if figureon:
    plt.plot(ut_hat)
    plt.title("ut hat estimation")


#%% norm 구하기 --------------------------------------------------------
sig_hist = sig_hist[0]
back_hist = back_hist[0]

back_hist_l1norm = back_hist / sum(back_hist)
back_hist_l2norm = np.linalg.norm(back_hist)
#%% 변수명으로 교체해버리기 --------------------------------------------------------

# 우리는 background가 있으니까 estimation 없이 사용해도 될듯
ut = back_hist
ut_hat = back_hist_l1norm
xt = sig_hist
xt_l1norm = sum(abs(xt))

epsilon = 1e-5 # 0값이 있는경우는 오류나서..
dt = 2*(xt * np.log(xt / (xt_l1norm * ut_hat + epsilon) + epsilon)) - (xt - (xt_l1norm * ut_hat))

#%% --------------------------------------------------------
plt.plot(xt_l1norm * ut_hat, label='weight of background as signal')
plt.plot(xt, label='signal spectrum')
plt.plot(ut, label='background')
plt.legend()
#%% --------------------------------------------------------
plt.plot(dt, label="Poisson Deviance")
plt.legend()

#%% 민은기 박사님 background 제거 방식 --------------------------------------------------------
tc = (xt-ut)**2 / (xt+epsilon)
plt.plot(tc, label='background removed spectrum')
plt.legend()











#* 240704
# Part 1. Data 준비
#%% #* ===================================================================
# 그럼 이제 2D (모든 t에 관해서) 진행 
# 1. --------------------------------------------------------
import numpy as np
from joblib import Parallel, delayed, cpu_count
from modi_hist_extract import modi_hist_extract

index = [i for i in range(len(source))]   # source list에서 뽑을 index

source_ = 'ba133'
distance = 'close'    
time = '5min'

rowsize = 300 # 이미지의 row 개수

starttime = 1
finaltime = 299 # data의 마지막 시간   
interval = (finaltime-starttime) / (rowsize-1)

# data 불러오기
csv_file = f"../../../Data/240603_nucare/ori/{source_}_{distance}_{time}.csv"
fil_dt = modi_hist_extract(csv_file)  # filtered data

fil_dt.dt["ene"] = round(fil_dt.dt[" total_energy (keV)"] / 1000)



# endtime 별 해당하는 histogram row 1개씩 뽑기
accumulate = 0

# debug 용
debug_counts = []

endtime_values = np.linspace(starttime+interval, finaltime, rowsize, endpoint=True)
# 그냥 fix해서 사용
endtime_values = np.linspace(1, 300, 300, endpoint=True)
#%% --------------------------------------------------------
# 2.
def onestack_histimage(startidx, endidx):
    global endtime_values
         
    # filter할 time 구간지정
    previous_time = endtime_values[startidx]
    if endidx > rowsize-1: # row(시간)이 끝 이후라면..
        return np.zeros(250)
    endtime = endtime_values[endidx]
    
    # 누적상황을 보여주고 싶으면 filter time 구간 처음은 무조건 0
    if accumulate: previous_time = 0
    
    # filter 진행
    fil_dt.filtered(previous_time, endtime)
    output = np.histogram(fil_dt.filtered_dt["ene"], bins=250)
    
    return output[0]
    
# starttime ~ finaltime 사이를 rowsize 간격으로 나누어서 누적상태로 각 row에 저장.

results = Parallel(n_jobs=cpu_count(), verbose=10)(delayed(onestack_histimage)(i, i+1) for i in range(len(endtime_values)))
results = np.array(results)

#%% --------------------------------------------------------
# (not essential) 3. for background
source_ = 'background'
distance = 'close'    
time = '10min'

csv_file = f"../../../Data/240603_nucare/ori/{source_}_{distance}_{time}.csv"
fil_dt = modi_hist_extract(csv_file)  # filtered data
fil_dt.dt["ene"] = round(fil_dt.dt[" total_energy (keV)"] / 1000)

def onestack_histimage(startidx, endidx):
    global endtime_values
         
    # filter할 time 구간지정
    previous_time = endtime_values[startidx]
    if endidx > rowsize-1: # row(시간)이 끝 이후라면..
        return np.zeros(250)
    endtime = endtime_values[endidx]
    
    # 누적상황을 보여주고 싶으면 filter time 구간 처음은 무조건 0
    if accumulate: previous_time = 0
    
    # filter 진행
    fil_dt.filtered(previous_time, endtime)
    output = np.histogram(fil_dt.filtered_dt["ene"], bins=250)
    
    return output[0]
    
# starttime ~ finaltime 사이를 rowsize 간격으로 나누어서 누적상태로 각 row에 저장.

results_2 = Parallel(n_jobs=cpu_count(), verbose=10)(delayed(onestack_histimage)(i, i+1) for i in range(len(endtime_values)))
results_2 = np.array(results_2)

#%% (extra) 4. 2개의 신호 이어 붙이기 ----------------------------------------------------
# 1번 실행 - 4번 실행 - 다시 다른 source로 1번 실행
#result_cs137 = results
#result_ba133 = results

#%% 4-2.
results = np.zeros([300, 250])
results[0:150,:] = result_cs137[0:150]
results[150:,:] = result_ba133[150:]


#%% saved file  --------------------------------------------------------
np.savez("./pois_test2d_sigba133.npz", sig=results, back=results_2)











# Part 2. Data Processing in 2D
#%% Now we start again in 2D ============================================================
# 0. measured counts spectrum
import matplotlib.pyplot as plt

plt.imshow(results, cmap='turbo', vmin=0, vmax=200)
plt.colorbar()
# figure settings
plt.xlabel("Energy bin")
plt.ylabel("Time(s)")
pixelsize = 50

bins = 50
x_ticks = [i*bins for i in range(0,6)]
x_labels = [4*i*bins for i in range(0,6)]
plt.xticks(ticks=x_ticks, labels=x_labels)

y_ticks = [i*pixelsize for i in range(0,6)]
y_labels = [(finaltime/299)*pixelsize*i for i in range(0,6)]
plt.yticks(ticks=y_ticks, labels=y_labels)

if accumulate :
    text = "Accumulate On"
else:
    text = "Accumulate Off"
#plt.text(100, 50, text, fontsize=12, color='blue')

#%% 0-2. total count
grosscount = np.zeros(300)
for i in range(300):
    grosscount[i] = sum(results[i])
plt.plot(grosscount)
plt.title("Gross Count")
plt.xlabel("TIme(s)")
plt.ylabel("Gross count")


#%% -------------------------------------------------------------
# 1. background estimation
x = results
back_est = np.zeros([300, 250])  # ut ( 0 <= t <= 300)
alpha = 0.2
epsilon = 1e-5

for i in range(300):
    if i==0:
        back_est[i] = np.mean(x, axis=0)    
        continue
    back_est[i] = (1-alpha)*back_est[i-1] + alpha*x[i-1]/sum(abs(x[i-1]))


plt.imshow(back_est, cmap='turbo', vmax=20)
plt.title("Normalize Estimated Background")
plt.colorbar()
# figure settings
plt.xlabel("Energy bin")
plt.ylabel("Time(s)")

bins = 50
x_ticks = [i*bins for i in range(0,6)]
x_labels = [4*i*bins for i in range(0,6)]
plt.xticks(ticks=x_ticks, labels=x_labels)

pixelsize = 50
y_ticks = [i*pixelsize for i in range(0,6)]
y_labels = [(finaltime/299)*pixelsize*i for i in range(0,6)]
plt.yticks(ticks=y_ticks, labels=y_labels)


#%% -------------------------------------------------------------------
# 1-2. signal and background
timenum = 250
plt.title(f"{timenum} sec")
plt.plot(x[timenum],label="signal")
plt.plot(back_est[timenum]*sum(abs(x[timenum])),label="estimated background")
plt.legend()
#%% -------------------------------------------------------------------
# 2. deviance poisson
epsilon = 1e-5 # 0값이 있는경우는 오류나서..
pos_dev = np.zeros([300, 250])  # dt - Poisson Deviance,  row:times, col:bins
for t in range(300):
    pos_dev[t] = 2*(x[t]*np.log(x[t] / (sum(abs(x[t]))*back_est[t] + epsilon) + epsilon)) - (x[t]-(sum(abs(x[t]))*back_est[t]))

plt.imshow(pos_dev, cmap='turbo', vmin=0, vmax=200)  
plt.title("Poisson Deviance")
plt.colorbar()
# figure settings
plt.xlabel("Energy bin")
plt.ylabel("Time(s)")

bins = 50   # 초기에 정한 histogram의 bins 수를 기준으로, 총 표시할 y label 수로 나눈 값
x_ticks = [i*bins for i in range(0,6)]
x_labels = [4*i*bins for i in range(0,6)]
plt.xticks(ticks=x_ticks, labels=x_labels)

pixelsize = 50
y_ticks = [i*pixelsize for i in range(0,6)]
y_labels = [(finaltime/299)*pixelsize*i for i in range(0,6)]
plt.yticks(ticks=y_ticks, labels=y_labels)

#%% ----------------------------------------------------------------------
# 2-2. Total Deviance poisson
tot_pos_dev = np.zeros(300)
for i in range(300):
    tot_pos_dev[i] = sum(pos_dev[i])
plt.title("Total Poisson Deviance")
plt.plot(tot_pos_dev)
#%% 확대
plt.plot(tot_pos_dev[100:])
plt.xticks(ticks=[i*25 for i in range(9)], labels=[25*i+100 for i in range(9)])


#%% ---------------------------------------------------------------------------
# 3. weighting vector
beta = 0.5

dt = pos_dev
dt_l1norm = np.sum(abs(dt), axis=1)
n = 250
epsilon = 1e-5

# time : 300, bin : 250
wt = np.zeros([300, 250])  
for i in range(300):
    if i == 0:
        wt[i] = np.ones(250)  # n = bins, initialize in ones
        continue
    wt[i] = (1 - beta)*wt[i-1] + (beta*n)*dt[i] / (dt_l1norm[i]+epsilon)

#%% 3-2. weighting vector plot
plt.imshow(wt, cmap='turbo', vmax=200)  
plt.title("Weighted Poisson Deviance")
plt.colorbar()
# figure settings
plt.xlabel("Energy bin")
plt.ylabel("Time(s)")

bins = 50   # 초기에 정한 histogram의 bins 수를 기준으로, 총 표시할 y label 수로 나눈 값
x_ticks = [i*bins for i in range(0,6)]
x_labels = [4*i*bins for i in range(0,6)]
plt.xticks(ticks=x_ticks, labels=x_labels)

pixelsize = 50
y_ticks = [i*pixelsize for i in range(0,6)]
y_labels = [(finaltime/299)*pixelsize*i for i in range(0,6)]
plt.yticks(ticks=y_ticks, labels=y_labels)

#%%
# 2-3. Total Weighted Deviance poisson
tot_weighted_pos_dev = np.zeros(300)
for i in range(300):
    tot_weighted_pos_dev[i] = sum(wt[i])
plt.title("Total Weighted Poisson Deviance")
plt.plot(tot_weighted_pos_dev)
plt.xlabel("Time(s)")
plt.ylabel("Weighted deviance with l-1 norm")














































#%% #* ============================= Poisson example ==================================
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

import matplotlib.pyplot as plt

plt.plot(X)
plt.plot(y)
#%% 
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd

# 예제 데이터 생성
np.random.seed(123)
n = 100
data = pd.DataFrame({
    'x1': np.random.normal(size=n),
    'x2': np.random.normal(size=n),
    'y': np.random.poisson(lam=1.5, size=n)
})

# 포아송 회귀 모델 적합
model = smf.poisson('y ~ x1 + x2', data=data).fit()

# 예측값 계산
predictions = model.predict(data)

# 실제값과 예측값을 이용하여 포아송 deviance 계산
actual = data['y']
poisson_deviance = 2 * np.sum(actual * np.log(actual / predictions) - (actual - predictions))

print('Poisson Deviance:', poisson_deviance)














































































#%% #* ===================================================================
#*                              Filters!
#* =======================================================================
# %% #* EWMA =======================================================================
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
plt.plot(ewma_values, label='EWMA_values')
plt.plot(sig, label='original', alpha=0.5)
plt.legend()
plt.title('Exponential weighted moving average')




# %% #* boxcar filter =======================================================================
import math
w = 7
boxcar_values = np.zeros(1000)
wrange = math.floor((w-1)/2)

sig_ = np.pad(sig, pad_width = wrange, mode='constant', constant_values=0)

def boxcar(sig_, w, idx):
    return (1/w) * sum([sig_[tt+wrange] for tt in range(idx-wrange, idx+wrange+1)])
        
for t in range(0,1000):
    boxcar_values[t] = boxcar(sig_, w, t)

# %%
plt.plot(sig, label='original spectrum', color='red')
plt.plot(boxcar_values, label='filtered')
plt.legend()
plt.title("Boxcar Filtering")
# %%











#%% #* Gaussian fitting =======================================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 가우시안 함수 정의
def gaussian(x, a, b, c):
    return a * np.exp(-(x - b)**2 / (2 * c**2))

# 예제 데이터 생성
x_data = np.linspace(0,1000,1001)
y_data = sig

# 가우시안 피팅
initial_guess = [1, 0, 1]  # 초기 추정값
params, covariance = curve_fit(gaussian, x_data, y_data, p0=initial_guess)

# 피팅 결과 파라미터
a_fit, b_fit, c_fit = params
print(f'Fitted parameters: a={a_fit}, b={b_fit}, c={c_fit}')

# 피팅된 가우시안 함수 계산
y_fit = gaussian(x_data, a_fit, b_fit, c_fit)

# 데이터와 피팅 결과 플롯
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, label='Data', color='blue', s=10)
plt.plot(x_data, y_fit, label='Gaussian Fit', color='red')
plt.title('Gaussian Fitting')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()



















# %% #*second derivative =======================================================================
import numpy as np

def second_derivative(f, x, h=1e-5):
    return (f(x + h) - 2 * f(x) + f(x - h)) / h**2









#%% #* Savitzky-Golay =======================================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

np.random.seed(0)

# Savitzky-Golay 필터 적용
window_length = 31  # 윈도우 길이는 홀수여야 함
polyorder = 3  # 다항식 차수

normalized_sig = sig / np.max(sig)
y_smooth = savgol_filter(normalized_sig, window_length, polyorder)

# 결과 시각화
plt.figure(figsize=(10, 6))
plt.plot(normalized_sig, label='Noisy data')
plt.plot(y_smooth, label='Smoothed data', color='red')
plt.title('Savitzky-Golay Smoothing Filter')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()












# %% #*=======================================================================
# using MODWT
from dataprocess import *
import tensorflow as tf
import sys
sys.path.append('./')


def noisefiltering_dd(data:np.array, outlayer, shift):
    layer = 6
    wavelet = 'haar'
    output = np.zeros(data.shape)
    print("output shape : ", output.shape)
    coef = np.zeros(data.shape, )
    for i in range(data.shape[0]):
        '''major signal extraction by using modwt method'''
        coefficient = modwt(data[i], wavelet, layer) # layer만큼 행이 나옴.
        output[i] = imodwt(coefficient[layer-outlayer:layer+1,:],wavelet)
        output[i] = np.roll(output[i],shift)
        '''scaling for preservation of signal data'''
        max_val_out = np.max(output[i])
        output[i] = output[i] / max_val_out
        '''thresholding unavailable data'''
        output[i][output[i]<0] = 0
    return output


sig_ = sig[tf.newaxis,:]
normalized_sig = sig_ / (np.max(sig_))
filtered1 = noisefiltering_dd(normalized_sig, 0, -32)
derivatives1 = derivative_signal(filtered1)
filtered2 = noisefiltering2(derivatives1, 0, 0) 

plt.figure(figsize=(10, 6))
plt.plot(normalized_sig[0], label='Noisy data')
plt.plot(filtered1[0], label='modwt sig')
plt.plot(derivatives1[0], label='deravatives')
plt.plot(filtered2[0], label='filtered 1', color='red')
plt.title('modwt')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()








# %% #* possion devaince ====================================================
def likelihood(beta1, beta2):
   pass 





#%% learning likelihood , Thanks for REF! : https://studyingrabbit.tistory.com/66
# 70, 30
from scipy.special import factorial
from scipy import signal
import matplotlib.pyplot as plt

pw = np.power

def log_test(ph):
    return np.log(factorial(100)/(factorial(30)*factorial(70))*pw(ph,70)*pw(1-ph,30))

ph = np.linspace(0,1,1000)
y = np.zeros(1000)
for i in range(len(ph)):
    y[i] = log_test(ph[i])
plt.plot(y)
plt.xticks(ticks=np.linspace(0,1000,5), labels=np.linspace(0,1,5))
plt.xlabel("ph")
plt.grid()
# %%
