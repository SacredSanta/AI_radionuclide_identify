#%% setting xticks
plt.xticks(ticks=np.linspace(0,1000,5), labels=np.linspace(0,1,5))





#%% imshow
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














#%% Parallel
from joblib import Parallel, delayed, cpu_count
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