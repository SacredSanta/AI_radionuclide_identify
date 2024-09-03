'''
최종 수정 : 2024.06.23.
사용자 : 서동휘

<수정 내용> 



<24.08.06>
- 이전 data (.xz) 와
최신 nucare data (.csv) 각각 data 를
poisson deviance 를 통해 background estimation 하는 것 연습
- poisson def 생성
- (중요!) data 생성이 아닌, 1단계 preprocessing 부분에서 같이 pos_dev 를 첨가해주는 식으로
  data를 구성하면 될듯함.


<처음>
poisson deviance 구현을 위한 코드
Kai Vetter 논문을 참고하여 생성.
Nucare 데이터를 기반으로 연습.
'''
import numpy as np

# xt : spectrum signal (0x1000)
# ut_hat : normalized background spectrum (0x1000)
# return : dt (0x1000)
def my_pos_dev(xt, ut_hat):
    epsilon = 1e-5
    return ( 2 * (xt * np.log(xt / (sum(abs(xt)) * ut_hat + epsilon) + epsilon))
                - (xt - (sum(abs(xt)) * ut_hat)))




if __name__ == "__main__":


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
    #%% 시작 === --------------------------------------------------------

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











    #* 240704 최신버전
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














































    #%% ====================================================================================
    # 24.08.06

    import lzma

    import pickle
    import numpy as np
    import matplotlib.pyplot as plt

    # [1] xz data에서..
    # [1] 1. xz data 불러오기


    RA = "Background"

    dtpath = f"../../../Data/230000_forTensorflow_xz/{RA}"

    num = np.random.randint(1, 400, size=1)
    with lzma.open(dtpath+"/{}_{}.xz".format(RA, num[0]), 'rb') as f:
        ut = pickle.load(f)

        
    plt.plot(ut, color='green')
    plt.title(f"{num} file ut")
    plt.xlabel("Energy bin(keV)")
    plt.ylabel("Count")

    #%%
    # [1] 2. xt 불러오기

    RA = "Ba133"

    dtpath = f"../../../Data/230000_forTensorflow_xz/{RA}"

    num = np.random.randint(1, 400, size=1)
    with lzma.open(dtpath+"/{}_{}.xz".format(RA, num[0]), 'rb') as f:
        xt = pickle.load(f)

        
    plt.plot(xt, color='purple')
    plt.title(f"{num} file xt")
    plt.xlabel("Energy bin(keV)")
    plt.ylabel("Count")
    plt.text(1080, max(xt), f"{RA}", fontweight='bold', color='blue')


    #%%
    # [1] 3. 원래 ut 를 normalize 해서 xt와 곱해주어서 ut를 구하는게 정석이지만
    # 우리는 이미 background 자체의 data를 지니고 있기에 바로 dt를 구해본다.

    epsilon = 1e-5
    #for t in range(300):
    pos_dev = ( 2 * (xt * np.log(xt / (sum(abs(xt)) * ut + epsilon) + epsilon))
                    - (xt - (sum(abs(xt)) * ut))
            )

    plt.plot(pos_dev, color='gray')
    plt.title("Poisson deviance")
    plt.xlabel("Energy bin(keV)")
    plt.ylabel("Poisson unit deviance")
    plt.text(1080, max(pos_dev),"ut did not normalized", fontweight='bold', color='blue')

    #%%
    # [1] 3-2 그렇다면 ut를 normalize 하면?
    ut_hat = ut/sum(abs(ut))
    pos_dev1_2 = ( 2 * (xt * np.log(xt / (sum(abs(xt)) * ut_hat + epsilon) + epsilon))
                    - (xt - (sum(abs(xt)) * ut_hat))
            )

    plt.plot(pos_dev1_2, color='gray')
    plt.title("Poisson deviance")
    plt.xlabel("Energy bin(keV)")
    plt.ylabel("Poisson unit deviance")
    plt.text(1080, max(pos_dev1_2),"ut normalized", fontweight='bold', color='red')


    #%%
    # [1] 3-3 background 평가 상태가 어떠한지

    plt.plot(xt, label='signal', color='purple')
    plt.plot((sum(abs(xt)) * ut_hat), label='background estimation', alpha=0.7, color='green')
    plt.legend()
    plt.title("Signal Spectrum")
    plt.xlabel("Energy bin(keV)")
    plt.ylabel("Counts")






    #%%  ------------------------------------------------------------------------------
    # [2] .csv 데이터에서..
    # [2]-1 xt2 불러오기 - 위치 보정 해주어야되서

    import sys
    import os

    sys.path.append(os.path.join(os.getcwd(), "lib"))
    print(sys.path)

    from modi_hist_extract import modi_hist_extract

    RA = "ba133"

    dtpath = f"../../../Data/240603_nucare/ori/{RA}_close_5min.csv"

    xt2_dt = modi_hist_extract(dtpath)
        
    # plt.plot(xt2_dt.hist)
    # plt.title(f"{num} file ut")
    # plt.xlabel("Energy bin(keV)")
    # plt.ylabel("Count")
    xt2_dt.show()


    #%%
    # [2]-1-2 위치보정..   #! 반드시 한번만 실행하여야됨!!!!!!!!!!
    xt2_dt.find_peak([170, 250])
    xt2_dt.fix_data(356)
    xt2_dt.show()



    #%%
    # [2]-1-3 시간 추출 (1초)
    xt2_dt.filtered(7,7.3)
    xt2 = xt2_dt.filtered_hist
    print("xt2 count : {}".format(sum(xt2)))


    #%%
    # [2]-2 background 불러오기 
    RA = "background"

    dtpath = f"../../../Data/240603_nucare/ori/{RA}_close_10min.csv"

    ut2_dt = modi_hist_extract(dtpath)
    
    ut2_dt.show()    
    #plt.plot(ut2_dt.hist)
    #plt.title(f"{num} file ut")
    #plt.xlabel("Energy bin(keV)")
    #plt.ylabel("Count")

    #%% 
    # [2]-2-2 위치보정 
    # xt2 에서 얻은 scale factor 이용해서 보정
    ut2_dt.fix_data_with_scalefactor(xt2_dt.scale_factor)
    ut2_dt.show()

    ut2 = ut2_dt.hist


    #%%
    # [2]-3 poisson deviance
    epsilon = 1e-5
    ut2_hat = ut2/sum(abs(ut2))
    pos_dev2 = ( 2 * (xt2 * np.log(xt2 / (sum(abs(xt2)) * ut2_hat + epsilon) + epsilon))
                    - (xt2 - (sum(abs(xt2)) * ut2_hat))
            )

    plt.plot(pos_dev2, color='gray')
    plt.title("Poisson deviance")
    plt.xlabel("Energy bin(keV)")
    plt.ylabel("Poisson unit deviance")
    plt.text(1080, max(pos_dev2),"ut normalized", fontweight='bold', color='red')

    #%%
    # [2] 3-3 background 평가 상태가 어떠한지

    plt.plot(xt2, label='signal', color='purple')
    plt.plot((sum(abs(xt2)) * ut2_hat), label='background estimation', alpha=0.7, color='green')
    plt.legend()
    plt.title("Signal Spectrum")
    plt.xlabel("Energy bin(keV)")
    plt.ylabel("Counts")
    #plt.text(0, -4, 
    #         "info : 2 sec / Cs137 / close / total count = {}".format(sum(xt2)),
    #         fontweight='bold',
    #         color='black')

    #%%
    # [2] 나머지 sig 랑 background 도 각각 뽑기
    plt.plot(xt2, color='purple')
    plt.title("xt2")
    plt.xlabel("Energy bin(keV)")
    plt.ylabel("Counts")
    #%%
    plt.plot((sum(abs(xt2)) * ut2_hat), color='green')
    plt.title("background estimation")
    plt.xlabel("Energy bin(keV)")
    plt.ylabel("Counts")












































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
