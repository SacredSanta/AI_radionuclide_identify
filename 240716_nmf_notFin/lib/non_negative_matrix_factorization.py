#%% ---------------------------------------------------------------
# -                      1.PCA                                   -
# -----------------------------------------------------------------
#%%
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# %%
np.random.seed(0)
X = np.random.rand(10, 5)  # 10개의 샘플과 5개의 특성

# PCA 모델 생성 (주성분의 수를 2로 설정)
pca = PCA(n_components=5)

# PCA 적용
X_pca = pca.fit_transform(X)

# 결과 출력
print("원본 데이터:\n", X)
print("PCA 변환된 데이터:\n", X_pca)

# 주성분 비율
print("설명된 분산의 비율:", pca.explained_variance_ratio_)

# 주성분을 시각화
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA result')
plt.show()





# %% ------------------------------------------------------------

D1 = np.array([170,150,160,180,170]).reshape(5,1)
D2 = np.array([70, 45, 55, 60, 80]).reshape(5,1)
D = np.hstack((D1, D2))
# %%
D[:,0] = D[:,0] - np.mean(D1)
D[:,1] = D[:,1] - np.mean(D2)
# %%
n = len(D1)
SIGMA = (D.transpose() @ D) / n
print(SIGMA)

# 고유값과 고유벡터 계산
eigenvalues, eigenvectors = np.linalg.eig(D)

print("고유값:")
print(eigenvalues)

print("고유벡터:")
print(eigenvectors) # 한개의 column이 하나의 벡터


# %% 
mag = np.sqrt(np.sum(eigenvectors[:,0]**2))
print(mag)
# unit 크기로 주는지 확인







# %%


















#%% ---------------------------------------------------------------
# -                      2.NMF practice                           -
# -----------------------------------------------------------------

#%% --------------------------------------------------------
# 1. NMF practice
import numpy as np
from sklearn.decomposition import NMF

# 예제 데이터 생성
V = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# NMF 모델 생성
n_components = 2  # W와 H의 랭크 설정
model = NMF(n_components=n_components, init='random', random_state=0)

# NMF 수행
W = model.fit_transform(V)  # learn a NMF model for the data X and return transformed data
H = model.components_ # factorization matrix (dictionary?)

# 결과 출력
print("원본 행렬 V:")
print(V)
print("\nW 행렬:")
print(W)
print("\nH 행렬:")
print(H)

# 근사 행렬 계산
V_approx = np.dot(W, H)
print("\n근사 행렬 V_approx:")
print(V_approx)
# %% 2. 예제
from PIL import Image
import matplotlib.pyplot as plt
img = np.array(Image.open("./test/img3.jpg"))

V = (1/3)*img[:,:,0] + (1/3)*img[:,:,1] + (1/3)*img[:,:,2]
rank = 2
model = NMF(n_components=2, init='random', random_state=0)
W = model.fit_transform(V)
H = model.components_
V_approx = np.dot(W, H)
reconstructions.append(V_approx)

plt.subplot(1,3,1)
plt.imshow(V)
plt.subplot(1,3,2)
plt.imshow(W)
plt.subplot(1,3,3)
plt.imshow(H)

#%% 3. 직접 NMF 구현
from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import time


img = np.array(Image.open("./test/img3.jpg"))

V = (1/3)*img[:,:,0] + (1/3)*img[:,:,1] + (1/3)*img[:,:,2]

m = V.shape[0]
n = V.shape[1]

p = 408 # the number of features

W = np.random.rand(m, p)*255
H = np.random.rand(p, n)*255

n_epoch = 200

X = V

for i in range(n_epoch):
    H = H * ( (W.transpose()@X) / (W.transpose()@W@H) )  # W:408x408  X:408x612      H:408x612
    W = W * ( (X@H.transpose()) / (W@(H@H.transpose())) )
    
    plt.imshow(H)
    plt.title(f"Iteration : {i}")
    
    display(plt.gcf())
    clear_output(wait=True)
    #time.sleep(0.1)



#%%
plt.subplot(2,1,1)
plt.imshow(H)
plt.subplot(2,1,2)
plt.imshow(W)







#%% ---------------------------------------------------------------
# -                      3.KL divergence practice                 -
# -----------------------------------------------------------------

#%% KL divergence  ----------------------------------------------------------------
import numpy as np
from scipy.special import rel_entr

# 두 개의 확률 분포 정의
P = np.array([0.1, 0.4, 0.5])
Q = np.array([0.3, 0.4, 0.3])

# KL divergence 계산
kl_divergence = np.sum(rel_entr(P, Q))
print(f"KL Divergence: {kl_divergence}")










#%% ---------------------------------------------------------------
# -                      4.NMF 1d signal에 적용                    -
# -----------------------------------------------------------------

#%% 0. get data  ----------------------------------------------------------------
from modi_hist_extract import *

source_ = 'ba133'
distance = 'close' 
csv_file = f"../../../Data/240603_nucare/ori/{source_}_{distance}_5min.csv"

sig_dt = modi_hist_extract(csv_file)
plt.plot(sig_dt.hist)

pixelsize=100
x_ticks = [i*pixelsize for i in range(0,10)]
x_labels = [(1000/1000)*pixelsize*i for i in range(0,10)]
plt.xticks(ticks=x_ticks, labels=x_labels)
# %% 1. estimate approximation of X ----------------------------------------------------------------

import scipy

# A : weights encoding data
# V : set of k basis vectors spanning a subspace of Rd

# k 값 으로 dimension을 떨굼

# NMF 를 통해서 matrix X를  A와 V의 관점에서 추정해본다. - 이 모든 3개의 matrix는 non-negative하다.

# 1. k 무조건 특정되어야함. (PCA와는 반대)

# x_ij : measured data in X
# x^_ij : low-rank approximation in X^ 
#   though as 스펙트럼마다, energy bin마다 mean rate 로 간주가능

#  (2) 식을 최대화하는 것은 X~X^ 사이의 KL divergence를 최소화하는 것과 같다.
 
from joblib import Parallel, delayed, cpu_count

X = np.random.randint(100, size=1000).reshape(1000,1)
def cal_log_likelihood(k):    
    global X
    A = np.random.rand(1000, k)
    V = np.random.rand(k, 1)

    X_hat = A@V
    result = 0

    for i in range(0,1000):
        for j in range(0,1):
            ans = -X_hat[i,j] + X[i,j]*np.log(X_hat[i,j]) - scipy.special.factorial(np.log(X_hat[i,j]))
            result += ans
    return result

results = Parallel(n_jobs=cpu_count(), verbose=10)(delayed(cal_log_likelihood)(k) for k in range(0,1000))
results = np.array(results)

#print(results)  # --> ln P(X|X^)
plt.plot(results)
plt.xlabel("k")
plt.ylabel("ln P(X|X^)")
plt.axhline(y=0, color='r', linestyle='--', label='y=0')
plt.axvline(x=538, color='b', linestyle='--', label='x=538')
plt.legend()
#np.where(abs(results)<17)



#%% 2. PNMF iterative update rules ----------------------------------------------------------------

#X = np.random.randint(100, size=1000).reshape(1000,1)
X = sig_dt.hist[:1000, np.newaxis]

k = 50
m = 1000
n = 1
A = np.random.rand(m, k)
V = np.random.rand(k, n)

X_hat = A@V

# A : n x k
# V : k x d

# i : 1 ~ n
# j : 1 ~ d 
# l : then l will be k

#%% 3. iterate setting  ----------------------------------------------------------------
iter_num = 10_000

# a_il <- from A
# v_lj <- from V

# numpy를 통해 한번에 계산할 수 있겠지만, 수식이 좀 복잡해서 일단 노가다로 쌩구현


#%%---------------------DEPRE!!-------------------------------
#%% starts - prototype  
while iter_num > 0:
    if iter_num%100 == 0: print(iter_num, " left")
    if iter_num%1000 == 0:
        print(iter_num, " left")
        print("keep iterate?")
        if input() == 0:
            break
        else:
            pass
    iter_num -= 1
    # 1. start iterative
    # (3) update a_il
    for i in range(m):
        for l in range(k):
            temp_numer = 0
            temp_deno = 0
            for j in range(n):
                temp_numer += (V[l,j]*X[i,j])/ (X_hat[i,j]+1e-5)
                temp_deno += V[l,j]
            A[i,l] = A[i,l] * (temp_numer/temp_deno+1e-5)       
            
            

    # (4) update v_lj
    for l in range(k):
        for j in range(n):
            temp_numer = 0
            temp_deno = 0
            for i in range(m):
                temp_numer += (A[i,l]*X[i,j]) / (X_hat[i,j]+1e-5)
                temp_deno += A[i,l]
            V[l,j] = V[l,j] * (temp_numer/temp_deno+1e-5)
            

    # (5) update v_lj
    for l in range(k):
        temp_nor = sum(V[l,:])
        for j in range(n):
            V[l,j] = V[l,j] / (temp_nor+1e-5)
            
    # (6) update a_ij
    for i in range(m):
        for l in range(k):
            temp_mul = sum(V[l,:])
            A[i,l] = A[i,l] * temp_mul


    # 2. L2NMF 
    XV_T = X @ (V.transpose())
    AVV_T = A @ V @ V.transpose()
    A_TX = (A.transpose()) @ X
    A_TAV = (A.transpose()) @ A @ V


    #(1)
    for i in range(m):
        for l in range(k):
            A[i,l] = A[i,l] * ( (XV_T[i,l]) / (AVV_T[i,l]+1e-5) )

    #(2)
    for l in range(k):
        for j in range(n):
            V[l,j] = V[l,j] * ( (A_TX[l,j]) / (A_TAV[l,j]+1e-5) )

    #(3)
    for l in range(k):
        temp_vl = sum(V[l,:])
        for j in range(n):
            V[l,j] = V[l,j] / (temp_vl+1e-5)

    #(4)
    for i in range(m):
        for l in range(k):
            temp_vl = sum(V[l,:])
            A[i,l] = A[i,l]*temp_vl
----------------------------------------------------------------



#%% save results ----------------------------------------------------------------
np.savez("./nmf_data.npz", X=X, A=A, V=V, k=k, m=m, n=n, info="iterated 10000")





#%% 4. 최적화 version   ----------------------------------------------------------------
#X = sig_dt.hist[np.newaxis, :1000]

def my_nmf(X):
    k = 3
    m = X.shape[0]
    n = X.shape[1]
    A = np.random.rand(m, k)
    V = np.random.rand(k, n)

    X_hat = A@V  # m x n

    iter_num = 30000

    while iter_num > 0:
        if iter_num%100 == 0: print(iter_num, " left")
        # if iter_num%1000 == 0:
        #     print(iter_num, " left")
        #     print("keep iterate?")
        #     if input() == 0:
        #         break
        #     else:
        #         pass
        iter_num -= 1
        
        # (3)
        temp_numer = np.dot(X / (X_hat + 1e-5), V.T)
        temp_deno = np.sum(V, axis=1) # column
        temp_deno = temp_deno[np.newaxis, :]
        A *= (temp_numer / (temp_deno + 1e-5))

        # (4)
        temp_numer = np.dot(A.T, X / (X_hat + 1e-5))
        temp_deno = np.sum(A, axis=0) # row
        temp_deno = temp_deno[:, np.newaxis]
        V *= (temp_numer / (temp_deno + 1e-5))
        
        # (5)
        temp_nor =  np.sum(V, axis=1)
        V = V / (temp_nor[:, np.newaxis] + 1e-5)
        
        # (6)
        temp_mul = np.sum(V, axis=1)
        A *= temp_mul


        # (8). L2NMF 
        XV_T = X @ V.T
        AVV_T = A @ V @ V.T
        A_TX = A.T @ X
        A_TAV = A.T @ A @ V

        # (8-1)
        A *= (XV_T / (AVV_T + 1e-5))

        # (8-2)
        V *= (A_TX / (A_TAV + 1e-5))
        
        # (8-3)
        temp_vl = np.sum(V, axis=1)
        V = V / (temp_vl[:, np.newaxis] + 1e-5)
        
        # (8-4)
        temp_vl = np.sum(V, axis=1)
        A *= temp_vl[np.newaxis, :]
        
    return X_hat, A, V

#%% 5. plot the result (A) ----------------------------------------------------------------

plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
plt.plot(V[0,:], linestyle=':', label='row=0')
plt.plot(V[1,:], linestyle=':', label='row=1')
plt.plot(V[2,:], linestyle=':', label='row=2')

plt.legend()
plt.title('Three(k=3) Poisson NMF basis vectors V')
plt.xlabel("Energy (keV)")
plt.ylabel("NMF Component(a.u.)")

plt.subplot(2,2,2)
plt.plot(V[0,:], color='blue')
plt.title('row=0')
plt.xlabel("Energy (keV)")
plt.ylabel("NMF Component(a.u.)")

plt.subplot(2,2,3)
plt.plot(V[1,:], color='orange')
plt.title('row=1')
plt.xlabel("Energy (keV)")
plt.ylabel("NMF Component(a.u.)")

plt.subplot(2,2,4)
plt.plot(V[2,:], color='green')
plt.title('row=2')
plt.xlabel("Energy (keV)")
plt.ylabel("NMF Component(a.u.)")

#%% 6. original signal ----------------------------------------------------------------
plt.plot(X[0,:], color='blue')
plt.title('Original Signal')
plt.xlabel("Energy (keV)")
plt.ylabel("Counts")

#%% 7. Weight  ----------------------------------------------------------------
plt.scatter(A,[1,1,1]) # [1,1,1] 은 이제 sample index number가 되면 된다.
plt.title('Plot of NMF Weight in one sample')
plt.xlabel("NMF Weight")
plt.ylabel("Sample Index")
plt.yticks([1])


#%% 8. Now apply into 2D data ----------------------------------------------------------------

import numpy as np
from joblib import Parallel, delayed, cpu_count
from modi_hist_extract import *

source_ = 'ba133'
distance = 'close'    

rowsize = 1000 # 이미지의 row 개수

# data 시간 범위
starttime = 0
finaltime = 300 # data의 마지막 시간   
interval = (finaltime-starttime) / (rowsize-1)
endtime_values = np.linspace(starttime+interval, finaltime, rowsize, endpoint=True)

# data 불러오기
csv_file = f"../../../Data/240603_nucare/ori/{source_}_{distance}_5min.csv"
fil_dt = modi_hist_extract(csv_file)  # filtered data

# endtime 별 해당하는 histogram row 1개씩 뽑기
accumulate = 0

# debug 용
debug_counts = []


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
    
    return fil_dt.filtered_hist
    
# starttime ~ finaltime 사이를 rowsize 간격으로 나누어서 누적상태로 각 row에 저장.

results = Parallel(n_jobs=cpu_count(), verbose=10)(delayed(onestack_histimage)(i, i+1) for i in range(len(endtime_values)))
results = np.array(results)


#%% 8-1. imshow of results

plt.imshow(results)
plt.title('0-300sec of ba133 signal')
plt.xlabel("Energy (keV)")
plt.ylabel("Time(sec)")
plt.colorbar()

pixelsize = 100
y_ticks = [i*pixelsize for i in range(0,10)]
y_labels = [(finaltime/1000)*pixelsize*i for i in range(0,10)]
plt.yticks(ticks=y_ticks, labels=y_labels)


#%% 8-2. get nmf

X_hat, A, V = my_nmf(results)
#np.savez("./test/nmf_data_2D.npz", X=results, X_hat=X_hat, A=A, V=V, k=3, info="iterated 30000")


#%% 8-3. plot results V

plt.figure(figsize=(10,10))
plt.plot(V[0,:], linestyle=':', label='row=0')
plt.plot(V[1,:], linestyle=':', label='row=1')
plt.plot(V[2,:], linestyle=':', label='row=2')

plt.legend()
plt.title('Three(k=3) Poisson NMF basis vectors V')
plt.xlabel("Energy (keV)")
plt.ylabel("NMF Component(a.u.)")


#%% 8-4 plot results A

# 이건 아닌듯
for i in range(1000):
    plt.scatter(A[i,:],[i,i,i])








#%% 9. V - background modeling, t_s - source templete

csv_file = f"../../../Data/240527_simulation/Ba133/{source_}_{distance}_5min.csv"
fil_dt = modi_hist_extract(csv_file) 















































#%%
# DEPRECATED TRY  --------

# for l in range(k):
#     A[:,l] = A[:,l] * ( (sum(V[l,:]*X[l,:]/X_hat[l,:])) / (sum(V[l,:])) )
#     V[l,:] = V[l,:] * ( (sum(A[:,l]*X[:,0]/X_hat[:,0])) / (sum(A[:,l])) )  # V[:,j] : 
#     V[l,:] = V[l,:] / sum(V[l,:])    # V[l,:] : 1x1,     V : kx1
#     A[:,l] = A[:,l] * V
    

# for _ in range(iter_num):
#     A = A*((sum(V*X/X_hat))/(sum(V)))  # V:kx1, X:1000x1, X_hat:1000x1
#     V = V*((sum(A*X/X_hat))/(sum(A)))
    
        
# for i in range(n_epoch):
# H = H * ( (W.transpose()@X) / (W.transpose()@W@H) )  # 
# W = W * ( (X@H.transpose()) / (W@(H@H.transpose())) )

# plt.imshow(H)
# plt.title(f"Iteration : {i}")

# display(plt.gcf())
# clear_output(wait=True)
# #time.sleep(0.1)

# ---------------------------

# %%
