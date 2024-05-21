#%%
import numpy as np
import os
import lzma
import pickle
import matplotlib.pyplot as plt
data = np.load("/tf/latest_version/3.AI/Tensorflow/Data/spectrum_off1.npz")
x_train = data['x_train']
y_train = data['y_train']
x_val = data['x_val']
y_val = data['y_val']
x_test = data['x_test']
y_test = data['y_test']


plt.plot(x_train[15000,:,:,0].flatten())
# x_train = np.reshape(x_train[:,:,:,2],(24000,1,1000,1))
# x_val = np.reshape(x_val[:,:,:,2],(8000,1,1000,1))
# Check data

# %% simulation
debug = 0

def gate_data_conversion(dpath):
    global debug
    dt = np.zeros((1,1000))
    if debug: print(dt)
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
            if debug: print("idx : ",idx); break;
            if not idx>=1000 and not idx<0:
                dt[:,idx] += 1
    return dt
            
        
components = ["Ba133"]
i=0
nu_data=[100]

#%% gate data plot ---------------------------------------------------------------------
debug = 0
datadir = "/tf/latest_version/3.AI/Tensorflow/Data/simulation"
datapath = "Ba133/id1061.dat"

pathh = os.path.join(datadir,datapath)
if debug:print(pathh)
dt = gate_data_conversion(pathh)
if debug:print(dt)

plt.plot(dt[0,:])
plt.show()
# 시뮬레이션 데이터가 정상적으로 안나옴/??
print("max : ", max((dt[0,:])))
print("count : ", sum(dt[0,:]))

#%% gate data with nz background--------------------------------------------------------------------
idname = 'id1061'
datadir = "/tf/latest_version/3.AI/Tensorflow/Data/"
datapath = "simulation/Ba133/{}.dat".format(idname)

pathh = os.path.join(datadir,datapath)

dt = gate_data_conversion(pathh)

print(max(dt[0]))
print(sum(dt[0]))
npzpath = "single/"
components = ["Background"]
dtname = "{}/{}_{}.xz".format(components[0], components[0],10)
pathh2 = os.path.join(datadir,npzpath,dtname)

with lzma.open(pathh2, 'rb') as f:
    temp = pickle.load(f)
    final = dt + temp.reshape(1,1000)
    print(sum(max(temp)))

saveflag = 0

if saveflag:
    savepath = os.path.join(datadir, "simulation/Ba133/withBack/{}_back.npz".format(idname))
    print(savepath)
    with open(savepath,'wb') as f:
        np.save(f, final)
plt.plot(final[0])
plt.show()



#%% check npz data--------------------------------------------------------------------
from IPython import display

npzpath = "single/"
components = ["Background"]

superposition = 0
arr = np.zeros([1000,1])
superpositioncut = 1  # 신호 합칠 개수

for datanum in range(1000):
    dtname = "{}/{}_{}.xz".format(components[0], components[0],datanum)
    pathh2 = os.path.join(datadir,npzpath,dtname)
    superposition += 1
    with lzma.open(pathh2, 'rb') as f:
        temp = pickle.load(f)
        arr += temp
        if superposition == superpositioncut:
            plt.plot(arr)
            plt.ylim([0,30])
            display.display(plt.gcf())
            plt.clf()
            print("total : ", sum(arr))
            arr = np.zeros([1000,1])
            superposition = 0
    display.clear_output(wait=True)
    
# %%
