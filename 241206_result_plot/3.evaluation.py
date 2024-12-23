'''
최종 수정 : 2024.08.06.
사용자 : 서동휘

<수정 내용> 

<24.08.06>
- 240806_30to300sec_7source_5000.h 관련 평가
'''







#%%
@tf.keras.utils.register_keras_serializable()   # 아니 이게 되네; - 
def swish(x):
    """Swish activation function.

    # Arguments
        x: Input tensor.

    # Returns
        The Swish activation: `x * sigmoid(x)`.

    # References
        [Searching for Activation Functions](https://arxiv.org/abs/1710.05941)
    """
    if backend.backend() == 'tensorflow':
        try:
            # The native TF implementation has a more
            # memory-efficient gradient implementation
            return backend.tf.nn.swish(x)
        except AttributeError:
            pass

    return x * backend.sigmoid(x)

#%% 만약 커스텀 함수 있다면
from tensorflow.keras.utils import get_custom_objects
import sys

get_custom_objects().update({'swish' : swish})





#%% 0. package import for evaluation -----------------------------------------------------------------
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
import tensorflow as tf
import seaborn as sns
import numpy as np
import os

#%% 1. load model (legacy version - .h5) 옛날버전 ========================================================
base_direc = "/tf/latest_version/3.AI/Tensorflow/Code/donghui_prac/2.model"
model_name = "240806_30to300sec_7source_5000"   # <-----------이거 고쳐서 이용!
model_direc = base_direc + '/' + model_name + '.h5'
Model = tf.keras.models.load_model(model_direc)
Model.summary()

#%% 1. load model (new version - .keras) ========================================================
base_direc = "/tf/latest_version/new_AI/docker_data_241016/Tensorflow/Code/donghui_prac/2.model/241021_efficient_shallow"
model_name = "241004_10to20sec_3series_merged_241021_efficientnet_shallow_0.1.keras"   # <-----------이거 고쳐서 이용!
model_direc = os.path.join(base_direc, model_name)
Model = tf.keras.models.load_model(model_direc)
Model.summary()


#%% 선택!!  기존 모델로 ========================================================
model_direc = "/tf/latest_version/3.AI/Tensorflow/Assets/saved_model_off1/model.v1.h5"
Model = tf.keras.models.load_model(model_direc)
Model.summary()










#%% 2. Load data for evaluation ========================================================
data_direc = "./1.preprocess_data/"

change = 1
if change : data_name = '241004_10to20sec_3series_merged_all'

test_npz = np.load(data_direc + data_name + ".npz")
x_test = test_npz["x_test"]#[:,:,:2]
y_test = test_npz["y_test"]

print(x_test.shape)
print(y_test.shape)
#%% 선택(1)! Load from 이전 data -------------------------------------------------------
test_npz = np.load("../../Data/spectrum_off1.npz")
x_test = test_npz["x_test"]
y_test = test_npz["y_test"]

# output 개수가 달라져서 이전의 data에서는 0으로 모두 추가.
tail = np.zeros([len(y_test), 3])
y_test = np.concatenate((y_test, tail), axis=1)

#%% 선택(2)! Load 2D data -------------------------------------------------------
test_img_x = np.load(f"./0.dataset/{data_name}.npy")
test_img_y = np.load(f"./1.preprocess_data/{data_name}_y.npy")

x_test = test_img_x[800:,:,:]
y_test = test_img_y[800:,:,:]

print(x_test.shape)
print(y_test.shape)
#%% 최종적으로 shape 변환 -------------------------------------------------------
# x_test 들어가야하는 방식이 n, 1, 1000, 2
# y 들어가야하는 방식이 n, source개수
x_test = x_test[:,tf.newaxis,:,:]
print(x_test.shape)
print(y_test.shape)















#%% 3. predict data ========================================================
backend = tf.keras.backend
layers = tf.keras.layers
models = tf.keras.models
keras_utils = None

tasks = range(len(x_test))

# 2d input model flag
is2d = 0

# 하나씩 predict 구현 -> joblib으로 돌리려고
def predict_data(idx:int):
    global Model
    global x_test
    
    # input이 2D인 모델의 경우
    if is2d : pred_thing = x_test[idx]
    # 일반 기존 model
    else: pred_thing = x_test[idx][tf.newaxis,:,:,:]
    
    pred = Model.predict(pred_thing)
    return pred

#%% joblib 이용해서.. -> 오류! ---------------------------------------------
from joblib import Parallel, delayed, cpu_count 
num_cores = cpu_count()

# parallel 이용해서 data 모두 predict 
# 오류 생겨서 보류..
predictions = predict_data(Model, x_test, num_cores)

#%% 단순 반복 -----------------------------------------------------------------
# pred = []
# for task in tasks:
#     if task % 100 == 0: print(task, " data proceed..")
#     pred.append(predict_data(task)[0])
# pred = np.vstack(pred)

''' ==legacy==
pred = Model.predict(test_cases_xtest[i][tf.newaxis,tf.newaxis,:,:])
print(pred)  # Ba133, Cs137, Am241, Co60, Ga67, I131, Ra226, Tc99m, Th232, Tl201, Na22
print(test_cases_y[i])  # Ba, Na, Cs
'''
print("Predict Done!")








#%% 3-1. save predict data  ==============================================================
#data_name = "5to15sec_7_source_close_35cm_12000"
date = "241007_unetpp"
np.save(f"./3.predict/efficient_shallow/{model_name}_pred.npy", pred)

#%% 선택! load predicted data (kernel 끊겼을 때)  ---------------------------------
#data_name = "old_data"
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
import tensorflow as tf
import seaborn as sns
import numpy as np
import os
#%%
data_name = "241004_10to20sec_3series_merged_all"
_date = "241021_densenet2d_v3_nestdense"
pred = np.load(f"./3.predict/{data_name}_pred{_date}.npy")








#%% 4. evaluation the Model ==============================================================
#%% 4-(0) current data shape check
print(pred.shape)
print(y_test.shape)
#%% !선택! data shape 안맞는다면..
y_test = y_test[:,0,:]



#%% 4-(1) accuracy

''' == legacy == 
print(np.sum((x_train_pred>0.5) == y_train) == 4)
print(np.sum((x_train_pred>0.5) == y_train, axis=1))
step2 = np.sum((x_train_pred>0.5) == y_train, axis=1)
'''

print('accuracy : {} %'.format(
    np.sum(
        np.sum(
            (pred > 0.5) == y_test, axis=1) == 8) / len(y_test) * 100))

#%% 4-(2) set pred and answer

threshold = 0.5
TF_pred = pred > threshold  # 일정값 이상으로 정리
TF_y = y_test > threshold  # 한 행당 [T,F,F,F] 이런식
#%% 4-(2-2)  confusion matrix, // other index for each classification
class_num = 8
# confusion matrix 직접 구현 - actual: col, pred: row
# <Act1> 개별 source에 대해서 마다 confusion matrix를 구해야할듯
# <Act2> 다중분류 하나하나 모두 개별 case로 간주하여 confusion matrix를 구하는것도 방법인듯.
import pandas as pd

# <Act1>
#cm_all = pd.DataFrame(np.zeros([class_num, class_num]), columns=["ba133", "cs137", "na22", "background"]) # column 이름 지정
cm_all = pd.DataFrame(np.zeros([4, class_num]), columns=[i for i in range(class_num)])
cm_all.index = ["TP", "FP", "TN", "FN"] # row 이름 지정

# 각기 알맞게 값 +1 해주기. pandas 관련 warning이 아마 뜰거임. 무시해도 괜찮음.
for col in range(class_num):
    for row in range(len(pred)):
        if TF_pred[row, col] == TF_y[row, col]:
            if TF_pred[row, col] == 1:
                cm_all.loc[:, col]["TP"] += 1   # pos를 정확히 예측
            else:
                cm_all.loc[:, col]["TN"] += 1   # neg를 정확히 예측
        else:
            if TF_pred[row, col] == 1:   # 가짜 True
                cm_all.loc[:, col]["FP"] += 1
            else:
                cm_all.loc[:, col]["FN"] += 1

#%% 4-(3) calculate the index

#? new column은 선택해서 쓰면됨.
# <------------------- 이 아래 cm_all.column             
#cm_all.columns = ["ba133", "cs137", "na22", "background"]
cm_all.columns = np.array(['ba', 'cs', 'na', 'background', 'co57', 'th', 'ra', 'am'])

# 지표용 row 부분 초기화해주기
new_row = ["accuracy", "recall", "specificity", "precision", "F1 Score"]
for row in new_row:
    cm_all.loc[row] = np.zeros(class_num)

# 값 넣어주기    
cm_all.loc["accuracy"] = (cm_all.loc["TP"] + cm_all.loc["TN"]) / (cm_all.loc["TP"] + cm_all.loc["TN"] + cm_all.loc["FP"] + cm_all.loc["FN"])
cm_all.loc["recall"] = (cm_all.loc["TP"]) / (cm_all.loc["TP"] + cm_all.loc["FN"])
cm_all.loc["specificity"] = (cm_all.loc["TN"]) / (cm_all.loc["TN"] + cm_all.loc["FP"])
cm_all.loc["precision"] = (cm_all.loc["TP"]) / (cm_all.loc["TP"] + cm_all.loc["FP"])
cm_all.loc["F1 Score"] = 2*(cm_all.loc["TP"]) / (2*(cm_all.loc["TP"]) + (cm_all.loc["FP"]) + (cm_all.loc["FN"]))

""" === legacy === 
TN = ((~predicted_output) & (~y_test_100)).sum(1)
FP = (predicted_output & (~y_test_100)).sum(1)
FN = ((~predicted_output) & (y_test_100)).sum(1)


specificity = np.mean(TN/(FP+TN+1e-12))  
print("specificity: {}".format(specificity))
precision = np.mean(TP/(TP+FP+1e-12))
print("precision: {}".format(precision))
sensitivity = np.mean(TP/(TP+FN+1e-12))
print("sensitivity: {}".format(sensitivity))
detection_rate = np.mean((TP+TN)/(TP+TN+FP+FN+1e-12))
print("detection rate: {}".format(detection_rate))
"""
print(cm_all)
#%% 5.최종 결과 저장. cm_all.csv로

cm_all.to_csv(f'./3.predict/{data_name}_results.csv', index=False)




#%% 6.AUC Curve ===========================================================================================

src_num = len(cm_all.columns)
data_num = pred.shape[0]
roc_len = 20  # thres 간격 개수

roc_values = np.zeros([src_num, 6, roc_len]) # 행 : source, 열 : thres
# row 의 6은 순서대로 : TP, TN, FP, FN, TPR, FPR

thres_col = 0
for thres in np.arange(0, 1, 0.05):
    TF_pred = pred > thres  
    TF_y = y_test > thres  # 한 행당 [T,F,F,F] 이런식

    for src_dim in range(src_num): # source 에 대한 반복
        for cnt_row in range(data_num): # data 개수에 대한 반복
            
            if TF_pred[cnt_row, src_dim] == TF_y[cnt_row, src_dim]:
                if TF_pred[cnt_row, src_dim] == 1:
                    roc_values[src_dim, 0, thres_col] += 1   # TP : pos를 정확히 예측
                else:
                    roc_values[src_dim, 1, thres_col] += 1   # TN : neg를 정확히 예측
            else:
                if TF_pred[cnt_row, src_dim] == 1:   # FP : 가짜 True
                    roc_values[src_dim, 2, thres_col] += 1
                else:   # FN 
                    roc_values[src_dim, 3, thres_col] += 1
    
    thres_col += 1

# TPR
roc_values[:,4,:] = roc_values[:,0,:] / (roc_values[:,0,:] + roc_values[:,3,:] + 0.00001)

# FPR
roc_values[:,5,:] = roc_values[:,2,:] / (roc_values[:,2,:] + roc_values[:,1,:] + 0.00001)

#%% 6-2. Plot ROC ! ===========================================================================================
import matplotlib.pyplot as plt

fig, ax = plt.subplots(src_num, 1, figsize=(8,40))

temp = np.linspace(0,1,roc_len)

for i in range(src_num):
    area = round(abs(np.trapz(np.append(roc_values[i,4,:],0), np.append(roc_values[i,5,:],0))),3)
    ax[i].plot(np.append(roc_values[i,5,:],0), np.append(roc_values[i,4,:],0),
               linewidth='2', color='red')
    ax[i].plot(temp, temp, linestyle=':', linewidth='2', color='blue')
    ax[i].grid(True)
    ax[i].set_title(f"{cm_all.columns[i]}")
    ax[i].set_xlabel("False Positive Rate")
    ax[i].set_ylabel("True Positive Rate")
    ax[i].text(0.8,0.8, f"AUC : {area}", color='red', fontsize=20, 
               bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
    #ax[i].margins(x=0,y=0)

plt.subplots_adjust(wspace=10)
plt.show()

#%% 6-3. bar Plot ====================================================================================
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 1, figsize=(10,10))

bar_x = cm_all.columns
bar_x = np.delete(bar_x, 3)

bar_y = cm_all.loc['accuracy'].drop('background')
bar_y2 = cm_all.loc['F1 Score'].drop('background')

ax[0].bar(bar_x, bar_y)
ax[0].set_xlabel("Source Name", fontsize=15)
ax[0].set_ylabel("Prediction Accuracy", fontsize=15)

ax[1].bar(bar_x, bar_y2)
ax[1].set_xlabel("Source Name", fontsize=15)
ax[1].set_ylabel("F1 Score", fontsize=15)










#%% Debug!   ===========================================================================================
import matplotlib.pyplot as plt

ba133_rows = np.where((y_test[:,0] == 1) & (y_test[:,1] == 0))[0]
print(len(ba133_rows))

picked_rows = np.random.choice(ba133_rows, 6)

fig2, ax2 = plt.subplots(2,3)

for i in range(6):
    print(i)
    ax2[i//3, i%3].plot(x_test[picked_rows[i],0,:,0])





















#%% DEPRECATED BELOW !! -----------------------------------------------------------
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#%% 5-3. print confusion matrix  --------------------------------------------------
cm = multilabel_confusion_matrix(y_test_100, predicted_output)
# multilabel_cm
# 각 열에 대한 요소(label)에 대해서 하나씩 confusion matrix 생성
print(cm)

#%% 5-3. print the final confusion matrix --------------------------------------------------

Ba133 = cm[0,:,:] 
group_names = ['True N', 'False P', 'False N', 'True P']
group_counts = ['{0:0.0f}'.format(value) for value in
                Ba133.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in
                     Ba133.flatten()/np.sum(Ba133)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(Ba133, annot=labels, annot_kws={"fontsize":20}, fmt='',cmap='gray')

Na22 = cm[1,:,:]
group_names = ['True N', 'False P', 'False N', 'True P']
group_counts = ['{0:0.0f}'.format(value) for value in
                Na22.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in
                     Na22.flatten()/np.sum(Na22)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(Na22, annot=labels, annot_kws={"fontsize":20}, fmt='',cmap='gray')

Cs137 = cm[2,:,:]
group_names = ['True N', 'False P', 'False N', 'True P']
group_counts = ['{0:0.0f}'.format(value) for value in
                Cs137.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in
                     Cs137.flatten()/np.sum(Cs137)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(Ba133, annot=labels, annot_kws={"fontsize":20}, fmt='',cmap='gray')













































#%% test --------------------------------------------------------------
import numpy as np
import os

debug = 1

dir_ = "C:/Users/MIPL/Desktop/gatetest/mac/data"
id = 1050

if debug:
    print("id{}.txtSingles.dat".format(id))

pathAll = os.path.join(dir_, "id{}.txtSingles.dat".format(id))

if debug:
    #print(pathAll)
    print(type(pathAll))

with open(pathAll, 'r') as f:
    ff = f.read()
    #print(ff)
    if debug:
        print(type(ff))

#%%
import numpy as np
import os

debug = 1

dir_ = "C:/Users/MIPL/Desktop/gatetest/mac/data"
id = 1050

if debug:
    print("id{}.txtSingles.dat".format(id))

pathAll = os.path.join(dir_, "id{}.Singles.npy".format(id))

dt = np.load(pathAll)

print(dt)
