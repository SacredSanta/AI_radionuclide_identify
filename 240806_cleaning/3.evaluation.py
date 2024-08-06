'''
최종 수정 : 2024.08.06.
사용자 : 서동휘

<수정 내용> 

<24.08.06>
- 240806_30to300sec_7source_5000.h 관련 평가
'''














#%% 0. package import for evaluation -----------------------------------------------------------------
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
import tensorflow as tf
import seaborn as sns
import numpy as np
import os

#%% 1. load model ========================================================
base_direc = "/tf/latest_version/3.AI/Tensorflow/Code/donghui_prac/2.model"
data_name = "240806_30to300sec_7source_5000"   # <-----------이거 고쳐서 이용!
model_direc = base_direc + '/' + data_name + '.h5'
Model = tf.keras.models.load_model(model_direc)
Model.summary()

#%% 선택!!  기존 모델로
model_direc = "/tf/latest_version/3.AI/Tensorflow/Assets/saved_model_off1/model.v1.h5"
Model = tf.keras.models.load_model(model_direc)
Model.summary()







#%% 2. Load data for evaluation ========================================================
data_direc = "./1.preprocess_data/"

change = 1
if change : data_name = '240806_4source_3000_xzfile'

test_npz = np.load(data_direc + data_name + "_all.npz")
x_test = test_npz["x_test"]
y_test = test_npz["y_test"]
#%% 선택(1)! Load from 이전 data
test_npz = np.load("../../Data/spectrum_off1.npz")
x_test = test_npz["x_test"]
y_test = test_npz["y_test"]

# output 개수가 달라져서 이전의 data에서는 0으로 모두 추가.
tail = np.zeros([len(y_test), 3])
y_test = np.concatenate((y_test, tail), axis=1)

#%% 선택(2)! Load 2D data
test_img_x = np.load(f"./0.dataset/{data_name}.npy")
test_img_y = np.load(f"./1.preprocess_data/{data_name}_y.npy")

x_test = test_img_x[800:,:,:]
y_test = test_img_y[800:,:,:]

print(x_test.shape)
print(y_test.shape)
#%% !선택! 최종적으로 shape 변환
# x_test 들어가야하는 방식이 n, 1, 1000, 2
# y 들어가야하는 방식이 n, source개수
x_test = x_test[:,tf.newaxis,:,:]
print(x_test.shape)
print(y_test.shape)















#%% 3. predict data ========================================================
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
pred = []
for task in tasks:
    if task % 100 == 0: print(task, " data proceed..")
    pred.append(predict_data(task)[0])
pred = np.vstack(pred)

''' ==legacy==
pred = Model.predict(test_cases_xtest[i][tf.newaxis,tf.newaxis,:,:])
print(pred)  # Ba133, Cs137, Am241, Co60, Ga67, I131, Ra226, Tc99m, Th232, Tl201, Na22
print(test_cases_y[i])  # Ba, Na, Cs
'''
print("Predict Done!")








#%% 3-1. save predict data
#data_name = "5to15sec_7_source_close_35cm_12000"
np.save(f"./3.predict/{data_name}_pred.npy", pred)

#%% 선택! load predicted data (kernel 끊겼을 때)
#data_name = "old_data"
pred = np.load(f"./3.predict/{data_name}_pred.npy")








#%% 4. evaluation the Model ---------------------------------------------
#%% (0) current data shape check
print(pred.shape)
print(y_test.shape)
#%% !선택! data shape 안맞는다면..
y_test = y_test[:,0,:]



#%% (1) accuracy

''' == legacy == 
print(np.sum((x_train_pred>0.5) == y_train) == 4)
print(np.sum((x_train_pred>0.5) == y_train, axis=1))
step2 = np.sum((x_train_pred>0.5) == y_train, axis=1)
'''

print('accuracy : {} %'.format(
    np.sum(
        np.sum(
            (pred > 0.5) == y_test, axis=1) == 7) / len(y_test)))

#%% (2) set pred and answer

threshold = 0.5
TF_pred = pred > threshold  # 일정값 이상으로 정리
TF_y = y_test > threshold  # 한 행당 [T,F,F,F] 이런식
#%% (2-2)  confusion matrix, // other index for each classification
class_num = 7
# confusion matrix 직접 구현 - actual: col, pred: row
# Act1. 개별 source에 대해서 마다 confusion matrix를 구해야할듯
# Act2. 다중분류 하나하나 모두 개별 case로 간주하여 confusion matrix를 구하는것도 방법인듯.
import pandas as pd

# Act1.
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

#%% (3) calculate the index

#? new column은 선택해서 쓰면됨.
# <------------------- 이 아래 cm_all.column             
#cm_all.columns = ["ba133", "cs137", "na22", "background"]
cm_all.columns = ['ba133', 'cs137', 'na22', 'background', 'co57', 'th232', 'ra226']

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
#%% 최종 결과 저장. cm_all.csv로

cm_all.to_csv(f'./3.predict/{data_name}_results.csv', index=False)
















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
