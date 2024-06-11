#%% 0. package import for evaluation -----------------------------------------------------------------
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
import tensorflow as tf
import seaborn as sns
import numpy as np
import os

#%% 1. load model -----------------------------------------------------------------
base_direc = "/tf/latest_version/3.AI/Tensorflow/Code/donghui_prac/2.model"
model_name = "5sec_basic_source.h5"
model_direc = os.path.join(base_direc, model_name)

# 우선 기존 모델로..
model_direc = "/tf/latest_version/3.AI/Tensorflow/Assets/saved_model_off1/model.v1.h5"
Model = tf.keras.models.load_model(model_direc)
Model.summary()




#%% 2. Load data for evaluation
data_name = "5sec_basic_source"
data_direc = "./1.preprocess_data/"

test_cases_xtest = np.load(data_direc + data_name + "_xtest.npy")
test_cases_y = np.load(data_direc + data_name + "_y.npy")




#%% 3. predict data -----------------------------------------------------------------
#? 
from joblib import Parallel, delayed, cpu_count 

num_cores = cpu_count()
tasks = range(3000)

# 하나씩 predict 구현 -> joblib으로 돌리려고
def predict_data(idx:int):
    global Model
    global test_cases_xtest
    pred = Model.predict(test_cases_xtest[idx][tf.newaxis, tf.newaxis, :, :])
    return pred

# parallel 이용해서 data 모두 predict 
# 오류 생겨서 보류..
#results = Parallel(n_jobs=num_cores, verbose=10)(delayed(predict_data)(task) for task in tasks)

# 단순 반복
pred = []
for task in tasks:
    pred.append(predict_data(task)[0])
pred = np.vstack(pred)

''' ==legacy==
pred = Model.predict(test_cases_xtest[i][tf.newaxis,tf.newaxis,:,:])
print(pred)  # Ba133, Cs137, Am241, Co60, Ga67, I131, Ra226, Tc99m, Th232, Tl201, Na22
print(test_cases_y[i])  # Ba, Na, Cs
'''
print("Predict Done!")

#%% 3-1. save predict data
filename = "5sec_basic_source"
np.save(f"./3.predict/{filename}_pred.npy", pred)







#%% 4. evaluation the Model ---------------------------------------------
#%% (0) current data shape check
print(pred.shape)
print(test_cases_y.shape)

#%% (1) accuracy

''' == legacy == 
print(np.sum((x_train_pred>0.5) == y_train) == 4)
print(np.sum((x_train_pred>0.5) == y_train, axis=1))
step2 = np.sum((x_train_pred>0.5) == y_train, axis=1)
'''

print('accuracy : {} %'.format(
    np.sum(
        np.sum(
            (pred > 0.5) == test_cases_y[:, 0, :], axis=1) == 4) / len(test_cases_y)))

#%% (2) set pred and answer

threshold = 0.5
TF_pred = pred > threshold  # 일정값 이상으로 정리
TF_y = test_cases_y[:, 0, :] > threshold  # 한 행당 [T,F,F,F] 이런식
#%% (2-2)  confusion matrix, // other index for each classification
class_num = 4
# confusion matrix 직접 구현 - actual: col, pred: row
# Act1. 개별 source에 대해서 마다 confusion matrix를 구해야할듯
# Act2. 다중분류 하나하나 모두 개별 case로 간주하여 confusion matrix를 구하는것도 방법인듯.
import pandas as pd

# Act1.
#cm_all = pd.DataFrame(np.zeros([class_num, class_num]), columns=["ba133", "cs137", "na22", "background"]) # column 이름 지정
cm_all = pd.DataFrame(np.zeros([class_num, class_num]), columns=[0, 1, 2, 3])
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
                
cm_all.columns = ["ba133", "cs137", "na22", "background"]

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
