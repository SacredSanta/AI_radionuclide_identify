#%% ---------------------------------------------------------------------------------
# ---------------------   Evaluation  -------------------------------------
# --------------------------------------------------------------------

#%% 0. package import for evaluation -----------------------------------------------------------------
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
import tensorflow as tf
import seaborn as sns

#%% 1. load model -----------------------------------------------------------------
Model = tf.keras.models.load_model("/tf/latest_version/3.AI/Tensorflow/Assets/saved_model_off1/model.v1.h5")
Model.summary()


#%% 1-1. test model just test -----------------------------------------------------------------
#? 
data = np.load("/tf/latest_version/3.AI/Tensorflow/Data/spectrum_off1.npz")
test_cases_x = data["x_test"]
test_cases_y = data["y_test"]

i = 500
pred = Model.predict(test_cases_x[i][tf.newaxis,:,:,:])
print(pred)  # Ba133, Cs137, Am241, Co60, Ga67, I131, Ra226, Tc99m, Th232, Tl201, Na22
print(test_cases_y[i])  # Ba, Na, Cs

#%% 5-3. evaluation the Model ---------------------------------------------
# (1) accuracy

'''predicted_output = Model.predict(x_test)'''
print(np.sum((x_train_pred>0.5) == y_train) == 4)
print(np.sum((x_train_pred>0.5) == y_train, axis=1))
step2 = np.sum((x_train_pred>0.5) == y_train, axis=1)

print('acc: {} %'.format(
    np.sum(
        np.sum(
            (x_train_pred>0.5) == y_train, axis=1)== 4) / y_train.shape[0]))

#%% 5-3. --------------------------------------------------
# (2) Sensitivity & Precision & Specificity 

# ! 현재 train으로 test를 해서 임시방편으로 x_test를 변경함.
y_test_100 = y_train

threshold = 0.5
predicted_output = x_train_pred > threshold  # 일정값 이상으로 정리

y_test_100 = y_test_100 > threshold  # 한 행당 [T,F,F,F] 이런식

TP = (predicted_output & y_test_100).sum(1)  #np.ndarray.sum(axis=1)
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
