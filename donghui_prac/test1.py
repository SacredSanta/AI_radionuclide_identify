#%% ---------------------------------------------------------
'''
# model = tf.keras.Sequential([
#     tf.keras.layers.Conv2D(64, 7, activation='relu'), padding="same", input_shape=[28,28,1],
#     tf.keras.layers.MaxPool2d(2),
#     tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
#     tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
#     tf.keras.layers.MaxPool2d(2),
#     tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
#     tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
#     tf.keras.layers.MaxPool2d(2),
#     tf.keras.Flatten(),
#     tf.keras.layers.Dense(128, activation="relu")
# ])
'''
#%% DenseNet github---------------------------------------------------------
'''
# input_ = tf.keras.Input(shape=(224, 224, 3))
# output = tf.keras.applications.DenseNet121()(input_)
# Model = tf.keras.Model(inputs=[input_], outputs=[output])
# Model.summary()
'''
# %% 1. init ---------------------------------------------------------
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
print(os.environ)

# Library
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Check GPU
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
tf.config.set_visible_devices(physical_devices[0],'GPU')
'''
x_train: (#ofData, height, width, channel)
y_train: (#ofData)
x_test: (#ofData, height, width, channel)
y_test: (#ofData)
'''
## Data load
data = np.load("/tf/latest_version/3.AI/Tensorflow/Data/spectrum_off1.npz")
x_train = data['x_train']
y_train = data['y_train']
x_val = data['x_val']
y_val = data['y_val']
x_test = data['x_test']
y_test = data['y_test']

'''
# x_train = np.reshape(x_train[:,:,:,2],(24000,1,1000,1))
# x_val = np.reshape(x_val[:,:,:,2],(8000,1,1000,1))
# Check data
plt.plot(x_train[15000,:,:,0].flatten())
'''
'''
# # Dataset setting
# train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# train_dataset = train_dataset.shuffle(buffer_size=1024).batch(32)
# val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
# val_dataset = val_dataset.shuffle(buffer_size=1024).batch(32)
# test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
# test_dataset = test_dataset.shuffle(buffer_size=1024).batch(32)
'''
#%% 2. Model ---------------------------------------------------------
def bottleneck(x, growth_rate):
    inner_channel = 4 * growth_rate
    h = tf.keras.layers.BatchNormalization()(x)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2D(inner_channel,1,use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2D(growth_rate, 3, padding='same', use_bias=False)(h)
    return tf.keras.layers.Concatenate()([x,h])

def transition(x, out_channels):
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(out_channels, kernel_size=1, use_bias=False)(x)
    return tf.keras.layers.AvgPool2D(pool_size=(1,2), strides=2)(x)

def dense_layer(x, nblocks, growth_rate):
    for _ in range(nblocks):
        x = bottleneck(x, growth_rate)
    return x

def radionuclides_densenet():
    growth_rate = 12
    inner_channels = 2 * growth_rate   # convolutio 이후 feature map 개수, filter 개수
    nblocks = [2,4,8,4]
    reduction = 0.5
    num_class = 4

    input_tensor = tf.keras.Input(shape=(1,1000,2))
    x = tf.keras.layers.Conv2D(inner_channels, kernel_size=15, strides=2, padding='same', use_bias=False)(input_tensor)
    x = tf.keras.layers.MaxPool2D(pool_size=(1,7), strides=2)(x)  # pool size는 행,열    열을 7개씩 pool)

    x = dense_layer(x, nblocks[0], growth_rate)
    inner_channels += growth_rate * nblocks[0]
    out_channels = int(reduction * inner_channels)
    x = transition(x, out_channels)
    inner_channels = out_channels

    x = dense_layer(x, nblocks[1], growth_rate)
    inner_channels += growth_rate * nblocks[1]
    out_channels = int(reduction * inner_channels)
    x = transition(x, out_channels)
    inner_channels = out_channels

    x = dense_layer(x, nblocks[2], growth_rate)
    inner_channels += growth_rate * nblocks[2]
    out_channels = int(reduction * inner_channels)
    x = transition(x, out_channels)
    inner_channels = out_channels

    x = dense_layer(x, nblocks[3], growth_rate)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    output_tensor = tf.keras.layers.Dense(num_class, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)

    return model

Model = radionuclides_densenet()
Model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=0.1, weight_decay=1e-4),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['binary_accuracy'])

Model.summary()








#%% 3. train Model ---------------------------------------------------------
Model.fit(x_train,
          y_train,
          validation_data=(x_val,y_val),
          epochs=1,
          )







#%% 5-0. package import for evaluation -----------------------------------------------------------------
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
import seaborn as sns

#%% 5-0. load model and data ---------------------------------------------------------
Model = tf.keras.models.load_model("/tf/latest_version/3.AI/Tensorflow/Assets/saved_model_off1/model.v1.h5")

Model.summary()

data = np.load("/tf/latest_version/3.AI/Tensorflow/Data/spectrum_off1.npz")
train_cases_x = data["x_train"]
train_cases_y = data["y_train"]

#%% 5-1. predict train data ---------------------------------------------
data_num = 25000 # len(train_cases_y) : 24000
x_train_pred = np.zeros([data_num,4])   # pred 된 것들 정리 => 각 case별 확률로
print(x_train_pred.shape)

for i in range(0, data_num):
    pred = Model.predict(train_cases_x[i][tf.newaxis,:,:,:])    
    x_train_pred[i] = pred 
#predictions = Model.predict(test_one[tf.newaxis,:,:,:])



#%% 5-1. predict test data ------------------------------------------

test_num = x_test.shape[0]
x_test_pred = np.zeros([test_num,4])

for i in range(test_num):
    pred = Model.predict(x_test[i][tf.newaxis,:,:,:])
    x_test_pred[i] = pred
    
    


#%% 5-2. load predicted train data --------------------
# 일반 data 불러오기
x_train_pred = np.load("../predict_result/x_train_all.npy")
x_train_pred = x_train_pred[0:24000]



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

#%% 5-3.
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


#%% 5-3. print confusion matrix
cm = multilabel_confusion_matrix(y_test_100, predicted_output)
# multilabel_cm
# 각 열에 대한 요소(label)에 대해서 하나씩 confusion matrix 생성
print(cm)

#%% 5-3. print the final confusion matrix

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






















'''
여기부터 기존 코드
'''

#%%
## Test performance
# 1. accuracy
predicted_output = Model.predict(x_test)
print(np.sum((predicted_output>0.5) == y_test, axis=1) == 4)
print('acc: {} %'.format(np.sum(np.sum((predicted_output>0.5) == y_test, axis=1) == 4)/y_test.shape[0]))

# Sensitivity & Precision & Specificity 
threshold = 0.5
predicted_output = predicted_output > threshold
y_test = y_test > threshold
TP = (predicted_output & y_test).sum(1)
TN = ((~predicted_output) & (~y_test)).sum(1)
FP = (predicted_output & (~y_test)).sum(1)
FN = ((~predicted_output) & (y_test)).sum(1)

specificity = np.mean(TN/(FP+TN+1e-12))
print("specificity: {}".format(specificity))
precision = np.mean(TP/(TP+FP+1e-12))
print("precision: {}".format(precision))
sensitivity = np.mean(TP/(TP+FN+1e-12))
print("sensitivity: {}".format(sensitivity))
detection_rate = np.mean((TP+TN)/(TP+TN+FP+FN+1e-12))
print("detection rate: {}".format(detection_rate))

# Confusion matrix
from sklearn.metrics import multilabel_confusion_matrix
import seaborn as sns
cm = multilabel_confusion_matrix(y_test, predicted_output)
print(cm)
#%%
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
#%%
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







#%% ---------------------------------------------------------

# ## log with tensorboard
# import datetime
# ### designate log folder
# log_dir = "../logs/model1/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# ### define tensorboard callback
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
# Model.fit(x_train,
#           y_train,
#           validation_data=(x_val,y_val),
#           epochs=30,
#           callbacks=[tensorboard_callback],
#           )
# Model.summary()

# tf.keras.models.save_model(Model, "../Assets/saved_model_off1/")
# converter = tf.lite.TFLiteConverter.from_saved_model("../Assets/saved_model_off1/")
# tflite_model = converter.convert()
# with open('../Assets/model_off1.tflite', 'wb') as f:
#   f.write(tflite_model)
#%%
prob2 = x_train[15000,:,:,0].flatten()
plt.plot(prob2)

def gauss(x, H, A, x0, sigma): 
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

# %%
