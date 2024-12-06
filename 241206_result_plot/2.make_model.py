'''
최종 수정 : 2024.08.06.
사용자 : 서동휘

- 241121
초기 densenet model 코드.



<수정 내용> 

<24.08.06>
- model input 을 (? x 1000 x 3) 으로 변경
'''
#%%
#%% Model Part ==========================================================
#%% 0. init 
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
print(os.environ)

# Library
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys

# Check GPU
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
tf.config.set_visible_devices(physical_devices[0],'GPU')

#%% 0. functions ====================================================================

def bottleneck(x, growth_rate, num):
    inner_channel = 4 * growth_rate
    h01 = tf.keras.layers.BatchNormalization()(x)
    h01 = tf.keras.layers.ReLU()(h01)
    h02 = tf.keras.layers.Conv2D(inner_channel, kernel_size=1, use_bias=False)(h01)
    h02 = tf.keras.layers.BatchNormalization()(h02)
    h02 = tf.keras.layers.ReLU()(h02)
    h03 = tf.keras.layers.Conv2D(growth_rate, kernel_size=3, padding='same', use_bias=False)(h02)
    print("=======BottleNeck=======", inner_channel, growth_rate)
    # print("growth rate : ", growth_rate)
    print("inner channel : ", inner_channel)
    print("x : " , x.shape)
    print("h02 : ", h02.shape)
    print("h03 : ", h03.shape)
    return tf.keras.layers.Concatenate()([x,h03])

def transition(x, out_channels, *layers):
    # 이전 Dense Block 정보들 넘겨주기
    lynum = 0
    for i in layers:
        lynum += 1
        x = tf.keras.layers.Concatenate()([x,i])
    
    tt1 = tf.keras.layers.BatchNormalization()(x)
    tt2 = tf.keras.layers.ReLU()(tt1)
    tt3 = tf.keras.layers.Conv2D(out_channels, kernel_size=1, use_bias=False)(tt2)  # reduction 적용된 filter 개수
    y = tf.keras.layers.AvgPool2D(pool_size=(1,2), strides=2)(tt3)
    print("========Trans========", out_channels)
    print("t01 : ", tt1.shape)
    print("t02 : ", tt2.shape)
    print("t03 : ", tt3.shape)
    print("y : ", y.shape)
    return y

def dense_layer(x, nblocks, growth_rate):
    print("--====--==--== Dense layer --==--==--==--==")
    for _ in range(nblocks):
        x = bottleneck(x, growth_rate, _)
        print("h : ", x.shape)
    print("--====--==--== ENd --==--==--==--==")
    return x


# 1. model definition (version 1) 

def radionuclides_densenet():
    
    # main parameters
    #? 각 Layer에서 몇 개의 feature map을 뽑을지 결정 - 각 layer가 전체 output에 어느정도 기여할지
    growth_rate = 12  
    out_channels = []
    
    inner_channels = [2 * growth_rate]   # convolution 이후 feature map 개수, filter 개수
    nblocks = [2,4,8,4]
    
    #? transition layer에서 반환하는 feature map
    reduction = 0.5 # 논문에서는 0.5 #0.2
    
    #? source = np.array(['ba', 'cs', 'na', 'background', 'co57', 'th', 'ra', 'am'])
    num_class = 8
    
    
    # model build
    phase = 0
    
    # input
    input_tensor = tf.keras.Input(shape=(1,1000,3))
  
    x01_c = tf.keras.layers.Conv2D(inner_channels[phase], kernel_size=7, strides=2, padding='same', use_bias=False)(input_tensor)
    x01_p = tf.keras.layers.MaxPool2D(pool_size=(1,7), strides=2)(x01_c)  # pool size는 행,열    열을 7개씩 pool)
    #print("x01_c : ", x01_c.shape)
    #print("x01_p : ", x01_p.shape)
    
    # Dense Block 1
    d01 = dense_layer(x01_p, nblocks[phase], growth_rate)
    #d01_p = tf.keras.layers.AvgPool2D(pool_size=(1,2), strides=2)(d01)
    #d01_pp = tf.keras.layers.AvgPool2D(pool_size=(1,2), strides=2)(d01_p)
    #d01_ppp = tf.keras.layers.AvgPool2D(pool_size=(1,2), strides=2)(d01_pp)
    #print("d01 : ", d01.shape)
    #print("d01_p : ", d01_p.shape)
    #print("d01_pp : ", d01_pp.shape)
    #print("d01_ppp : ", d01_ppp.shape)
    
    inner_channels.append(inner_channels[2*phase]+growth_rate * nblocks[phase])
    out_channels.append(int(reduction * inner_channels[2*phase+1]))
    print("<<<inner_channels>>> : ", inner_channels)
    print("<<<out_channels>>> : ", out_channels)
    
    # Trans 1
    t01 = transition(d01, out_channels[phase])
    inner_channels.append(out_channels[phase])   # 현 filter 개수 저장용.

    print("\n\n")
    phase += 1
    
    # Dense Block 2
    d02 = dense_layer(t01, nblocks[phase], growth_rate)
    #d02_p = tf.keras.layers.AvgPool2D(pool_size=(1,2), strides=2)(d02)
    #d02_pp = tf.keras.layers.AvgPool2D(pool_size=(1,2), strides=2)(d02_p)
    print("d02 : ", d02.shape)
    #print("d02_p : ", d02_p.shape)
    #print("d02_pp : ", d02_pp.shape)
    
    inner_channels.append(inner_channels[2*phase] + growth_rate * nblocks[phase])
    out_channels.append(out_channels[phase-1] + int(reduction * inner_channels[2*phase+1]))
    print("<<<inner_channels>>> : ", inner_channels)
    print("<<<out_channels>>> : ", out_channels)
    
    # Trans 2
    t02 = transition(d02, out_channels[phase])#, d01_p)
    inner_channels.append(out_channels[phase])
    
    print("\n\n")
    phase += 1

    # Dense Block 3
    d03 = dense_layer(t02, nblocks[phase], growth_rate)
    #d03_p = tf.keras.layers.AvgPool2D(pool_size=(1,2), strides=2)(d03)
    
    print("d03 : ", d03.shape)
    #print("d03_p : ", d03_p.shape)


    inner_channels.append(inner_channels[2*phase] + growth_rate * nblocks[phase])
    out_channels.append(out_channels[phase-1] + int(reduction * inner_channels[2*phase+1]))
    print("<<<inner_channels>>> : ", inner_channels)
    print("<<<out_channels>>> : ", out_channels)
    
    # Trans 3
    t03 = transition(d03, out_channels[phase])#, d01_pp, d02_p)
    inner_channels.append(out_channels[phase])
    #sys.exit()
    print("\n\n")
    phase += 1



    # Dense Block 4
    d04 = dense_layer(t03, nblocks[phase], growth_rate)
    print("d04 : ", d04.shape)
    
    inner_channels.append(inner_channels[2*phase] + growth_rate * nblocks[phase])
    out_channels.append(out_channels[phase-1] + int(reduction * inner_channels[2*phase+1]))
    print("<<<inner_channels>>> : ", inner_channels)
    print("<<<out_channels>>> : ", out_channels)
    
    #for i in [d01_ppp, d02_pp, d03_p]:
    #    d04 = tf.keras.layers.Concatenate()([d04, i])
    print("final d04 : ", d04.shape)
    
    d04 = tf.keras.layers.BatchNormalization()(d04)
    d04 = tf.keras.layers.ReLU()(d04)
    d04 = tf.keras.layers.GlobalAveragePooling2D()(d04)

    print("GlobalAveragePooling : ", d04.shape)
    

    output_tensor = tf.keras.layers.Dense(num_class, activation='sigmoid')(d04)
    model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)

    return model

Model = radionuclides_densenet()
Model.summary(line_length=100)
#%% 2. model compile ====================================================================
Model.compile(optimizer=tf.keras.optimizers.Adam(),#learning_rate=0.01, clipnorm=0.1, weight_decay=1e-4),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['binary_accuracy'])

Model.summary()



#%% 2. train the data - get data =======================================================
import numpy as np
#model_data = np.load("/tf/latest_version/3.AI/Tensorflow/Code/donghui_prac/spectrum_off_gate_2.npz")
filename = "241004_10to20sec_3series_merged_orispectrum_normed_noisefiltered"
model_data = np.load(f"./1.preprocess_data/{filename}.npz")

x_train = model_data["x_train"]#[:,:,:2]
y_train = model_data["y_train"]

x_val = model_data["x_val"]#[:,:,:2]
y_val = model_data["y_val"]

x_test = model_data["x_test"]
y_test = model_data["y_test"]

#%% (Debug) Nan이 있다면?  =======================================================
print(np.isnan(x_train).any())
nan_indices = np.where(np.isnan(x_train))
print(len(nan_indices[0]))

#%% 2-1. newaxis가 필요하다면 =======================================================
import tensorflow as tf
x_train = x_train[:,tf.newaxis,:,:]
y_train = y_train[:,0,:]
x_val = x_val[:,tf.newaxis,:,:]
y_val = y_val[:,0,:]
x_test = x_test[:,tf.newaxis,:,:]
y_test = y_test[:,0,:]

#%% =======================================================
print(x_train.shape)  # 최종 형태는 (개수, 1, 1000, 3)
print(y_train.shape)  # 최종 형태는 (개수, source 개수)
print(x_val.shape)
print(y_val.shape)
#%% 3. CSV Logger

date = "241113_densenet_vfrog_adam" 
csv_logger = tf.keras.callbacks.CSVLogger(filename=f"./2.model/241110_frogmodel/{filename}_{date}.log",
                                          separator=',',
                                          append=False)
weights = tf.keras.callbacks.ModelCheckpoint(filepath=f"./2.model/241110_frogmodel/{filename}_{date}.keras",
                                             save_weights_only=False,
                                             verbose=1)
#%% (선택사항!) 기존 model 불러와서 추가학습이라면...
base_direc = "/tf/latest_version/3.AI/Tensorflow/Code/donghui_prac/2.model/241110_frogmodel/"
model_name = "240923_10to20sec_8source_10000"   # <-----------이거 고쳐서 이용!
model_direc = base_direc + '/' + model_name + '.keras'
Model = tf.keras.models.load_model(model_direc)
Model.summary()




#%% 4. fit
import datetime
fit_time_start = datetime.datetime.now()
# Nonetype 관련 오류가 뜰 때도 있는데 gpu가 안잡혀서 그럴 수도 있으니 커널 재시작해보고 다시해볼것.
Model.fit(x_train,
          y_train,
          validation_data=(x_val,y_val),
          epochs=30,
          batch_size=32,
          callbacks=[csv_logger, weights]
          )
fit_time_end = datetime.datetime.now()
fit_time_total = fit_time_end - fit_time_start
print("학습시간 : ", fit_time_total)

#%% 4. save model
from tensorflow import keras

#tf.keras.models.save_model(Model, f"./2.model/{filename}.h5")
Model.save(f"./2.model/241110_frogmodel/Model_{filename}__{date}.keras")

#%%
Model2 = tf.keras.models.load_model("./2.model/241004/241004_10to20sec_3series_merged__241004_dense.keras")

#%% Predict

pred = Model.predict(x_test)
#pred = np.load("./3.predict/241004_10to20sec_3series_merged_all_pred241007_dense.npy")

#%%
test_acc = (np.sum(np.sum( (pred > 0.5) == y_test, axis=1) == 8) / len(y_test) * 100)
test_acc


# %% ======================================================
import pandas as pd
import os
import pandas as pd
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

foldername = "241110_frogmodel/241113_frog_adam_default"
Model.save(f"./2.model/{foldername}/model_{filename}_{date}.keras")
print("<<<Model Saved>>>")

test_x = tf.random.normal(shape=(1,1,1000,3))
pred_time_start = datetime.datetime.now()
Model.predict(test_x)
pred_time_end = datetime.datetime.now()
pred_time = pred_time_end - pred_time_start
print("<<<predtime calculation Done>>>")

model = Model
input_data = tf.random.normal([1,1, 1000, 3])

# 모델을 고정 그래프로 변환
@tf.function
def model_fn(x):
    return model(x)

concrete_func = model_fn.get_concrete_function(input_data)
frozen_func = convert_variables_to_constants_v2(concrete_func)

# 프로파일링 옵션 설정: max_depth와 출력 형식 지정
profiler_options = ProfileOptionBuilder.float_operation()
                    # .with_max_depth(5) \
                    #     .select(['micros', 'bytes']) \
                    #         .with_output_directory('profiler_logs') \
                    #             .build()
                    # 깊이를 5로 제한           
                    # 시간(microseconds)와 메모리(bytes) 정보 선택
                    # 출력 파일 지정
profiler_options['max_depth'] = 4
graph_info = profile(frozen_func.graph, options=profiler_options)
flops = graph_info.total_float_ops   
print("<<<FLOPs calcuation Done>>>")

# predict   
pred = Model.predict(x_test)

test_acc = (np.sum(np.sum( (pred > 0.5) == y_test, axis=1) == 8) / len(y_test) * 100)
np.save(f"./2.model/{foldername}/predict/{filename}_{date}_pred.npy", pred)
print("<<<Predict Done>>>")


# confusion matrix
threshold = 0.5
TF_pred = pred > threshold  # 일정값 이상으로 정리
TF_y = y_test > threshold  # 한 행당 [T,F,F,F] 이런식
class_num = 8

cm_all = pd.DataFrame(np.zeros([4, class_num]), columns=[i for i in range(class_num)])
cm_all.index = ["TP", "FP", "TN", "FN"] 

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

cm_all.columns = np.array(['ba', 'cs', 'na', 'background', 'co57', 'th', 'ra', 'am'])

new_row = ["accuracy", "recall", "specificity", "precision", "F1 Score"]
for row in new_row:
    cm_all.loc[row] = np.zeros(class_num)

cm_all.loc["accuracy"] = (cm_all.loc["TP"] + cm_all.loc["TN"]) / (cm_all.loc["TP"] + cm_all.loc["TN"] + cm_all.loc["FP"] + cm_all.loc["FN"])
cm_all.loc["recall"] = (cm_all.loc["TP"]) / (cm_all.loc["TP"] + cm_all.loc["FN"])
cm_all.loc["specificity"] = (cm_all.loc["TN"]) / (cm_all.loc["TN"] + cm_all.loc["FP"])
cm_all.loc["precision"] = (cm_all.loc["TP"]) / (cm_all.loc["TP"] + cm_all.loc["FP"])
cm_all.loc["F1 Score"] = 2*(cm_all.loc["TP"]) / (2*(cm_all.loc["TP"]) + (cm_all.loc["FP"]) + (cm_all.loc["FN"]))

cm_all = cm_all.drop('background', axis=1)

cm_all.to_csv(f'./2.model/{foldername}/predict/{filename}_{date}_results.csv', index=False)

    
with open(f"./2.model/{foldername}/{filename}_{date}.txt", 'a') as f:     
    trainable_params = sum([tf.size(variable).numpy() for variable in Model.trainable_variables])
    f.write(f"traiable_params : {trainable_params} \n")
    
    opti_params = sum([tf.size(variable).numpy() for variable in Model.optimizer.variables])
    f.write(f"optimizer_params : {opti_params} \n")
    
    f.write(f"total_params : {trainable_params+opti_params} \n")
    
    f.write(f"train_time : {fit_time_total} \n")
    
    f.write(f"FLOPs : {flops} \n")
    
    f.write(f"growth_rate : 12 \n")
    
    f.write(f"n_blocks : [2,4,8,4] \n")
    
    f.write(f"resolution : 1000 \n")
    
    f.write(f"pred time : {pred_time} \n")
    
    f.write(f"test acc : {test_acc}% \n")        
    
    f.write(f"Macro-average : precision-{sum(cm_all.loc['precision'])/7}, recall-{sum(cm_all.loc['precision'])/7}, F1 Score-{sum(cm_all.loc['F1 Score'])/7}")

    f.write(f"efficiency score = Accuracy / FLOPs : {str(test_acc/flops * 100000)} ")


