'''
최종 수정 : 2024.08.06.
사용자 : 서동휘

<수정 내용> 

<24.08.06>
- model input 을 (? x 1000 x 3) 으로 변경
'''
#%%
#%% Model Part ==============================================================================================================================
#%% 0. init 
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
print(os.environ)

# Library
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2_as_graph,
)


# Check GPU
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
tf.config.set_visible_devices(physical_devices[0],'GPU')

# 0. functions ========================================================================================================================================
debug = 0

def bottleneck(x, growth_rate, name, num):
    inner_channel = 4 * growth_rate
    h01 = tf.keras.layers.BatchNormalization(name=f"bottleneck_bn_{name}_{num}")(x)
    h01 = tf.keras.layers.ReLU(name=f"bottleneck_ReLU_{name}_{num}")(h01)
    h02 = tf.keras.layers.Conv1D(inner_channel, kernel_size=1, use_bias=False, name=f"bottleneck_con1d1_{name}_{num}")(h01)
    h02 = tf.keras.layers.BatchNormalization(name=f"bottleneck_bn2_{name}_{num}")(h02)
    h02 = tf.keras.layers.ReLU(name=f"bottleneck_ReLU2_{name}_{num}")(h02)
    h03 = tf.keras.layers.Conv1D(growth_rate, kernel_size=3, padding='same', use_bias=False, name=f"bottleneck_conv1d2_{name}_{num}")(h02)
    if debug:
        print("=======BottleNeck=======", "in-",inner_channel, "grow-",growth_rate)
        # print("growth rate : ", growth_rate)
        print("inner channel : ", inner_channel)
        print("x : " , x.shape)
        print("h02 : ", h02.shape)
        print("h03 : ", h03.shape)
    return tf.keras.layers.Concatenate(name=f"bottleneck_concate_{name}_{num}")([x,h03])

def transition(x, out_channels, num, *layers):
    # 이전 Dense Block 정보들 넘겨주기
    ttemp = 0
    for i in layers:
        ttemp += 1
        x = tf.keras.layers.Concatenate(name=f"trans_concate_{num}_{ttemp}")([x,i])
    
    tt1 = tf.keras.layers.BatchNormalization(name=f"trans_{num}_bn")(x)
    tt2 = tf.keras.layers.ReLU(name=f"trans_{num}_ReLU")(tt1)
    tt3 = tf.keras.layers.Conv1D(out_channels, kernel_size=1, use_bias=False, name=f"trans_{num}_conv2d")(tt2)  # reduction 적용된 filter 개수
    y = tf.keras.layers.AvgPool1D(pool_size=2, strides=2, name=f'tran_{num}_avg')(tt3)
    if debug:
        print("========Trans========", "out-",out_channels)
        print("t01 : ", tt1.shape)
        print("t02 : ", tt2.shape)
        print("t03 : ", tt3.shape)
        print("y : ", y.shape)
        print()
    return y

def dense_layer(x, nblocks, growth_rate, name):
    if debug: print("--====--==--== Dense layer --==--==--==--==", "phase-",name)
    for _ in range(nblocks):
        x = bottleneck(x, growth_rate, name, _)
        if debug: print("h : ", x.shape)
    if debug: print("--====--==--== ENd --==--==--==--==", "phase-",name)
    return x


# 1. model definition (version 1) 

def radionuclides_densenet(growth_rate:int, nblocks:list, width_coef=1, depth_coef=1, resol_coef=1):
    
    # main parameters
    #? 각 Layer에서 몇 개의 feature map을 뽑을지 결정 - 각 layer가 전체 output에 어느정도 기여할지
    out_channels = []
    inner_channels = [2 * growth_rate]   # convolution 이후 feature map 개수, filter 개수
    
    #? transition layer에서 반환하는 feature map
    reduction = 0.5 # 논문에서는 0.5 #0.2
    
    #? source = np.array(['ba', 'cs', 'na', 'background', 'co57', 'th', 'ra', 'am'])
    num_class = 8
    
    
    # width_coef
    if width_coef != 1:
        growth_rate = int(growth_rate * width_coef)
    
    # depth_coef
    if depth_coef != 1:
        nblocks = list(map(lambda x: int(np.ceil(x*depth_coef)), nblocks))
    
    # resol_coef
    if resol_coef != 1:
        rw = int(1000 * resol_coef)
        cl = 3
    else:
        rw = 1000
        cl = 3    
        
    phase = 0
    
    # model build
    # input
    input_tensor = tf.keras.Input(shape=(rw, cl), name="input")
  
    x01_c = tf.keras.layers.Conv1D(inner_channels[phase], kernel_size=7, strides=2, padding='same', use_bias=False, name='x01_c')(input_tensor)
    x = tf.keras.layers.MaxPool1D(pool_size=7, strides=2, name='x01_p')(x01_c)  # pool size는 행,열    열을 7개씩 pool)
    #print("x01_c : ", x01_c.shape)
    #print("x01_p : ", x01_p.shape)
    
    avgpool_layers = [[] for i in range(len(nblocks)+1)]
    if debug: print("-- avg layer -- ", avgpool_layers)
    for phase in range(0, len(nblocks)):    
        # (1) Dense Block
        x = dense_layer(x, nblocks[phase], growth_rate, f'd0{phase}')
        inner_channels.append(inner_channels[2*phase]+growth_rate * nblocks[phase])
        if debug: print("first inner : ", inner_channels)


        # out 으로 나오는 channel 수 계산
        temp = 0
        if debug: print("phase : " , phase)
        for i in range(phase+1):
            temp += inner_channels[2*i+1]
            if debug: print("temp - ", temp)
            
        if phase == len(nblocks)-1:
            out_channels.append(int(temp))
        else:
            out_channels.append(int(reduction * temp))
            
            
        # 추후에 붙여줄 block을 위한 pooling 과정
        temp_x = tf.keras.layers.AvgPool1D(pool_size=2, strides=2)(x)
        for i in range(phase, len(nblocks)):
            if i == phase:
                pass
            else:
                temp_x = tf.keras.layers.AvgPool1D(pool_size=2, strides=2)(temp_x)
            # 각 리스트마다 현재 layer pooling 을 하나씩 중첩하면서 각 리스트에 넣기
            avgpool_layers[i+1].append(temp_x)


        # avgpool debug용
        if debug:
            print(" -- avg layer -- ")
            for ii in range(len(nblocks)+1):
                print(ii, avgpool_layers[ii])
        
        # final phase면 transition skip
        if phase == len(nblocks)-1:
            for i in avgpool_layers[phase]:
                x = tf.keras.layers.Concatenate()([x,i])
            break

        
        # (2) transition
        x = transition(x, out_channels[phase], phase+1, *avgpool_layers[phase])  # transition 에서는 avgpool_layers 한 단계 아래 참조
        inner_channels.append(out_channels[phase])   # 현 filter 개수 저장용.
        if debug:
            print("inner_channels : ", inner_channels)
            print("outer_channels : ", out_channels)
            print()

    
    x = tf.keras.layers.BatchNormalization(name="d04_BN")(x)
    x = tf.keras.layers.ReLU(name="d04_relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D(name="d04_global")(x)

    if debug: print("GlobalAveragePooling : ", x.shape)
    
    output_tensor = tf.keras.layers.Dense(num_class, activation='sigmoid', name='last_Dense')(x)
    model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)

    return model


# ---------------------------------
def scale_rows(array, row_scale):
    
    # 원본 배열의 크기
    original_rows, original_cols = array.shape

    # 새로운 행(row) 크기 계산
    new_rows = int(original_rows * row_scale)

    # 확장된 배열 생성 (행 방향으로만 확장, 열은 기존과 동일)

    expanded_array = np.zeros((new_rows, original_cols), dtype=array.dtype)

    # 행 방향 보간
    for col in range(original_cols):
        expanded_array[:, col] = np.interp(
            np.linspace(0, original_rows - 1, new_rows),  # 새 행 x_coordi
            np.arange(original_rows),                     # 기존 행 x_coordi
            array[:, col]                                 # 각 열의 값 y_coordi
        )

    return expanded_array

def get_flops(model, batch_size = None) -> int:
    """
    Calculate FLOPS for tf.keras.Model or tf.keras.Sequential .
    Ignore operations used in only training mode such as Initialization.
    Use tf.profiler of tensorflow v1 api.
    """
    # convert tf.keras model into frozen graph to count FLOPS about operations used at inference
    # FLOPS depends on batch size
    inputs = [
        tf.TensorSpec([batch_size] + list(inp.shape[1:]), inp.dtype) for inp in model.inputs
    ]
    real_model = tf.function(model).get_concrete_function(inputs)
    frozen_func, _ = convert_variables_to_constants_v2_as_graph(real_model)

    # Calculate FLOPS with tf.profiler
    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.compat.v1.profiler.profile(
        graph=frozen_func.graph, run_meta=run_meta, cmd="scope", options=opts
    )
    # print(frozen_func.graph.get_operations())
    # TODO: show each FLOPS
    return flops.total_float_ops

# optimizers = [
#     tf.keras.optimizers.Adadelta(),
#     tf.keras.optimizers.Adafactor(),
#     tf.keras.optimizers.Adagrad(),
#     tf.keras.optimizers.Adam(),
#     tf.keras.optimizers.AdamW(),
#     tf.keras.optimizers.Adamax(),
#     tf.keras.optimizers.Ftrl(),
#     tf.keras.optimizers.Lion(),
#     tf.keras.optimizers.Nadam(),
#     tf.keras.optimizers.RMSprop(),
#     tf.keras.optimizers.SGD()
# ]
#%% make model (for test) ! 필요시만 ! ---------------------------------------
test_make = 0

if test_make:
    Model = radionuclides_densenet(growth_rate=1, nblocks=[6,12,24,16], width_coef=1, depth_coef=1, resol_coef=1)
    Model.summary(line_length=100)
    

#%% 2. model compile  ! 필요시만 ! --------------------------------------------------------
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

if test_make:
    Model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['binary_accuracy'])

    test_x = tf.random.normal((1000, 3))
    test_x = test_x[tf.newaxis,:,:]
   
    get_flops(Model,32)
    # model_fn = tf.function(Model)
    # concrete_func = model_fn.get_concrete_function(tf.TensorSpec(test_x.shape, test_x.dtype))

    # #frozen_func = convert_variables_to_constants_v2(concrete_func)
    # #frozen_func.graph.as_graph_def()

    # # 프로파일링을 위한 옵션 설정
    # profiler_options = ProfileOptionBuilder.float_operation()  # FLOPs 계산 옵션
    # graph_info = profile(model_fn.get_concrete_function(test_x).graph, options=profiler_options)


#%% ============================================================
# all-in-one 
# ==============================================================

#%% 2. train the data - get data =======================================================
import datetime
import numpy as np
from tensorflow import keras
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

#model_data = np.load("/tf/latest_version/3.AI/Tensorflow/Code/donghui_prac/spectrum_off_gate_2.npz")
filename = "241004_10to20sec_3series_merged_orispectrum_normed_noisefiltered"
model_data = np.load(f"./1.preprocess_data/{filename}.npz")

x_train = model_data["x_train"]#[:,:,:2]
y_train = model_data["y_train"]

x_val = model_data["x_val"]#[:,:,:2]
y_val = model_data["y_val"]

x_test = model_data["x_test"]
y_test = model_data["y_test"]

x_train = x_train[:,:,:]
y_train = y_train[:,0,:]
x_val = x_val[:,:,:]
y_val = y_val[:,0,:]
x_test = x_test[:,:,:]
y_test = y_test[:,0,:]

print(x_train.shape)  # 최종 형태는 (개수, 1, 1000, 3)
print(y_train.shape)  # 최종 형태는 (개수, source 개수)
print(x_val.shape)
print(y_val.shape)
print(x_test.shape)
print(y_test.shape)

#%% Models -------------------------------------------
# for comparisono of width
from itertools import  product
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
import tensorflow as tf
import seaborn as sns
import numpy as np
import os
import pandas as pd
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder


width_coefs = [1,0.8,0.6,0.4,0.2,0.1,0.05]#[0.8,0.6,0.4,0.2,0.1,0.05,0.03]
depth_coefs = [0.6,0.4]
resol_coefs = [1]
coefs = [width_coefs, depth_coefs, resol_coefs]
coefs_combis = list(product(*coefs))
epoch = 30
nblocks_params = [6,12,24,16] # [6,12,24,16] [6,12,32,32] [6,12,48,32] [6,12,36,24]
foldername = "241107_densenet1d_dd_combi/241004_10to20sec_3series_merged_orispectrum_normed_noisefiltered/block6,12,24,16"

# 모델 비교 반복
for coefs_combi in coefs_combis:
    inner_folder = f'{coefs_combi[0]},{coefs_combi[1]},{coefs_combi[2]}'
    os.makedirs(f"./2.model/{foldername}/{inner_folder}", exist_ok=True)
    os.makedirs(f"./2.model/{foldername}/{inner_folder}/predict", exist_ok=True)
    try:
        del Model
    except:
        pass
    
    try:
        del cm_all
    except:
        pass
    
    # 모델 생성
    Model = radionuclides_densenet(growth_rate=32,   # growth_rate:int, nblocks:list, width_coef=1, depth_coef=1, resol_coef=1
                                   nblocks=nblocks_params,
                                   width_coef=coefs_combi[0], 
                                   depth_coef=coefs_combi[1],
                                   resol_coef=coefs_combi[2]
                                   )
    optimizer = tf.keras.optimizers.Adam()    
    # 모델 컴파일
    Model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['binary_accuracy'])


    # callback 지정
    date = f"241107_densenet1d_w{coefs_combi[0]}_d{coefs_combi[1]}_r{coefs_combi[2]}" 
    csv_logger = tf.keras.callbacks.CSVLogger(f"./2.model/{foldername}/{inner_folder}/{filename}_{date}.log",
                                            separator=',',
                                            append=False)
    weights = tf.keras.callbacks.ModelCheckpoint(filepath=f"./2.model/{foldername}/{inner_folder}/{filename}_{date}_ckpt.keras",
                                                save_weights_only=False,
                                                verbose=1)
    
    fit_time_start = datetime.datetime.now()
    
    # resolution 관련 설정
    if coefs_combi[2] != 1:
        rw = int(1000 * coefs_combi[2])
        
        scaled_xtrain = np.zeros((len(x_train), rw, 3))
        scaled_xval = np.zeros((len(x_val), rw, 3))
        scaled_xtest = np.zeros((len(x_test), rw, 3))
        
        for i in range(len(x_train)):
            if i % 1000 == 0 : print("x_train rescale prcessed in ", i)
            scaled_xtrain[i] = scale_rows(x_train[i], coefs_combi[2])  
        
        for i in range(len(x_val)):
            if i % 1000 == 0 : print("x_val rescale processed in ", i)
            scaled_xval[i] = scale_rows(x_val[i], coefs_combi[2])
            
        for i in range(len(x_test)):
            if i % 1000 == 0 : print("x_test rescale processed in ", i)
            scaled_xtest[i] = scale_rows(x_test[i], coefs_combi[2])
    else:
        scaled_xtrain = x_train
        scaled_xval = x_val
        scaled_xtest = x_test
        rw = 1000
    
    # FLOPs 계산
    # 모델 정의
    model = Model
    input_data = tf.random.normal([1, rw, 3])

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
    
    
        
    Model.fit(scaled_xtrain,
              y_train,
              validation_data=(scaled_xval,y_val),
              epochs=epoch,
              batch_size=32,
              callbacks=[csv_logger, weights]
              )
    
    fit_time_end = datetime.datetime.now()
    fit_time_total = fit_time_end - fit_time_start

    Model.save(f"./2.model/{foldername}/{inner_folder}/{filename}_{date}.keras")
    print("<<<Model Saved>>>")

    
    test_x = tf.random.normal(shape=(1,rw,3))
    pred_time_start = datetime.datetime.now()
    Model.predict(test_x)
    pred_time_end = datetime.datetime.now()
    pred_time = pred_time_end - pred_time_start
    print("<<<predtime calculation Done>>>")
    
    
    # predict   
    pred = Model.predict(scaled_xtest)

    test_acc = (np.sum(np.sum( (pred > 0.5) == y_test, axis=1) == 8) / len(y_test) * 100)
    np.save(f"./2.model/{foldername}/{inner_folder}/predict/{filename}_{date}_pred.npy", pred)
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
    
    cm_all.to_csv(f'./2.model/{foldername}/{inner_folder}/predict/{filename}_{date}_results.csv', index=False)
    
        
    with open(f"./2.model/{foldername}/{inner_folder}/{filename}_{date}.txt", 'a') as f:     
        trainable_params = sum([tf.size(variable).numpy() for variable in Model.trainable_variables])
        f.write(f"traiable_params : {trainable_params} \n")
        
        opti_params = sum([tf.size(variable).numpy() for variable in optimizer.variables])
        f.write(f"optimizer_params : {opti_params} \n")
        
        f.write(f"total_params : {trainable_params+opti_params} \n")
        
        f.write(f"train_time : {fit_time_total} \n")
        
        f.write(f"FLOPs : {flops} \n")
        
        f.write(f"growth_rate : {int(32 * coefs_combi[0])} \n")
        
        f.write(f"n_blocks : {int(np.ceil(nblocks_params[0] * coefs_combi[1]))} {int(np.ceil(nblocks_params[1] * coefs_combi[1]))} {int(np.ceil(nblocks_params[2] * coefs_combi[1]))} {int(np.ceil(nblocks_params[3] * coefs_combi[1]))} \n")
        
        f.write(f"resolution : {int(1000*coefs_combi[2])} \n")
        
        f.write(f"pred time : {pred_time} \n")
        
        f.write(f"test acc : {test_acc}% \n")        
        
        f.write(f"Macro-average : precision-{sum(cm_all.loc['precision'])/7}, recall-{sum(cm_all.loc['precision'])/7}, F1 Score-{sum(cm_all.loc['F1 Score'])/7}")






sys.exit()


#%% =================================================================
# re normalize the original spectrum
# ===================================================================

#%% test

import numpy as np

filename = "241004_10to20sec_3series_merged"
model_data = np.load(f"./1.preprocess_data/{filename}_all.npz")

x_train = model_data["x_train"]#[:,:,:2]

#%%
xx = x_train[0]

resol_coef = 1.5
after_shape = [int(i*resol_coef) for i in xx.shape]

def scale_rows(array, row_scale):
    
    # 원본 배열의 크기
    original_rows, original_cols = array.shape
    
    # 새로운 행(row) 크기 계산
    new_rows = int(original_rows * row_scale)
    
    # 확장된 배열 생성 (행 방향으로만 확장, 열은 기존과 동일)
    expanded_array = np.zeros((new_rows, original_cols), dtype=array.dtype)
    
    # 행 방향 보간
    for col in range(original_cols):
        expanded_array[:, col] = np.interp(
            np.linspace(0, original_rows - 1, new_rows),  # 새 행 위치
            np.arange(original_rows),                      # 기존 행 위치
            array[:, col]                                  # 각 열의 값
        )
    
    return expanded_array

# 사용 예제
original_array = xx  # (1000, 3) 배열 생성
scaled_array = scale_rows(original_array, row_scale=1.5)  # 행을 1.5배 확장
print(scaled_array.shape)  # 출력: (1500, 3)

plt.subplot(1,2,1)
plt.plot(xx)
plt.subplot(1,2,2)
plt.plot(scaled_array)


#%%
from joblib import Parallel, delayed, cpu_count

filename = "241004_10to20sec_3series_merged"
model_data = np.load(f"./1.preprocess_data/{filename}_all.npz")
#%%

x_train = model_data["x_train"]
y_train = model_data["y_train"]

x_val = model_data["x_val"]
y_val = model_data["y_val"]

x_test = model_data["x_test"]
y_test = model_data["y_test"]
#%%
def normalize_0_to_1(num):
    global x_train, x_val, x_test
    
    # 데이터의 최소값과 최대값
    data = x_train[num,:,0]
    
    min_val = np.min(data)
    max_val = np.max(data)
    
    # 정규화: (data - min) / (max - min)
    normalized_data = (data - min_val) / (max_val - min_val)
    
    return normalized_data

# 사용 예제
dt = x_test

results = Parallel(n_jobs=cpu_count(), verbose=10)(delayed(normalize_0_to_1)(i) for i in range(len(dt)))
results = np.array(results)

x_test[:,:,0] = results


#%%
newfilename = filename+"_norm-ori_241107"
np.savez(f"./1.preprocess_data/{newfilename}_all.npz", 
         x_train=x_train,
         y_train=y_train,
         x_val=x_val,
         y_val=y_val,
         x_test=x_test,
         y_test=y_test)