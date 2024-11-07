'''
최종 수정 : 2024.08.06.
사용자 : 서동휘

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

def bottleneck(x, growth_rate, name, num):
    inner_channel = 4 * growth_rate
    h01 = tf.keras.layers.BatchNormalization(name=f"bottleneck_bn_{name}_{num}")(x)
    h01 = tf.keras.layers.ReLU(name=f"bottleneck_ReLU_{name}_{num}")(h01)
    h02 = tf.keras.layers.Conv1D(inner_channel, kernel_size=1, use_bias=False, name=f"bottleneck_con1d1_{name}_{num}")(h01)
    h02 = tf.keras.layers.BatchNormalization(name=f"bottleneck_bn2_{name}_{num}")(h02)
    h02 = tf.keras.layers.ReLU(name=f"bottleneck_ReLU2_{name}_{num}")(h02)
    h03 = tf.keras.layers.Conv1D(growth_rate, kernel_size=3, padding='same', use_bias=False, name=f"bottleneck_conv1d2_{name}_{num}")(h02)
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
    print("========Trans========", "out-",out_channels)
    print("t01 : ", tt1.shape)
    print("t02 : ", tt2.shape)
    print("t03 : ", tt3.shape)
    print("y : ", y.shape)
    print()
    return y

def dense_layer(x, nblocks, growth_rate, name):
    print("--====--==--== Dense layer --==--==--==--==", "phase-",name)
    for _ in range(nblocks):
        x = bottleneck(x, growth_rate, name, _)
        print("h : ", x.shape)
    print("--====--==--== ENd --==--==--==--==", "phase-",name)
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
        nblocks = list(map(lambda x: int(x*depth_coef), nblocks))
    
    # resol_coef
    if resol_coef != 1:
        rw = int(1000 * resol_coef)
        cl = int(3 * resol_coef)
    else:
        rw = 1000
        cl = 3    
        
    phase = 0
    
    # model build
    # input
    input_tensor = tf.keras.Input(shape=(rw,cl), name="input")
  
    x01_c = tf.keras.layers.Conv1D(inner_channels[phase], kernel_size=7, strides=2, padding='same', use_bias=False, name='x01_c')(input_tensor)
    x = tf.keras.layers.MaxPool1D(pool_size=7, strides=2, name='x01_p')(x01_c)  # pool size는 행,열    열을 7개씩 pool)
    #print("x01_c : ", x01_c.shape)
    #print("x01_p : ", x01_p.shape)
    
    avgpool_layers = [[] for i in range(len(nblocks)+1)]
    print("-- avg layer -- ", avgpool_layers)
    for phase in range(0, len(nblocks)):    
        # (1) Dense Block
        x = dense_layer(x, nblocks[phase], growth_rate, f'd0{phase}')
        inner_channels.append(inner_channels[2*phase]+growth_rate * nblocks[phase])
        print("first inner : ", inner_channels)


        # out 으로 나오는 channel 수 계산
        temp = 0
        print("phase : " , phase)
        for i in range(phase+1):
            temp += inner_channels[2*i+1]
            print("temp - ", temp)
            
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
        print("inner_channels : ", inner_channels)
        print("outer_channels : ", out_channels)
        print()

    
    x = tf.keras.layers.BatchNormalization(name="d04_BN")(x)
    x = tf.keras.layers.ReLU(name="d04_relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D(name="d04_global")(x)

    print("GlobalAveragePooling : ", x.shape)
    
    output_tensor = tf.keras.layers.Dense(num_class, activation='sigmoid', name='last_Dense')(x)
    model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)

    return model

#%% scaling spectrum
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

optimizers = [
    tf.keras.optimizers.Adadelta(),
    tf.keras.optimizers.Adafactor(),
    tf.keras.optimizers.Adagrad(),
    tf.keras.optimizers.Adam(),
    tf.keras.optimizers.AdamW(),
    tf.keras.optimizers.Adamax(),
    tf.keras.optimizers.Ftrl(),
    tf.keras.optimizers.Lion(),
    tf.keras.optimizers.Nadam(),
    tf.keras.optimizers.RMSprop(),
    tf.keras.optimizers.SGD()
]
#%% make model

Model = radionuclides_densenet(growth_rate=32, nblocks=[6,12,24,16], width_coef=1, depth_coef=1, resol_coef=1)
Model.summary(line_length=100)

#%% 2. model compile ====================================================================

#%%
Model.compile(optimizer=optimizers[3],
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['binary_accuracy'])

Model.summary()






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
filename = "241004_10to20sec_3series_merged"
model_data = np.load(f"./1.preprocess_data/{filename}_all.npz")

x_train = model_data["x_train"]#[:,:,:2]
y_train = model_data["y_train"]

x_val = model_data["x_val"]#[:,:,:2]
y_val = model_data["y_val"]


x_train = x_train[:,:,:]
y_train = y_train[:,0,:]
x_val = x_val[:,:,:]
y_val = y_val[:,0,:]

print(x_train.shape)  # 최종 형태는 (개수, 1, 1000, 3)
print(y_train.shape)  # 최종 형태는 (개수, source 개수)
print(x_val.shape)
print(y_val.shape)


#%% Models -------------------------------------------
# for comparisono of width
from itertools import  product

width_coefs = [1]
depth_coefs = [1]
resol_coefs = [1]
coefs = [width_coefs, depth_coefs, resol_coefs]
coefs_combis = list(product(*coefs)) 
epoch = 50

foldername = "241107_densenet1d_dd_combi"

# 모델 비교 반복
for coefs_combi in coefs_combis:
    try:
        del Model
    except:
        pass
    
    # 모델 생성
    Model = radionuclides_densenet(growth_rate=32,   # growth_rate:int, nblocks:list, width_coef=1, depth_coef=1, resol_coef=1
                                   nblocks=[],
                                   width_coef=coefs_combi[0], 
                                   depth_coef=coefs_combi[1],
                                   resol_coef=coefs_combi[2]
                                   )
    optimizer = optimizers[3]

    # 모델 컴파일
    Model.compile(optimizer=optimizer[3],
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['binary_accuracy'])

    # callback 지정
    date = f"241107_densenet1d_w{coefs_combi[0]}_d{coefs_combi[1]}" 
    csv_logger = tf.keras.callbacks.CSVLogger(f"./2.model/{foldername}/{filename}_{date}.log",
                                            separator=',',
                                            append=False)
    weights = tf.keras.callbacks.ModelCheckpoint(filepath=f"./2.model/{foldername}/{filename}_{date}_ckpt.keras",
                                                save_weights_only=False,
                                                verbose=1)
    
    fit_time_start = datetime.datetime.now()
    
    Model.fit(x_train,
            y_train,
            validation_data=(x_val,y_val),
            epochs=epoch,
            batch_size=100,
            callbacks=[csv_logger, weights]
            )
    fit_time_end = datetime.datetime.now()
    fit_time_total = fit_time_end - fit_time_start

    Model.save(f"./2.model/{foldername}/{filename}_{date}.keras")
    
    # TensorFlow 그래프 생성
    @tf.function
    def model_fn(x):
        return Model(x)

    # 임의의 입력 텐서 생성
    test_x = tf.random.normal((1000, 3))
    test_x = test_x[tf.newaxis,:,:]

    # 프로파일링을 위한 옵션 설정
    profiler_options = ProfileOptionBuilder.float_operation()  # FLOPs 계산 옵션
    graph_info = profile(model_fn.get_concrete_function(test_x).graph, options=profiler_options)

    # 결과 출력 (FLOPs)
    flops = graph_info.total_float_ops
    
    with open(f"./2.model/{foldername}/{filename}_{date}.txt", 'a') as f:     
        trainable_params = sum([tf.size(variable).numpy() for variable in Model.trainable_variables])
        f.write(f"traiable_params : {trainable_params} \n")
        
        opti_params = sum([tf.size(variable).numpy() for variable in optimizer.variables])
        f.write(f"optimizer_params : {opti_params} \n")
        
        f.write(f"total_params : {trainable_params+opti_params} \n")
        
        f.write(f"train_time : {fit_time_total} \n")
        
        f.write(f"FLOPs : {flops} \n")
        
        
        









#%% ==================================================================
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