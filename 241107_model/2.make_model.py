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
    h02 = tf.keras.layers.Conv2D(inner_channel, kernel_size=1, use_bias=False, name=f"bottleneck_con2d1_{name}_{num}")(h01)
    h02 = tf.keras.layers.BatchNormalization(name=f"bottleneck_bn2_{name}_{num}")(h02)
    h02 = tf.keras.layers.ReLU(name=f"bottleneck_ReLU2_{name}_{num}")(h02)
    h03 = tf.keras.layers.Conv2D(growth_rate, kernel_size=3, padding='same', use_bias=False, name=f"bottleneck_conv2d2_{name}_{num}")(h02)
    print("=======BottleNeck=======", inner_channel, growth_rate)
    # print("growth rate : ", growth_rate)
    print("inner channel : ", inner_channel)
    print("x : " , x.shape)
    print("h02 : ", h02.shape)
    print("h03 : ", h03.shape)
    return tf.keras.layers.Concatenate(name=f"bottleneck_concate_{name}_{num}")([x,h03])

def transition(x, out_channels, num, *layers):
    # 이전 Dense Block 정보들 넘겨주기
    for i in layers:
        x = tf.keras.layers.Concatenate(name=f"{num}_trans_concate_{i}")([x,i])
    
    tt1 = tf.keras.layers.BatchNormalization(name=f"trans_{num}_bn")(x)
    tt2 = tf.keras.layers.ReLU(name=f"trans_{num}_ReLU")(tt1)
    tt3 = tf.keras.layers.Conv2D(out_channels, kernel_size=1, use_bias=False, name=f"trans_{num}_conv2d")(tt2)  # reduction 적용된 filter 개수
    y = tf.keras.layers.AvgPool2D(pool_size=(1,2), strides=2, name=f'tran_{num}_avg')(tt3)
    print("========Trans========", out_channels)
    print("t01 : ", tt1.shape)
    print("t02 : ", tt2.shape)
    print("t03 : ", tt3.shape)
    print("y : ", y.shape)
    return y

def dense_layer(x, nblocks, growth_rate, name):
    print("--====--==--== Dense layer --==--==--==--==")
    for _ in range(nblocks):
        x = bottleneck(x, growth_rate, name, _)
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
    input_tensor = tf.keras.Input(shape=(1,1000,3), name="input")
  
    x01_c = tf.keras.layers.Conv2D(inner_channels[phase], kernel_size=7, strides=2, padding='same', use_bias=False, name='x01_c')(input_tensor)
    x01_p = tf.keras.layers.MaxPool2D(pool_size=(1,7), strides=2, name='x01_p')(x01_c)  # pool size는 행,열    열을 7개씩 pool)
    print("x01_c : ", x01_c.shape)
    print("x01_p : ", x01_p.shape)
    
    # Dense Block 1
    d01 = dense_layer(x01_p, nblocks[phase], growth_rate, 'd01')
    d01_p = tf.keras.layers.AvgPool2D(pool_size=(1,2), strides=2, name="d01_p")(d01)
    d01_pp = tf.keras.layers.AvgPool2D(pool_size=(1,2), strides=2, name="d01_pp")(d01_p)
    d01_ppp = tf.keras.layers.AvgPool2D(pool_size=(1,2), strides=2, name="d01_ppp")(d01_pp)
    print("d01 : ", d01.shape)
    print("d01_p : ", d01_p.shape)
    print("d01_pp : ", d01_pp.shape)
    print("d01_ppp : ", d01_ppp.shape)
    
    inner_channels.append(inner_channels[2*phase]+growth_rate * nblocks[phase])
    out_channels.append(int(reduction * inner_channels[2*phase+1]))
    print("<<<inner_channels>>> : ", inner_channels)
    print("<<<out_channels>>> : ", out_channels)
    
    # Trans 1
    t01 = transition(d01, out_channels[phase], 1)
    inner_channels.append(out_channels[phase])   # 현 filter 개수 저장용.

    print("\n\n")
    phase += 1
    
    # Dense Block 2
    d02 = dense_layer(t01, nblocks[phase], growth_rate, 'd02')
    d02_p = tf.keras.layers.AvgPool2D(pool_size=(1,2), strides=2, name="d02_p")(d02)
    d02_pp = tf.keras.layers.AvgPool2D(pool_size=(1,2), strides=2, name='d02_pp')(d02_p)
    print("d02 : ", d02.shape)
    print("d02_p : ", d02_p.shape)
    print("d02_pp : ", d02_pp.shape)
    
    inner_channels.append(inner_channels[2*phase] + growth_rate * nblocks[phase])
    out_channels.append(out_channels[phase-1] + int(reduction * inner_channels[2*phase+1]))
    print("<<<inner_channels>>> : ", inner_channels)
    print("<<<out_channels>>> : ", out_channels)
    
    # Trans 2
    t02 = transition(d02, out_channels[phase], 2, d01_p)
    inner_channels.append(out_channels[phase])
    
    print("\n\n")
    phase += 1

    # Dense Block 3
    d03 = dense_layer(t02, nblocks[phase], growth_rate, 'd03')
    d03_p = tf.keras.layers.AvgPool2D(pool_size=(1,2), strides=2, name="d03_p")(d03)
    
    print("d03 : ", d03.shape)
    print("d03_p : ", d03_p.shape)


    inner_channels.append(inner_channels[2*phase] + growth_rate * nblocks[phase])
    out_channels.append(out_channels[phase-1] + int(reduction * inner_channels[2*phase+1]))
    print("<<<inner_channels>>> : ", inner_channels)
    print("<<<out_channels>>> : ", out_channels)
    
    # Trans 3
    t03 = transition(d03, out_channels[phase], 3, d01_pp, d02_p)
    inner_channels.append(out_channels[phase])
    #sys.exit()
    print("\n\n")
    phase += 1



    # Dense Block 4
    d04 = dense_layer(t03, nblocks[phase], growth_rate, name='d04')
    print("d04 : ", d04.shape)
    
    inner_channels.append(inner_channels[2*phase] + growth_rate * nblocks[phase])
    out_channels.append(out_channels[phase-1] + int(reduction * inner_channels[2*phase+1]))
    print("<<<inner_channels>>> : ", inner_channels)
    print("<<<out_channels>>> : ", out_channels)
    
    for i in [d01_ppp, d02_pp, d03_p]:
        d04 = tf.keras.layers.Concatenate(name=f"d04_concate_{i}")([d04, i])
    print("final d04 : ", d04.shape)
    
    d04 = tf.keras.layers.BatchNormalization(name="d04_BN")(d04)
    d04 = tf.keras.layers.ReLU(name="d04_relu")(d04)
    d04 = tf.keras.layers.GlobalAveragePooling2D(name="d04_global")(d04)

    print("GlobalAveragePooling : ", d04.shape)
    

    output_tensor = tf.keras.layers.Dense(num_class, activation='sigmoid', name='last Dense')(d04)
    model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)

    return model

Model = radionuclides_densenet()
Model.summary(line_length=100)
#%% 2. model compile ====================================================================
Model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01, clipnorm=0.1, weight_decay=1e-4),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['binary_accuracy'])

Model.summary()



#%% 2. train the data - get data =======================================================
import numpy as np
#model_data = np.load("/tf/latest_version/3.AI/Tensorflow/Code/donghui_prac/spectrum_off_gate_2.npz")
filename = "241004_10to20sec_3series_merged"
model_data = np.load(f"./1.preprocess_data/{filename}_all.npz")

x_train = model_data["x_train"]#[:,:,:2]
y_train = model_data["y_train"]

x_val = model_data["x_val"]#[:,:,:2]
y_val = model_data["y_val"]
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

#%% =======================================================
print(x_train.shape)  # 최종 형태는 (개수, 1, 1000, 3)
print(y_train.shape)  # 최종 형태는 (개수, source 개수)
print(x_val.shape)
print(y_val.shape)
#%% 3. CSV Logger

date = "241021_densenet_vfrog" 
csv_logger = tf.keras.callbacks.CSVLogger(filename=f"./2.model/{filename}__{date}.log",
                                          separator=',',
                                          append=False)
weights = tf.keras.callbacks.ModelCheckpoint(filepath=f"./2.model/{filename}_{date}.keras",
                                             save_weights_only=False,
                                             verbose=1)
#%% (선택사항!) 기존 model 불러와서 추가학습이라면...
base_direc = "/tf/latest_version/3.AI/Tensorflow/Code/donghui_prac/2.model"
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
          epochs=50,
          batch_size=100,
          callbacks=[csv_logger, weights]
          )
fit_time_end = datetime.datetime.now()
fit_time_total = fit_time_end - fit_time_start
print("학습시간 : ", fit_time_total)

#%% 4. save model
from tensorflow import keras

#tf.keras.models.save_model(Model, f"./2.model/{filename}.h5")
Model.save(f"./2.model/{filename}__{date}.keras")

