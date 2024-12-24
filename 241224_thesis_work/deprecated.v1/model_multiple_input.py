'''
최종 수정 : 2024.08.06.
사용자 : 서동휘

<수정 내용> 

<24.08.06>
- model input 을 (? x 1000 x 3) 으로 변경
'''

#%% Model Part ==========================================================
#%% 0. init 
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

#%% 0. functions ====================================================================

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
    x = tf.keras.layers.Conv2D(out_channels, kernel_size=1, use_bias=False)(x)  # reduction 적용된 filter 개수
    return tf.keras.layers.AvgPool2D(pool_size=(1,2), strides=2)(x)

def dense_layer(x, nblocks, growth_rate):
    for _ in range(nblocks):
        x = bottleneck(x, growth_rate)
    return x


#%% 1. model definition (version 1) ====================================================================

def radionuclides_densenet():
    
    # main parameters
    #? 각 Layer에서 몇 개의 feature map을 뽑을지 결정 - 각 layer가 전체 output에 어느정도 기여할지
    growth_rate = 12  
    out_channels = 0
    inner_channels = 2 * growth_rate   # convolutio 이후 feature map 개수, filter 개수
    nblocks = [2,4,8,4]
    
    #? transition layer에서 반환하는 feature map
    reduction = 0.2
    
    #? (input 개수)Ba133, Cs137, Na22, Am241, Co60, Ga67, I131, Ra226, Tc99m, Th232, Tl201, 
    num_class = 7
    
    
    # model build
    input_tensor = tf.keras.Input(shape=(1,1000,3))
  
    x = tf.keras.layers.Conv2D(inner_channels, kernel_size=15, strides=2, padding='same', use_bias=False)(input_tensor)
    x = tf.keras.layers.MaxPool2D(pool_size=(1,7), strides=2)(x)  # pool size는 행,열    열을 7개씩 pool)

    x = dense_layer(x, nblocks[0], growth_rate)
    inner_channels += growth_rate * nblocks[0]
    out_channels = int(reduction * inner_channels)
    x = transition(x, out_channels)
    inner_channels = out_channels   # 현 filter 개수 저장용.

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



#%% 1. model (version 2)  ====================================================================

def multi_input_dense():
    growth_rate = 12  
    inner_channels = 2 * growth_rate   # convolutio 이후 feature map 개수, filter 개수
    nblocks = [2,4,8,4]
    reduction = 0.5
    num_class = 7 # ['ba133', 'cs137’, 'na22’,'background’, 'co57', 'th232', 'ra226']
    drop_rate = 0.5


    input_tensor = tf.keras.Input(shape=(1,1000,3))
    
    # 1
    input_tensor_x = input_tensor[:,:,:,0,tf.newaxis]
    x = tf.keras.layers.Conv2D(inner_channels, kernel_size=15, strides=2, padding='same', use_bias=False)(input_tensor_x)
    x = tf.keras.layers.MaxPool2D(pool_size=(1,7), strides=2)(x)  # pool size는 행,열    열을 7개씩 pool)

    x = dense_layer(x, nblocks[0], growth_rate)
    inner_channels += growth_rate * nblocks[0]
    out_channels = int(reduction * inner_channels)
    x = transition(x, out_channels)
    inner_channels = out_channels   # 현 filter 개수 저장용.

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

    # 2
    inner_channels = 2 * growth_rate # inner_channel 의 초기화
    input_tensor_y = input_tensor[:,:,:,1,tf.newaxis]
    y = tf.keras.layers.Conv2D(inner_channels, kernel_size=15, strides=2, padding='same', use_bias=False)(input_tensor_y)
    y = tf.keras.layers.MaxPool2D(pool_size=(1,7), strides=2)(y)  # pool size는 행,열    열을 7개씩 pool)

    y = dense_layer(y, nblocks[0], growth_rate)
    inner_channels += growth_rate * nblocks[0]
    out_channels = int(reduction * inner_channels)
    y = transition(y, out_channels)
    inner_channels = out_channels   # 현 filter 개수 저장용.

    y = dense_layer(y, nblocks[1], growth_rate)
    inner_channels += growth_rate * nblocks[1]
    out_channels = int(reduction * inner_channels)
    y = transition(y, out_channels)
    inner_channels = out_channels

    y = dense_layer(y, nblocks[2], growth_rate)
    inner_channels += growth_rate * nblocks[2]
    out_channels = int(reduction * inner_channels)
    y = transition(y, out_channels)
    inner_channels = out_channels

    y = dense_layer(y, nblocks[3], growth_rate)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.ReLU()(y)
    y = tf.keras.layers.GlobalAveragePooling2D()(y)

    # 3
    inner_channels = 2 * growth_rate # inner_channel 의 초기화
    input_tensor_z = input_tensor[:,:,:,2,tf.newaxis]
    z = tf.keras.layers.Conv2D(inner_channels, kernel_size=15, strides=2, padding='same', use_bias=False)(input_tensor_z)
    z = tf.keras.layers.MaxPool2D(pool_size=(1,7), strides=2)(z)  # pool size는 행,열    열을 7개씩 pool)

    z = dense_layer(z, nblocks[0], growth_rate)
    inner_channels += growth_rate * nblocks[0]
    out_channels = int(reduction * inner_channels)
    z = transition(z, out_channels)
    inner_channels = out_channels   # 현 filter 개수 저장용.

    z = dense_layer(z, nblocks[1], growth_rate)
    inner_channels += growth_rate * nblocks[1]
    out_channels = int(reduction * inner_channels)
    z = transition(z, out_channels)
    inner_channels = out_channels

    z = dense_layer(z, nblocks[2], growth_rate)
    inner_channels += growth_rate * nblocks[2]
    out_channels = int(reduction * inner_channels)
    z = transition(z, out_channels)
    inner_channels = out_channels

    z = dense_layer(z, nblocks[3], growth_rate)
    z = tf.keras.layers.BatchNormalization()(z)
    z = tf.keras.layers.ReLU()(z)
    z = tf.keras.layers.GlobalAveragePooling2D()(z)
    
    re = tf.keras.layers.Concatenate()([x,y,z])
    re = tf.keras.layers.Dropout(rate=drop_rate)(re)
    output_tensor = tf.keras.layers.Dense(num_class, activation='sigmoid')(re)
    model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
    
    return model


Model = multi_input_dense()
Model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=0.1, weight_decay=1e-4),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['binary_accuracy'])

Model.summary()


#%% 2. train the data - get data =======================================================
import numpy as np
#model_data = np.load("/tf/latest_version/3.AI/Tensorflow/Code/donghui_prac/spectrum_off_gate_2.npz")
filename = "240827_240806-merged_8000"
model_data = np.load(f"./1.preprocess_data/{filename}_all.npz")

x_train = model_data["x_train"]#[:,:,:2]
y_train = model_data["y_train"]

x_val = model_data["x_val"]#[:,:,:2]
y_val = model_data["y_val"]
#%% (Debug) Nan이 있다면?  =======================================================
print(np.isnan(x_train).any())
nan_indices = np.where(np.isnan(x_train))
print(len(nan_indices[0]))

#%% 2-1. 선택! newaxis가 필요하다면 =======================================================
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

date = "240830_2" 
csv_logger = tf.keras.callbacks.CSVLogger(f"./2.model/{filename}__{date}.log", separator=',', append=False)

#%% 4. fit
# Nonetype 관련 오류가 뜰 때도 있는데 gpu가 안잡혀서 그럴 수도 있으니 커널 재시작해보고 다시해볼것.
Model.fit(x_train,
          y_train,
          validation_data=(x_val,y_val),
          epochs=50,
          callbacks=[csv_logger]
          )

#%% 4. save model
from tensorflow import keras

#tf.keras.models.save_model(Model, f"./2.model/{filename}.h5")
Model.save(f"./2.model/{filename}__{date}.keras")

# %%
