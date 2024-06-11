#%% ---------------------------------------------------------------------------------
# ---------------------   Model Part   -------------------------------------
# --------------------------------------------------------------------

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


#%% 1. model definition

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
    # main parameters
    #? 각 Layer에서 몇 개의 feature map을 뽑을지 결정 - 각 layer가 전체 output에 어느정도 기여할지
    growth_rate = 12  
    inner_channels = 2 * growth_rate   # convolutio 이후 feature map 개수, filter 개수
    nblocks = [2,4,8,4]
    
    #? transition layer에서 반환하는 feature map
    reduction = 0.5
    
    #? (input 개수)Ba133, Cs137, Na22, Am241, Co60, Ga67, I131, Ra226, Tc99m, Th232, Tl201, 
    num_class = 7
    
    
    # model build
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


#%% 2. train the data - get data

#model_data = np.load("/tf/latest_version/3.AI/Tensorflow/Code/donghui_prac/spectrum_off_gate_2.npz")
filename = "10to20sec_7_source"
model_data = np.load(f"./1.preprocess_data/{filename}_all.npz")

x_train = model_data["x_train"]
y_train = model_data["y_train"]

x_val = model_data["x_val"]
y_val = model_data["y_val"]

#%% 2-1. 선택! newaxis가 필요하다면
x_train = x_train[:,tf.newaxis,:,:]
y_train = y_train[:,0,:]
x_val = x_val[:,tf.newaxis,:,:]
y_val = y_val[:,0,:]

print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)
#%% 3. CSV Logger
csv_logger = tf.keras.callbacks.CSVLogger(f"./2.model/{filename}.log", separator=',', append=False)

#%% 4. fit
# Nonetype 관련 오류가 뜰 때도 있는데 gpu가 안잡혀서 그럴 수도 있으니 커널 재시작해보고 다시해볼것.
Model.fit(x_train,
          y_train,
          validation_data=(x_val,y_val),
          epochs=30,
          callbacks=[csv_logger]
          )

#%% 4. save model

tf.keras.models.save_model(Model, f"./2.model/{filename}.h5")

# %%
