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


debug = 1
def radionuclides_densenet2d():
    # main parameters
    #? 각 Layer에서 몇 개의 feature map을 뽑을지 결정 - 각 layer가 전체 output에 어느정도 기여할지
    growth_rate = 12  
    inner_channels = 2 * growth_rate   # convolutio 이후 feature map 개수, filter 개수
    nblocks = [2,4,8,4]
    
    #? transition layer에서 반환하는 feature map
    reduction = 0.5
    
    #? (input 개수)Ba133, Cs137, Na22, Am241, Co60, Ga67, I131, Ra226, Tc99m, Th232, Tl201, 
    num_class = 4
    
    
    # model build
    input_tensor = tf.keras.Input(shape=(1000, 1000, 1))
    x = tf.keras.layers.Conv2D(inner_channels, 
                               kernel_size=7, 
                               strides=2, 
                               padding='same', 
                               use_bias=False)(input_tensor)
    if debug : print(x.shape)
    x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2)(x)  # pool size는 행,열    열을 7개씩 pool)
    if debug : print(x.shape)

    x = dense_layer(x, nblocks[0], growth_rate) # nblocks[0] = 2
    if debug : print(x.shape)
    inner_channels += growth_rate * nblocks[0]
    out_channels = int(reduction * inner_channels) # out_channles = 12
    x = transition(x, out_channels)
    if debug : print(x.shape)
    inner_channels = out_channels

    x = dense_layer(x, nblocks[1], growth_rate)
    if debug : print(x.shape)
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
    if debug : print(output_tensor.shape)
    model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)

    return model

Model = radionuclides_densenet2d()

#%%
Model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=0.1, weight_decay=1e-4),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['binary_accuracy'])

Model.summary()

import numpy as np
#model_data = np.load("/tf/latest_version/3.AI/Tensorflow/Code/donghui_prac/spectrum_off_gate_2.npz")
filename = "240624_1to100sec_4source_1000_acc0"
x = np.load(f"./0.dataset/{filename}.npy")
y = np.load(f"./1.preprocess_data/{filename}_y.npy")
#%% 2-0 (선택지) normalize
def normalize_img(img):
    min_ = np.min(img)
    max_ = np.max(img)
    epsilon = 1e-8
    return (img - min_) / (max_ - min_ + epsilon)
x = np.array([normalize_img(x[i,:,:]) for i in range(len(x))])

#%% 2-1 (선택지!) reallocate the axis
x = np.transpose(x, (1, 2, 0))
y = np.transpose(y, (2,1,0))
#%% 2-2 channle 추가
x_train = x[0:600,:,:,tf.newaxis]   # (600,1000,1000)
y_train = y[0:600,0,:]   # (600,4)

x_val = x[600:800,:,:,tf.newaxis]
y_val = y[600:800,0,:] 

print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)

#%% 3. CSV Logger
csv_logger = tf.keras.callbacks.CSVLogger(f"./2.model/{filename}.log", separator=',', append=False)

#%% 4. fit
# Nonetype 관련 오류가 뜰 때도 있는데 gpu가 안잡혀서 그럴 수도 있으니 커널 재시작해보고 다시해볼것.
Model.fit(x = x_train,
          y = y_train,
          validation_data=(x_val,y_val),
          epochs=30,
          callbacks=[csv_logger]
          )
print("Model Fit completed!")

#%% 4. save model

tf.keras.models.save_model(Model, f"./2.model/{filename}.h5")

# %%




#%% test

model_cnn = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same', input_shape=(10, 10, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu',  padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(4)  # 4개의 출력 유닛
])

model_cnn.compile(optimizer='adam', loss='mean_squared_error')  # 회귀 문제를 가정하여 손실 함수를 설정
model_cnn.summary()

# %%
xx_train = np.random.rand(600, 10, 10, 1).astype(np.float32)  # 600개의 1000x1000 흑백 이미지
yy_train = np.random.rand(600, 4).astype(np.float32)  # 각 이미지에 대한 4개의 값
# %%
model_cnn.fit(xx_train, yy_train, epochs=10, batch_size=32, validation_split=0.2)
# %%
