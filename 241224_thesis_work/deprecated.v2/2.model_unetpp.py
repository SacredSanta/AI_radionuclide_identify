#%%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1' # 특정 GPU 사용 활성화
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
tf.config.set_visible_devices(physical_devices[0],'GPU')  # TITAN 으로 지정


#%% ===========================================================================================
# Encoder
def conv2d_block(input_tensor, n_filters, kernel_size=3, iter=2):
    '''
    Add 2 convolutional layers with the parameters
    ''' 
    x = input_tensor
    for i in range(iter):
        x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=kernel_size,
                                   kernel_initializer='he_normal', activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
    return x
 
def encoder_block(inputs, pool_size=(1,2), dropout=0.3):
    '''
    Add 2 convolutional blocks and then perform down sampling on output of convolutions
    '''
    # f 처음 들어간 결과 - 추후 decoder 에 붙여줄 예정
    #f = conv2d_block(inputs, n_filters)  # -> (None, 1, 1000, 64)
    p = tf.keras.layers.MaxPooling2D(pool_size=pool_size)(inputs)  # -> (None, 0, 500, 64)
    p = tf.keras.layers.Dropout(dropout)(p)  # -> (None, 0, 500, 64)
    
    return p

def decoder_block(inputs, kernel_size=3, strides=(1,2), dropout=0.3, front=1):
    '''
    defines the one decoder block of the UNet 
    '''
    n_filters = inputs.shape[3]
    u = tf.keras.layers.Conv2DTranspose(n_filters, kernel_size, strides, padding='same')(inputs)

    return u
 
 
def concate_block(x, *y):
    for i in y:
        x = tf.keras.layers.concatenate([x, i])
    return x
 
 
def UNetpp(inputs):
    '''
    defines the encoder or downsampling path.
    '''
    # Phase 1 - inputs ~ x01_af
    x00_cv = conv2d_block(inputs, n_filters=64)
    x10 = encoder_block(x00_cv)
    #x00_cvt = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x00_cv, axis=0),  # shape 맞춰주기위함
    #                            output_shape=(0,x00_cv.shape[2],x00_cv.shape[3]))(x00_cv)
    x10_af = conv2d_block(x10, n_filters=128)
    x01_con = concate_block(x00_cv, decoder_block(x10_af))
    x01_af = conv2d_block(x01_con, n_filters=64)
    print("x00_cv : " , x00_cv.shape)
    print("x10 : " , x10.shape)
    print("x10_af : ", x10_af.shape)
    print("x01_con : ", x01_con.shape)
    print("x01_af : ", x01_af.shape)
    
    # Phase 2 - x20 ~ x02_af
    x20 = encoder_block(x10_af)
    x20_af = conv2d_block(x20, n_filters=256)
    x11_con = concate_block(x10_af, decoder_block(x20_af))
    x11_af = conv2d_block(x11_con, n_filters=128)
    x02_con = concate_block(x00_cv, x01_af, decoder_block(x11_af))
    x02_af = conv2d_block(x02_con, n_filters=64)
    print("x20 : ", x20.shape)
    print("x20_af : ", x20_af.shape)
    print("x11_con : ", x11_con.shape)
    print("x11_af : ", x11_af.shape)
    print("x02_con : ", x02_con.shape)
    print("x02_af : ", x02_af.shape)
    
    # Phase 3 - x30 ~ x03_af
    x30 = encoder_block(x20_af, pool_size=(1,5))
    x30_af = conv2d_block(x30, n_filters=512)
    x21_con = concate_block(x20_af, decoder_block(x30_af, strides=(1,5)))  # 50 -> 250
    x21_af = conv2d_block(x21_con, n_filters=256)
    x12_con = concate_block(x10_af, x11_af, decoder_block(x21_af))
    x12_af = conv2d_block(x12_con, n_filters=128)
    x03_con = concate_block(x00_cv, x01_af, x02_af, decoder_block(x12_af))
    x03_af = conv2d_block(x03_con, n_filters=64)
    print("x30 : ", x30.shape)
    print("x30_af : ", x30_af.shape)
    print("x21_con : ", x21_con.shape)
    print("x21_af : ", x21_af.shape)
    print("x12_con : ", x12_con.shape)
    print("x03_con : ", x03_con.shape)
    print("x03_af : ", x03_af.shape)
    
    # Phase 4 - x40 ~ x04_af 
    x40 = encoder_block(x30_af)
    x40_af = conv2d_block(x40, n_filters=1024)
    x31_con = concate_block(x30_af, decoder_block(x40_af))
    x31_af = conv2d_block(x31_con, n_filters=512)
    x22_con = concate_block(x20_af, x21_af, decoder_block(x31_af, strides=(1,5)))  # 50 -> 250
    x22_af = conv2d_block(x22_con, n_filters=256)
    x13_con = concate_block(x10_af, x11_af, x12_af, decoder_block(x22_af))
    x13_af = conv2d_block(x13_con, n_filters=128)
    x04_con = concate_block(x00_cv, x01_af, x02_af, x03_af, decoder_block(x13_af))
    x04_af = conv2d_block(x04_con, n_filters=64)
    print("x40 : ", x40.shape)
    print("x40_af : ", x40_af.shape)
    print("x31_con : ", x31_con.shape)
    print("x31_af : ", x31_af.shape)
    print("x22_con : ", x22_con.shape)
    print("x22_af : ", x22_af.shape)
    print("x13_con : ", x13_con.shape)
    print("x13_af : ", x13_af.shape)
    print("x04_con : ", x04_con.shape)
    print("x04_af : ", x04_af.shape)

    
    # End Phase
    x04_af_gm = tf.keras.layers.GlobalAveragePooling2D()(x04_af)
    xx = tf.keras.layers.Dense(8)(x04_af_gm)
    print("x04_af_gm : ", x04_af_gm.shape)
    
    # count = 0
    # print("=============Encoder===============")
    # for i in [(x00_af,x10),(x10_af,x20),(x20_af,x30),(x30_af,x40)]:
    #     count += 1
    #     print(f"f{count} : ", i[0].shape)
    #     print(f"p{count} : ", i[1].shape)
    
    return xx


inputs = tf.keras.layers.Input(shape=(1,1000,3))
outputs = UNetpp(inputs)
Model = tf.keras.Model(inputs, outputs)

Model.summary()


#%%
Model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01, clipnorm=0.1, weight_decay=1e-4),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['binary_accuracy'])


#%% 2. train the data - get data =======================================================
import numpy as np
#model_data = np.load("/tf/latest_version/3.AI/Tensorflow/Code/donghui_prac/spectrum_off_gate_2.npz")
filename = "241004_10to20sec_3series_merged"
model_data = np.load(f"./1.preprocess_data/{filename}_all.npz")

x_train = model_data["x_train"]#[:,:,:2]
y_train = model_data["y_train"]

x_val = model_data["x_val"]#[:,:,:2]
y_val = model_data["y_val"]


x_train = x_train[:,tf.newaxis,:,:]
y_train = y_train[:,0,:]
x_val = x_val[:,tf.newaxis,:,:]
y_val = y_val[:,0,:]

print(x_train.shape)  # 최종 형태는 (개수, 1, 1000, 3)
print(y_train.shape)  # 최종 형태는 (개수, source 개수)
print(x_val.shape)
print(y_val.shape)

#%% 3. CSV Logger

date = "241004_unetpp" 
csv_logger = tf.keras.callbacks.CSVLogger(f"./2.model/{filename}__{date}.log", separator=',', append=False)

#%% 4. fit
import datetime
fit_time_start = datetime.datetime.now()
# Nonetype 관련 오류가 뜰 때도 있는데 gpu가 안잡혀서 그럴 수도 있으니 커널 재시작해보고 다시해볼것.
Model.fit(x_train,
          y_train,
          validation_data=(x_val,y_val),
          epochs=50,
          batch_size=100,
          #callbacks=[csv_logger]
          )
fit_time_end = datetime.datetime.now()
fit_time_total = fit_time_end - fit_time_start
print("학습시간 : ", fit_time_total)
#%% 4. save model
from tensorflow import keras

#tf.keras.models.save_model(Model, f"./2.model/{filename}.h5")
Model.save(f"./2.model/{filename}__{date}.keras")
