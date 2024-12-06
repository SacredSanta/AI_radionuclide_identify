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
def conv2d_block(input_tensor, n_filters, kernel_size=3):
    '''
    Add 2 convolutional layers with the parameters
    ''' 
    x = input_tensor
    for i in range(2):
        x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=kernel_size,
                                   kernel_initializer='he_normal', activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
    return x
 
def encoder_block(inputs, n_filters=64, pool_size=(1,2), dropout=0.3):
    '''
    Add 2 convolutional blocks and then perform down sampling on output of convolutions
    '''
    # f는 처음 들어간 결과 - 추후 decoder 에 붙여줄 예정
    f = conv2d_block(inputs, n_filters)  # -> (None, 1, 1000, 64)
    p = tf.keras.layers.MaxPooling2D(pool_size=pool_size)(f)  # -> (None, 0, 500, 64)
    p = tf.keras.layers.Dropout(dropout)(p)  # -> (None, 0, 500, 64)
    return f, p
 
def encoder(inputs):
    '''
    defines the encoder or downsampling path.
    '''
    
    f1, p1 = encoder_block(inputs, n_filters=64)
    f2, p2 = encoder_block(p1, n_filters=128)
    f3, p3 = encoder_block(p2, n_filters=256, pool_size=(1,5))
    f4, p4 = encoder_block(p3, n_filters=512)
 
    count = 0
    print("=============Encoder===============")
    for i in [(f1,p1),(f2,p2),(f3,p3),(f4,p4)]:
        count += 1
        print(f"f{count} : ", i[0].shape)
        print(f"p{count} : ", i[1].shape)
    
    return p4, (f1, f2, f3, f4)

# Bottlenect
def bottleneck(inputs):
    bottle_neck = conv2d_block(inputs, n_filters=1024)
    return bottle_neck

# Decoder
def decoder_block(inputs, conv_output, n_filters=64, kernel_size=3, strides=2, dropout=0.3, flag=0):
    '''
    defines the one decoder block of the UNet
    '''
    
    # transpose conv 는 upsampling 을 위함.
    u = tf.keras.layers.Conv2DTranspose(n_filters, kernel_size, strides, padding='same')(inputs)
    print("u.shape : ", u.shape)
    print("conv_output : ", conv_output.shape)
    if flag:
        shape_info = u.shape
        u = tf.keras.layers.Lambda(lambda x: tf.expand_dims(u, axis=0),  # shape 맞춰주기위함
                                output_shape=(1,shape_info[2],shape_info[3]))(u)
        
    # else:
    #     shape_info = u.shape
    #     if shape_info[1]:
            
    c = tf.keras.layers.concatenate([u, conv_output])
    c = tf.keras.layers.Dropout(dropout)(c)
    c = conv2d_block(c, n_filters)
    
    return c
    
def decoder(inputs, convs, output_channels):
    '''
    Defines the decoder of the UNet chaining together 4 decoder blocks. 
    '''
    
    f1, f2, f3, f4 = convs

    c6 = decoder_block(inputs, f4, n_filters=512, kernel_size=3, strides=(1,2))
    c7 = decoder_block(c6, f3, n_filters=256, kernel_size=3, strides=(1,5))
    c8 = decoder_block(c7, f2, n_filters=128, kernel_size=3, strides=(1,2))
    c9 = decoder_block(c8, f1, n_filters=64, kernel_size=3, strides=(1,2))
    
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.GlobalAveragePooling2D()(c9)
    
    outputs = tf.keras.layers.Dense(output_channels, activation='sigmoid')(x)
    
    return outputs

OUTPUT_CHANNELS = 8

def UNet():

    inputs = tf.keras.layers.Input(shape=(1,1000,3))

    encoder_output, convs = encoder(inputs)
    print("convs : ", convs)


    print("===========bottleneck=============")
    bottle_neck = bottleneck(encoder_output)
    print("bottle_neck : ", bottle_neck.shape)


    print("===========decoder=============")
    outputs = decoder(bottle_neck, convs, OUTPUT_CHANNELS)

    model = tf.keras.Model(inputs, outputs)

    return model


Model = UNet()
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

date = "241024_unet" 
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

