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

# 사용가능한 device 목록
print(device_lib.list_local_devices())
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)

# 메모리 제한
memory_limit = 23552
tf.config.experimental.set_virtual_device_configuration(
                physical_devices[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)]
            )

tf.config.set_visible_devices(physical_devices[0],'GPU')




#%% ===========================================================================================
# Encoder
def conv2d_block(input_tensor, iters, n_filters, kernel_size=3):
    '''
    Add 2 convolutional layers with the parameters
    ''' 
    x = input_tensor
    for i in range(iters):
        x = tf.keras.layers.Conv1D(filters=n_filters, kernel_size=kernel_size,
                                   kernel_initializer='he_normal', activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
    return x
 
def encoder_block(inputs, iters, n_filters,  pool_size=2, dropout=0.3):
    '''
    Add 2 convolutional blocks and then perform down sampling on output of convolutions
    '''
    # f는 처음 들어간 결과 - 추후 decoder 에 붙여줄 예정
    f = conv2d_block(inputs, iters, n_filters)  # -> (None, 1, 1000, 64)
    p = tf.keras.layers.MaxPooling1D(pool_size=pool_size)(f)  # -> (None, 0, 500, 64)
    p = tf.keras.layers.Dropout(dropout)(p)  # -> (None, 0, 500, 64)
    return f, p
 
def encoder(inputs, filters, iters):
    '''
    defines the encoder or downsampling path.
    '''
    
    f1, p1 = encoder_block(inputs, iters, n_filters=filters[0])
    f2, p2 = encoder_block(p1, iters, n_filters=filters[1])
    f3, p3 = encoder_block(p2, iters, n_filters=filters[2], pool_size=5)
    f4, p4 = encoder_block(p3, iters, n_filters=filters[3])
 
    # count = 0
    # print("=============Encoder===============")
    # for i in [(f1,p1),(f2,p2),(f3,p3),(f4,p4)]:
    #     count += 1
    #     print(f"f{count} : ", i[0].shape)
    #     print(f"p{count} : ", i[1].shape)
    
    return p4, (f1, f2, f3, f4)

# Bottlenect
def bottleneck(inputs, filters, iters):
    bottle_neck = conv2d_block(inputs, iters, n_filters=filters[4])
    return bottle_neck

# Decoder
def decoder_block(inputs, conv_output, iters, n_filters, kernel_size=3, strides=2, dropout=0.3, flag=0):
    '''
    defines the one decoder block of the UNet
    '''
    # transpose conv 는 upsampling 을 위함.
    u = tf.keras.layers.Conv1DTranspose(n_filters, kernel_size, strides, padding='same')(inputs)
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
    c = conv2d_block(c, iters, n_filters)
    
    return c
    
def decoder(inputs, convs, output_channels, filters, iters):
    '''
    Defines the decoder of the UNet chaining together 4 decoder blocks. 
    '''
    
    f1, f2, f3, f4 = convs

    c6 = decoder_block(inputs, f4, iters, n_filters=filters[3], kernel_size=3, strides=2)
    c7 = decoder_block(c6, f3, iters, n_filters=filters[2], kernel_size=3, strides=5)
    c8 = decoder_block(c7, f2, iters, n_filters=filters[1], kernel_size=3, strides=2)
    c9 = decoder_block(c8, f1, iters, n_filters=filters[0], kernel_size=3, strides=2)
    
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.GlobalAveragePooling1D()(c9)
    
    outputs = tf.keras.layers.Dense(output_channels, activation='sigmoid')(x)
    
    return outputs

OUTPUT_CHANNELS = 8

def UNet(filters, iters, width_coef, depth_coef, resol_coef):

    # width_coef
    if width_coef != 1:
        filters = list(map(lambda x: int(np.ceil(x*width_coef)), filters))
    
    # depth_coef
    if depth_coef != 1:
        iters = int(np.ceil(iters*depth_coef))
    
    if resol_coef != 1:
        rw = int(np.ceil(1000 * resol_coef))
        cl = 3
    else:
        rw = 1000
        cl = 3

    inputs = tf.keras.layers.Input(shape=(rw,cl))

    encoder_output, convs = encoder(inputs, filters, iters)
    #print("convs : ", convs)


    #print("===========bottleneck=============")
    bottle_neck = bottleneck(encoder_output, filters, iters)
    #print("bottle_neck : ", bottle_neck.shape)


    #print("===========decoder=============")
    outputs = decoder(bottle_neck, convs, OUTPUT_CHANNELS, filters, iters)

    model = tf.keras.Model(inputs, outputs)

    return model





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

Model = UNet(filters=[64,128,256,512,1024], iters=2, width_coef=0.2, depth_coef=1, resol_coef=0.2)
Model.summary()



#%% 2. train the data - get data =======================================================
import datetime
import numpy as np
from tensorflow import keras
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

#model_data = np.load("/tf/latest_version/3.AI/Tensorflow/Code/donghui_prac/spectrum_off_gate_2.npz")
#filename = "241004_10to20sec_3series_merged_orispectrum_normed_noisefiltered"
filename = '241113_count5000down_3series_merged_12000_all_orispec_normed'
#temp = "./Tensorflow/Code/donghui_prac/"
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



#%% coef combi
import numpy as np

# 0.05 단위로 1 이하의 값을 가지는 a, b, c 배열 생성
values = np.arange(0.5, 1.6, 0.1)

# 모든 조합을 저장할 리스트
coef_results = []

# 조건을 만족하는 조합 찾기
for w in values:
    for d in values:
        for r in values:
            product = w * r * d
            # product 값이 0.5에 가까운 경우를 찾음
            if abs(product - 1) <= 0.01:  # 오차 범위 ±0.05
                coef_results.append([round(w,2), round(d,2), round(r,2)])

# 결과 출력
print("조건을 만족하는 (a, b, c) 조합:")
for i in coef_results:
    print(i)
print(len(coef_results))


#%% phi combi
init_ = 0.1
interv = 0.05
phis = [round(i*interv + init_, 2) for i in range(0,8,1)]

coef_phis = []
# 1.15,1.35,1.3
for phi in phis:
    coef_phis.append([round((2.9)**phi,2),
                         round((1.0)**phi,2),
                         round((0.3)**phi,2)])

for i in phis:
    print(i)
for i in coef_phis:
    print(i)


#%% width, depth, resol combi
from itertools import  product
init_ = 0.5
interv = 0.1 
coeff = [round(i*interv + init_, 2) for i in range(0,16,1)]

width_coefs = coeff
depth_coefs = [1]
resol_coefs = [1]
coefs = [width_coefs, depth_coefs, resol_coefs]
coefs_combis_list = list(product(*coefs))

for i in coefs_combis_list:
    print(i)




#%%
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
from tensorflow.keras import backend as K


epoch = 30
foldername = f"241117_unet_combi/{filename}/"
coefs_combis = [[0.11,1,1]]
phi_flag = 0
filter=[64,128,256,512,1024]

# 모델 비교 반복
for coefs_combi in coefs_combis:
    K.clear_session()
    
    if phi_flag:
        phi_folder = 'phi_'+str(round(coefs_combis.index(coefs_combi)*interv,2)+init_)
        os.makedirs(f"./2.model/{foldername}/{phi_folder}", exist_ok=True)
    else:
        os.makedirs(f"./2.model/{foldername}", exist_ok=True)
    
    if phi_flag:
        inner_folder = f'{phi_folder}/{coefs_combi[0]},{coefs_combi[1]},{coefs_combi[2]}'
    else:
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
    Model = UNet(filters=filter,
                 iters=2,#growth_rate=growth_rates,   # growth_rate:int, nblocks:list, width_coef=1, depth_coef=1, resol_coef=1
                #nblocks=nblocks_params,
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
    # early Stopping 미사용.
    # early = tf.keras.callbacks.EarlyStopping(
    #             monitor='val_binary_accuracy', min_delta=0.01, patience=5, verbose=1, mode='max',
    #             baseline=None, restore_best_weights=False
    #             )

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
              callbacks=[csv_logger, weights]#, early]
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
        f.write(f"layers : {len(Model.layers)} \n")
        
        trainable_params = sum([tf.size(variable).numpy() for variable in Model.trainable_variables])
        f.write(f"traiable_params : {trainable_params} \n")
        
        opti_params = sum([tf.size(variable).numpy() for variable in optimizer.variables])
        f.write(f"optimizer_params : {opti_params} \n")
        
        f.write(f"total_params : {trainable_params+opti_params} \n")
        
        f.write(f"train_time : {fit_time_total} \n")
        
        f.write(f"FLOPs : {flops} \n")
        
        f.write(f"n_blocks : {list(map(lambda x: int(np.ceil(x*coefs_combi[0])), filter))} \n")
        
        f.write(f"iter : {np.ceil(2*coefs_combi[1])} \n")
        
        f.write(f"resolution : {int(1000*coefs_combi[2])} \n")
        
        f.write(f"pred time : {pred_time} \n")
        
        f.write(f"test acc : {test_acc}% \n")        
        
        f.write(f"Macro-average : precision-{sum(cm_all.loc['precision'])/7}, recall-{sum(cm_all.loc['precision'])/7}, F1 Score-{sum(cm_all.loc['F1 Score'])/7}")

        f.write(f"efficiency score = Accuracy / FLOPs : {str(test_acc/flops * 100000)} \n")

        f.write(f"epoch : {str(epoch)} \n")


sys.exit()
