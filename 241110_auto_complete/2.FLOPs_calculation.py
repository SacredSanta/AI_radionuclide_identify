'''
flops 계산 함수 구성이 CPU 기반이라,
tensorflow 를 GPU로 설정해놓으면 작동이 멈추는 현상이 있었음.
단순히 cpu로 돌리면 금방 계산되어 나온다.
'''

#%%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys

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
test_make = 1
import tensorflow

if test_make:
    try:
        del Model
    except:
        pass
    
    try:
        del flops_
    except:
        pass
    
    Model = radionuclides_densenet(growth_rate=32, nblocks=[6,12,64,48], width_coef=1, depth_coef=1, resol_coef=1)
    Model.summary(line_length=100)
    
    
#%%
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2_as_graph,
)

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
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation() \
        .with_max_depth(10) \
            .build()
    flops = tf.compat.v1.profiler.profile(
        graph=frozen_func.graph, run_meta=run_meta, cmd="scope", options=opts
    )
    # print(frozen_func.graph.get_operations())
    # TODO: show each FLOPS
    return flops.total_float_ops


if __name__ == '__main__':
    flops_ = get_flops(Model, 32)
    print("FLOPs : ", flops_)
    
# %%
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

# 모델 정의
model = Model
input_data = tf.random.normal([1, 1000, 3])

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
profiler_options['max_depth'] = 5
#%%          

# 프로파일링 실행
graph_info = profile(frozen_func.graph, options=profiler_options)
print(f"FLOPs: {graph_info.total_float_ops}")
# %%
