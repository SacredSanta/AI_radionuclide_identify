'''
-241121
디펜스에 실제 사용된 efficientnet 코드


'''
#%%
!pip install tensorflow==2.17
!pip install numpy matplotlib scikit-learn seaborn
#%% ========================================================================================================================================
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # 특정 GPU 사용 활성화
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

#%% ========================================================================================================================================
import os
import math
sys.path.append("./lib")

CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        # EfficientNet actually uses an untruncated normal distribution for
        # initializing conv layers, but keras.initializers.VarianceScaling use
        # a truncated distribution.
        # We decided against a custom initializer for better serializability.
        'distribution': 'normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}

blocks_arg_2 = [
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 4, 'filters_out': 2,
     'expand_ratio': 1, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 2, 'filters_in': 2, 'filters_out': 3,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 2, 'filters_in': 3, 'filters_out': 5,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 3, 'filters_in': 5, 'filters_out': 10,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 3, 'filters_in': 10, 'filters_out': 14,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 4, 'filters_in': 14, 'filters_out': 24,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 24, 'filters_out': 40,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25}
]


# mobile inverted residual block   -----------------------------------
def block(inputs, drop_rate=0., name='',
          filters_in=32, filters_out=16, kernel_size=3, strides=1,
          expand_ratio=1, se_ratio=0., id_skip=True):
    """A mobile inverted residual block.

    # Arguments
        inputs: input tensor.
        activation_fn: activation function.
        drop_rate: float between 0 and 1, fraction of the input units to drop.
        name: string, block label.
        filters_in: integer, the number of input filters.
        filters_out: integer, the number of output filters.
        kernel_size: integer, the dimension of the convolution window.
        strides: integer, the stride of the convolution.
        expand_ratio: integer, scaling coefficient for the input filters.
        se_ratio: float between 0 and 1, fraction to squeeze the input filters.
        id_skip: boolean.

    # Returns
        output tensor for the block.
    """
    # image 면 channel 3개니까 마지막 axis 의 index가 몇인지 지칭하는 듯 함.
    bn_axis = 2
    
    # (1) Expansion phase - 보통 1로서 그대로 들어감
    filters = filters_in * expand_ratio

    if expand_ratio != 1:
        x = layers.Conv1D(filters,
                          kernel_size=1,
                          padding='same',
                          use_bias=False,
                          kernel_initializer=CONV_KERNEL_INITIALIZER,
                          name=name + 'expand_conv')(inputs)
        x = layers.BatchNormalization(axis=bn_axis, name=name + 'expand_bn')(x)
        x = activations.silu(x)
    else:
        x = inputs

    # (2) Depthwise Convolution - channel 하나 자체로만 연산을 하도록
    x = layers.DepthwiseConv1D(kernel_size,
                               strides=strides,
                               padding='same',
                               use_bias=False,
                               depthwise_initializer=CONV_KERNEL_INITIALIZER,
                               name=name + 'dwconv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=name + 'bn')(x)
    x = activations.silu(x)
    
    # (3) Squeeze and Excitation phase
    if 0 < se_ratio <= 1:
        filters_se = max(1, int(filters_in * se_ratio))
        se = layers.GlobalAveragePooling1D(name=name + 'se_squeeze')(x)
        if bn_axis == 1:
            se = layers.Reshape((filters, 1, 1), name=name + 'se_reshape')(se)
        else:
            se = layers.Reshape((1, filters), name=name + 'se_reshape')(se)
        se = layers.Conv1D(filters_se, 1,
                           padding='same',
                           kernel_initializer=CONV_KERNEL_INITIALIZER,
                           name=name + 'se_reduce')(se)
        se = activations.silu(se)
        se = layers.Conv1D(filters, 1,
                           padding='same',
                           kernel_initializer=CONV_KERNEL_INITIALIZER,
                           name=name + 'se_expand')(se)
        se = activations.silu(se)
        x = layers.multiply([x, se], name=name + 'se_excite')
    
    # (4) Output phase
    x = layers.Conv1D(filters_out, 1,
                      padding='same',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name=name + 'project_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=name + 'project_bn')(x)
    
    if (id_skip is True and strides == 1 and filters_in == filters_out):
        if drop_rate > 0:
            x = layers.Dropout(drop_rate,
                               noise_shape=(None, 1, 1),
                               name=name + 'drop')(x)
        x = layers.add([x, inputs], name=name + 'add')

    return x



layers = tf.keras.layers
activations = tf.keras.activations
debug = 1
# ----------------------------------------------------------------------
def EfficientNet(width_coef,
                 depth_coef,
                 resol_coef,
                 drop_connect_rate=0.2,
                 depth_divisor=2,
                 blocks_args=None,
                 model_name='efficientnet',
                 **kwargs):

    # Input 형태 취득
    if resol_coef != 1:
        rw = int(np.ceil(1000 * resol_coef))
        cl = 3
    else:
        rw = 1000
        cl = 3
    img_input = layers.Input(shape=(rw,cl))
    
    # width_coefficient
    def round_filters(filters, divisor=depth_divisor): # depth_divisor : default=8
        """Round number of filters based on depth multiplier."""
        filters *= width_coef  # 변환시키려는 filter 수 에 ,  width_계수 곱해주기
        new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)  # 새로운 값은, depth_divisor 통해서 나누기
        # Make sure that round down does not go down by more than 10%.
        if new_filters < 0.9 * filters:  # 기존 filter 수 보다 90%미만이면
            new_filters += divisor  # 새 filter 수에 divisor 를 합쳐준다
        return int(new_filters)

    # depth_coefficient
    
    def round_repeats(repeats):
        """Round number of repeats based on depth multiplier."""
        return int(math.ceil(depth_coef * repeats))

    
    # <<<<  1. Build stem  >>>>  ----------------------------
    x = layers.Conv1D(round_filters(blocks_args[0]["filters_in"]), 
                      kernel_size=3,
                      strides=2,
                      padding='same',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name='stem_conv')(img_input)
    x = layers.BatchNormalization(axis=2, name='stem_bn')(x)
    x = activations.silu(x)

    # <<<<  2. Build blocks  >>>>  ----------------------------
    from copy import deepcopy
    blocks_args = deepcopy(blocks_args) # 제어하다가 변경할 수 있으니, 기존 건드리지 않게 deepcopy 이용하는듯 
    # block_args : 
        #* example
        #*{'kernel_size': 3, 'repeats': 1, 'filters_in': 32, 'filters_out': 16,
        #* 'expand_ratio': 1, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
        
    b = 0
    blocks = float(sum(args['repeats'] for args in blocks_args))
    
    filter_info_a = []
    filter_info_b = []
    
    for (i, args) in enumerate(blocks_args):
        assert args['repeats'] > 0    # assert 는 조건문 검토
        # Update block input and output filters based on depth multiplier.
        args['filters_in'] = round_filters(args['filters_in'])
        args['filters_out'] = round_filters(args['filters_out'])

        filter_info_a.append(args['filters_in'])
        filter_info_b.append(args['filters_out'])
        
        for j in range(round_repeats(args.pop('repeats'))):
            # The first block needs to take care of stride and filter size increase.
            if j > 0:
                args['strides'] = 1
                args['filters_in'] = args['filters_out']
            x = block(x, drop_connect_rate * b / blocks,
                      name='block{}{}_'.format(i + 1, chr(j + 97)), **args)
            b += 1
        
    # Build top  ----------------------------
    x = layers.Conv1D(round_filters(blocks_args[-1]["filters_out"]*4),
                      kernel_size=1,
                      padding='same',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name='top_conv')(x)
    x = layers.BatchNormalization(axis=2, name='top_bn')(x)
    x = activations.silu(x)
    x = layers.GlobalMaxPooling1D(name='max_pool')(x)
    output = layers.Dense(units=8, activation='sigmoid')(x)

    # Create model.
    model = tf.keras.models.Model(img_input, output, name=model_name)

    return model, filter_info_a, filter_info_b

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


#%% all-in-one ===========================================================================================================================================================================================
import numpy as np
#model_data = np.load("/tf/latest_version/3.AI/Tensorflow/Code/donghui_prac/spectrum_off_gate_2.npz")
#filename = "241004_10to20sec_3series_merged_orispectrum_normed_noisefiltered"

# defense data
#filename = '241113_count5000down_3series_merged_12000_all_orispec_normed'

# 241121 data
#filename = "241121_set15000_min500_max1000_norm_all"
filename = "241121_set15000_min1000_max2000_norm_all"
#filename = "241121_set15000_min2000_max3000_norm_all"
#filename = "241121_set15000_min3000_max4000_norm_all"
#filename = "241121_set15000_min4000_max5000_norm_all"
#filename = "241121_set15000_min5000_max6000_norm_all"


model_data = np.load(f"./1.preprocess_data/241121/final/{filename}.npz")

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

#%% ======================================================================================================================================================
blocks_arg_full_ori = [ # original
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 32, 'filters_out': 16,
     'expand_ratio': 1, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 2, 'filters_in': 16, 'filters_out': 24,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 2, 'filters_in': 24, 'filters_out': 40,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 3, 'filters_in': 40, 'filters_out': 80,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 3, 'filters_in': 80, 'filters_out': 112,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 4, 'filters_in': 112, 'filters_out': 192,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 192, 'filters_out': 320,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25}
]

blocks_arg_2 = [ # filter 수 /8 버전
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 4, 'filters_out': 2,
     'expand_ratio': 1, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 2, 'filters_in': 2, 'filters_out': 3,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 2, 'filters_in': 3, 'filters_out': 5,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 3, 'filters_in': 5, 'filters_out': 10,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 3, 'filters_in': 10, 'filters_out': 14,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 4, 'filters_in': 14, 'filters_out': 24,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 24, 'filters_out': 40,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25}
]

blocks_arg_3 = [  # filter 수 /4 버전
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 8, 'filters_out': 4,
     'expand_ratio': 1, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 4, 'filters_out': 6,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 1, 'filters_in': 6, 'filters_out': 10,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 2, 'filters_in': 10, 'filters_out': 20,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 2, 'filters_in': 20, 'filters_out': 28,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 3, 'filters_in': 28, 'filters_out': 48,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 48, 'filters_out': 80,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25}
]

blocks_arg_4 = [ # MBN Block 4개 이용
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 8, 'filters_out': 4,
     'expand_ratio': 1, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 4, 'filters_out': 6,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 1, 'filters_in': 6, 'filters_out': 10,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 2, 'filters_in': 10, 'filters_out': 20,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    # {'kernel_size': 5, 'repeats': 2, 'filters_in': 20, 'filters_out': 28,
    #  'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    # {'kernel_size': 5, 'repeats': 3, 'filters_in': 28, 'filters_out': 48,
    #  'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    # {'kernel_size': 3, 'repeats': 1, 'filters_in': 48, 'filters_out': 80,
    #  'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25}
]

blocks_arg_5 = [ # MBN Block 5개 이용
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 8, 'filters_out': 4,
     'expand_ratio': 1, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 4, 'filters_out': 6,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 1, 'filters_in': 6, 'filters_out': 10,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 2, 'filters_in': 10, 'filters_out': 20,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 2, 'filters_in': 20, 'filters_out': 28,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    # {'kernel_size': 5, 'repeats': 3, 'filters_in': 28, 'filters_out': 48,
    #  'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    # {'kernel_size': 3, 'repeats': 1, 'filters_in': 48, 'filters_out': 80,
    #  'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25}
]

blocks_arg_6 = [ # MBN Block 6개 이용
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 8, 'filters_out': 4,
     'expand_ratio': 1, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 4, 'filters_out': 6,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 1, 'filters_in': 6, 'filters_out': 10,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 2, 'filters_in': 10, 'filters_out': 20,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 2, 'filters_in': 20, 'filters_out': 28,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 3, 'filters_in': 28, 'filters_out': 48,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    # {'kernel_size': 3, 'repeats': 1, 'filters_in': 48, 'filters_out': 80,
    #  'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25}
]

blocks_arg_6 = [ # MBN Block 6개 이용
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 8, 'filters_out': 4,
     'expand_ratio': 1, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 4, 'filters_out': 6,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 1, 'filters_in': 6, 'filters_out': 10,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 2, 'filters_in': 10, 'filters_out': 20,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 2, 'filters_in': 20, 'filters_out': 28,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 3, 'filters_in': 28, 'filters_out': 48,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    # {'kernel_size': 3, 'repeats': 1, 'filters_in': 48, 'filters_out': 80,
    #  'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25}
]

blocks_arg_7 = [ # MBN Block 6개 이용
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 8, 'filters_out': 4,
     'expand_ratio': 1, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 4, 'filters_out': 6,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 1, 'filters_in': 6, 'filters_out': 10,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 2, 'filters_in': 10, 'filters_out': 20,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 2, 'filters_in': 20, 'filters_out': 28,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 3, 'filters_in': 28, 'filters_out': 48,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    # {'kernel_size': 3, 'repeats': 1, 'filters_in': 48, 'filters_out': 80,
    #  'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25}
]

blocks_arg_8 = [ # original - 4개
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 32, 'filters_out': 16,
     'expand_ratio': 1, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 2, 'filters_in': 16, 'filters_out': 24,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 2, 'filters_in': 24, 'filters_out': 40,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 3, 'filters_in': 40, 'filters_out': 80,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    # {'kernel_size': 5, 'repeats': 3, 'filters_in': 80, 'filters_out': 112,
    #  'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    # {'kernel_size': 5, 'repeats': 4, 'filters_in': 112, 'filters_out': 192,
    #  'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    # {'kernel_size': 3, 'repeats': 1, 'filters_in': 192, 'filters_out': 320,
    #  'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25}
]

blocks_arg_9 = [ # original - 3개
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 32, 'filters_out': 16,
     'expand_ratio': 1, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 2, 'filters_in': 16, 'filters_out': 24,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 2, 'filters_in': 24, 'filters_out': 40,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    # {'kernel_size': 3, 'repeats': 3, 'filters_in': 40, 'filters_out': 80,
    #  'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    # {'kernel_size': 5, 'repeats': 3, 'filters_in': 80, 'filters_out': 112,
    #  'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    # {'kernel_size': 5, 'repeats': 4, 'filters_in': 112, 'filters_out': 192,
    #  'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    # {'kernel_size': 3, 'repeats': 1, 'filters_in': 192, 'filters_out': 320,
    #  'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25}
]

#%% coef combi - finding efficient combi ========================================================================================================================================
import numpy as np

# 0.05 단위로 1 이하의 값을 가지는 a, b, c 배열 생성
values = np.arange(0.5, 1.6, 0.1)

# 모든 조합을 저장할 리스트
coef_results = []

# 조건을 만족하는 조합 찾기
for w in values:
    for d in values:
        for r in values:
            if d > 1:
                continue
            product = w * r * d
            # product 값이 0.5에 가까운 경우를 찾음
            if (product - 1) <= 0 and (product - 1) >= -0.5:  # 오차 범위 ±0.05
                coef_results.append([round(w,2), round(d,2), round(r,2)])

# 결과 출력
print("조건을 만족하는 (a, b, c) 조합:")
for i in coef_results:
    print(i)
print(len(coef_results))








#%% phi combi ========================================================================================================================================
init_ = 1.5
interv = 0.5
phis = [round(i*interv + init_, 2) for i in range(0,5,1)]

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


#%% width, depth, resol combi ========================================================================================================================================
from itertools import  product
init_ = 0.5
interv = 0.1 
coeff = [round(i*interv + init_, 2) for i in range(0,16,1)]

width_coefs = [1.0]
depth_coefs = coeff
resol_coefs = [1.0]
coefs = [width_coefs, depth_coefs, resol_coefs]
coefs_combis_list = list(product(*coefs))

for i in coefs_combis_list:
    print(i)

#%% Models ========================================================================================================================================
# for comparisono of width
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
import copy
import random
import datetime
import numpy as np
from tensorflow import keras
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
import sys


epoch = 40
coefs_combis = [[1.0,1.0,1.0] for i in range(10)]
phi_flag = 0
blocks_arg_full = copy.deepcopy(blocks_arg_6)

# if data iteration mode ------
data_iteration_mode = 0  # <----- flag check 잘하기
data_count = 0
data_files = [#"241129_set15000_min500_max1000_norm_all",
              #"241129_set15000_min1000_max2000_norm_all",
              #"241129_set15000_min2000_max3000_norm_all",
              #"241129_set15000_min3000_max4000_norm_all",
              #"241129_set15000_min4000_max5000_norm_all",
              "241129_set15000_min5000_max6000_norm_all"]
if data_iteration_mode: # for 문 간소화를 위해 잠시 coefs_combis 변환
    coefs_combis = range(len(data_files))
#-------------------------------

seed = 42
# np.random.seed(seed)
tf.random.set_seed(seed)
# random.seed(seed)
seedtry = 0

# 모델 비교 반복
for coefs_combi in coefs_combis:  # <---------- 중간에 끊겼다면 ceofs_combis 에서 indexing, phi 부분 폴더 이름 순서가 섞일 수 있음
    K.clear_session()
    seedtry += 1
    # data iteration mode ------------------------------------------------------------------------

    if data_iteration_mode:
        try: del filename,model_data,x_train,y_train,x_val,y_val,x_test,y_test
        except: pass
    
        coefs_combi = [1.0,1.0,1.0]  # <--- 바뀐 coefs_combi 원복.
        filename = data_files[data_count]

        model_data = np.load(f"./1.preprocess_data/{filename}.npz")

        x_train = model_data["x_train"]
        y_train = model_data["y_train"][:,0,:]
        x_val = model_data["x_val"]
        y_val = model_data["y_val"][:,0,:]
        x_test = model_data["x_test"]
        y_test = model_data["y_test"][:,0,:]
        
        data_count += 1
    # --------------------------------------------------------------------------------------------
    foldername = f"241121_effinet_combi/{filename}/blocks_arg6/default_seedfix/{seedtry}"
    
    if os.path.isdir(f"./2.model/{foldername}/{coefs_combi[0]},{coefs_combi[1]},{coefs_combi[2]}"):
        print(f"{coefs_combi[0]},{coefs_combi[1]},{coefs_combi[2]} coef passed")
        continue
    
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
    
    try: del Model, cm_all, optimizer, pred
    except: pass    
           
    # 모델 생성 ------------------------------------------------------------------------------------------------------------
    Model, filter_info_a, filter_info_b = EfficientNet(width_coef=coefs_combi[0], 
                                                       depth_coef=coefs_combi[1],
                                                       resol_coef=coefs_combi[2],
                                                       blocks_args=blocks_arg_full,
                                                       classes=8)
    optimizer = tf.keras.optimizers.Adam()

    # 모델 컴파일
    Model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['binary_accuracy'])

    # callback 지정
    date = f"241202_effinet1d_w{coefs_combi[0]}_d{coefs_combi[1]}_r{coefs_combi[2]}" 
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
    
    #dataset = tf.data.Dataset.from_tensor_slices((scaled_xtrain, y_train))
    #dataset = dataset.shuffle(buffer_size=9000).batch(32).repeat()
    
    
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
        f.write(f"layers : {len(Model.layers)} \n")
        
        trainable_params = sum([tf.size(variable).numpy() for variable in Model.trainable_variables])
        f.write(f"traiable_params : {trainable_params} \n")
        
        opti_params = sum([tf.size(variable).numpy() for variable in optimizer.variables])
        f.write(f"optimizer_params : {opti_params} \n")
        
        f.write(f"total_params : {trainable_params+opti_params} \n")
        
        f.write(f"train_time : {fit_time_total} \n")
        
        f.write(f"FLOPs : {flops} \n")
        
        f.write(f"width_coef : {coefs_combi[0]} \n")
        
        f.write(f"blocks : {str(blocks_arg_full)} \n")
        
        f.write(f"<<< filters info >>> \n")
        f.write(f"{str(filter_info_a)} \n")
        f.write(f"{str(filter_info_b)} \n")
        
        f.write(f"resolution : {int(1000*coefs_combi[2])} \n")
        
        f.write(f"pred time : {pred_time} \n")
        
        f.write(f"test acc : {test_acc}% \n")        
        
        f.write(f"Macro-average : precision-{sum(cm_all.loc['precision'])/7}, recall/sensitivity-{sum(cm_all.loc['recall'])/7}, F1 Score-{sum(cm_all.loc['F1 Score'])/7}, specificity-{sum(cm_all.loc['specificity'])/7},")

        f.write(f"efficiency score = Accuracy / FLOPs : {str(test_acc/flops * 100000)} \n")

        f.write(f"epoch : {str(epoch)} \n")





sys.exit()

    
    
# %%
