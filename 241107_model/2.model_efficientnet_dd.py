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

import os
import math
sys.path.append("./lib")


#%%

DEFAULT_BLOCKS_ARGS = [
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
def EfficientNet(width_coefficient,
                 depth_coefficient,
                 dropout_rate=0.2,
                 drop_connect_rate=0.2,
                 depth_divisor=8,
                 activation_fn=swish,
                 blocks_args=DEFAULT_BLOCKS_ARGS,
                 model_name='efficientnet',
                 include_top=True,
                 input_tensor=None,
                 input_shape=None,
                 pooling=None,
                 classes=1000,
                 **kwargs):
    """Instantiates the EfficientNet architecture using given scaling coefficients.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    # Arguments
        width_coefficient: float, scaling coefficient for network width.
        depth_coefficient: float, scaling coefficient for network depth.
        default_size: integer, default input image size.
        dropout_rate: float, dropout rate before final classifier layer.
        drop_connect_rate: float, dropout rate at skip connections.
        depth_divisor: integer, a unit of network width.
        activation_fn: activation function.
        blocks_args: list of dicts, parameters to construct block modules.
        model_name: string, model name.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False.
            It should have exactly 3 inputs channels.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    #global backend, layers, models, keras_utils
    #backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    # Input 형태 취득
    img_input = layers.Input(shape=(1000,3))
    
    # width_coefficient
    def round_filters(filters, divisor=depth_divisor): # depth_divisor : default=8
        """Round number of filters based on depth multiplier."""
        filters *= width_coefficient  # 변환시키려는 filter 수 에 ,  width_계수 곱해주기
        new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)  # 새로운 값은, depth_divisor 통해서 나누기
        # Make sure that round down does not go down by more than 10%.
        if new_filters < 0.9 * filters:  # 기존 filter 수 보다 90%미만이면
            new_filters += divisor  # 새 filter 수에 divisor 를 합쳐준다
        return int(new_filters)

    # depth_coefficient
    
    def round_repeats(repeats):
        """Round number of repeats based on depth multiplier."""
        return int(math.ceil(depth_coefficient * repeats))

    
    
    # <<<<  1. Build stem  >>>>  ----------------------------
    x = img_input
    x = layers.Conv1D(round_filters(32), 
                      kernel_size=3,
                      strides=2,
                      padding='same',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name='stem_conv')(x)
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
    
    for (i, args) in enumerate(blocks_args):
        assert args['repeats'] > 0    # assert 는 조건문 검토
        # Update block input and output filters based on depth multiplier.
        args['filters_in'] = round_filters(args['filters_in'])
        args['filters_out'] = round_filters(args['filters_out'])

        for j in range(round_repeats(args.pop('repeats'))):
            # The first block needs to take care of stride and filter size increase.
            if j > 0:
                args['strides'] = 1
                args['filters_in'] = args['filters_out']
            x = block(x, drop_connect_rate * b / blocks,
                      name='block{}{}_'.format(i + 1, chr(j + 97)), **args)
            b += 1
        
    # Build top  ----------------------------
    x = layers.Conv1D(round_filters(1280),
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

    return model


def EfficientNetB0(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):
    return EfficientNet(1.0, 1.0, 224, 0.2,
                        model_name='efficientnet-b0',
                        include_top=include_top, weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        **kwargs)


def EfficientNetB1(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):
    return EfficientNet(1.0, 1.1, 240, 0.2,
                        model_name='efficientnet-b1',
                        include_top=include_top, weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        **kwargs)


def EfficientNetB2(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):
    return EfficientNet(1.1, 1.2, 260, 0.3,
                        model_name='efficientnet-b2',
                        include_top=include_top, weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        **kwargs)


def EfficientNetB3(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):
    return EfficientNet(1.2, 1.4, 300, 0.3,
                        model_name='efficientnet-b3',
                        include_top=include_top, weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        **kwargs)


def EfficientNetB4(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):
    return EfficientNet(1.4, 1.8, 380, 0.4,
                        model_name='efficientnet-b4',
                        include_top=include_top, weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        **kwargs)


def EfficientNetB5(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):
    return EfficientNet(1.6, 2.2, 456, 0.4,
                        model_name='efficientnet-b5',
                        include_top=include_top, weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        **kwargs)


def EfficientNetB6(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):
    return EfficientNet(1.8, 2.6, 528, 0.5,
                        model_name='efficientnet-b6',
                        include_top=include_top, weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        **kwargs)


def EfficientNetB7(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):
    return EfficientNet(2.0, 3.1, 600, 0.5,
                        model_name='efficientnet-b7',
                        include_top=include_top, weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        **kwargs)


def preprocess_input(x, data_format=None, **kwargs):
    """Preprocesses a numpy array encoding a batch of images.

    # Arguments
        x: a 3D or 4D numpy array consists of RGB values within [0, 255].
        data_format: data format of the image tensor.

    # Returns
        Preprocessed array.
    """
    return imagenet_utils.preprocess_input(x, data_format,
                                           mode='torch', **kwargs)



#Model = EfficientNet(width_coefficient=1,
#                     depth_coefficient=1,
#                     )
#Model.summary()

setattr(EfficientNetB0, '__doc__', EfficientNet.__doc__)
setattr(EfficientNetB1, '__doc__', EfficientNet.__doc__)
setattr(EfficientNetB2, '__doc__', EfficientNet.__doc__)
setattr(EfficientNetB3, '__doc__', EfficientNet.__doc__)
setattr(EfficientNetB4, '__doc__', EfficientNet.__doc__)
setattr(EfficientNetB5, '__doc__', EfficientNet.__doc__)
setattr(EfficientNetB6, '__doc__', EfficientNet.__doc__)
setattr(EfficientNetB7, '__doc__', EfficientNet.__doc__)







#%% all-in-one ===================================================
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

width_coefs = [0.1, 0.5, 1, 1.5, 2]
depth_coefs = [0.1, 0.5, 1, 1.5, 2]
coefs = [width_coefs, depth_coefs]
coefs_combis = list(product(*coefs)) 
epoch = 50

# 모델 비교 반복
for coefs_combi in coefs_combis:
    try:
        del Model
    except:
        pass
    # 모델 생성
    Model = EfficientNet(width_coefficient=coefs_combi[0], 
                        depth_coefficient=coefs_combi[1],
                        classes=8)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, clipnorm=0.1, weight_decay=1e-4)

    # 모델 컴파일
    Model.compile(optimizer=optimizer,
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=['binary_accuracy'])

    # callback 지정
    date = f"241025_efficient1d_dd_w{coefs_combi[0]}_d{coefs_combi[1]}" 
    csv_logger = tf.keras.callbacks.CSVLogger(f"./2.model/241025_efficient1d_dd_combi/{filename}_{date}.log",
                                            separator=',',
                                            append=False)
    weights = tf.keras.callbacks.ModelCheckpoint(filepath=f"./2.model/241025_efficient1d_dd_combi/{filename}_{date}_ckpt.keras",
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

    Model.save(f"./2.model/241025_efficient1d_dd_combi/{filename}_{date}.keras")
    
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
    
    with open(f"./2.model/241025_efficient1d_dd_combi/{filename}_{date}.txt", 'a') as f:     
        trainable_params = sum([tf.size(variable).numpy() for variable in Model.trainable_variables])
        f.write(f"traiable_params : {trainable_params} \n")
        
        opti_params = sum([tf.size(variable).numpy() for variable in optimizer.variables])
        f.write(f"optimizer_params : {opti_params} \n")
        
        f.write(f"total_params : {trainable_params+opti_params} \n")
        
        f.write(f"train_time : {fit_time_total} \n")
        
        f.write(f"FLOPs : {flops} \n")
        
        
        
    
    
    
    