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


#%%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
sys.path.append("./lib")

from lib.efficientnet_init import correct_pad
from lib.efficientnet_init import get_submodules_from_kwargs
from lib.efficientnet_init import imagenet_utils
#from imagenet_utils import decode_predictions
#from imagenet_utils import _obtain_input_shape


backend = None
layers = None
models = None
keras_utils = None


BASE_WEIGHTS_PATH = (
    'https://github.com/Callidior/keras-applications/'
    'releases/download/efficientnet/')
WEIGHTS_HASHES = {
    'b0': ('e9e877068bd0af75e0a36691e03c072c',
           '345255ed8048c2f22c793070a9c1a130'),
    'b1': ('8f83b9aecab222a9a2480219843049a1',
           'b20160ab7b79b7a92897fcb33d52cc61'),
    'b2': ('b6185fdcd190285d516936c09dceeaa4',
           'c6e46333e8cddfa702f4d8b8b6340d70'),
    'b3': ('b2db0f8aac7c553657abb2cb46dcbfbb',
           'e0cf8654fad9d3625190e30d70d0c17d'),
    'b4': ('ab314d28135fe552e2f9312b31da6926',
           'b46702e4754d2022d62897e0618edc7b'),
    'b5': ('8d60b903aff50b09c6acf8eaba098e09',
           '0a839ac36e46552a881f2975aaab442f'),
    'b6': ('a967457886eac4f5ab44139bdd827920',
           '375a35c17ef70d46f9c664b03b4437f2'),
    'b7': ('e964fd6e26e9a4c144bcb811f2a10f20',
           'd55674cc46b805f4382d18bc08ed43c1')
}


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


# activation function
def swish(x):
    """Swish activation function.

    # Arguments
        x: Input tensor.

    # Returns
        The Swish activation: `x * sigmoid(x)`.

    # References
        [Searching for Activation Functions](https://arxiv.org/abs/1710.05941)
    """
    if backend.backend() == 'tensorflow':
        try:
            # The native TF implementation has a more
            # memory-efficient gradient implementation
            return backend.tf.nn.swish(x)
        except AttributeError:
            pass

    return x * backend.sigmoid(x)


# mobile inverted residual block   -----------------------------------
def block(inputs, activation_fn=swish, drop_rate=0., name='',
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
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    # (1) Expansion phase - 보통 1로서 그대로 들어감
    filters = filters_in * expand_ratio
    if expand_ratio != 1:
        x = layers.Conv2D(filters, 1,
                          padding='same',
                          use_bias=False,
                          kernel_initializer=CONV_KERNEL_INITIALIZER,
                          name=name + 'expand_conv')(inputs)
        x = layers.BatchNormalization(axis=bn_axis, name=name + 'expand_bn')(x)
        x = layers.Activation(activation_fn, name=name + 'expand_activation')(x)
    else:
        x = inputs

    # (2) Depthwise Convolution - channel 하나 자체로만 연산을 하도록
    if strides == 2: # stride 가 다르면 padding 해놓도록
        x = layers.ZeroPadding2D(padding=correct_pad(backend, x, kernel_size), # zeropadding한 matrix 반환
                                 name=name + 'dwconv_pad')(x)
        conv_pad = 'valid'
    else:
        conv_pad = 'same'
    x = layers.DepthwiseConv2D(kernel_size,
                               strides=strides,
                               padding=conv_pad,
                               use_bias=False,
                               depthwise_initializer=CONV_KERNEL_INITIALIZER,
                               name=name + 'dwconv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=name + 'bn')(x)
    x = layers.Activation(activation_fn, name=name + 'activation')(x)

    # (3) Squeeze and Excitation phase
    if 0 < se_ratio <= 1:
        filters_se = max(1, int(filters_in * se_ratio))
        se = layers.GlobalAveragePooling2D(name=name + 'se_squeeze')(x)
        if bn_axis == 1:
            se = layers.Reshape((filters, 1, 1), name=name + 'se_reshape')(se)
        else:
            se = layers.Reshape((1, 1, filters), name=name + 'se_reshape')(se)
        se = layers.Conv2D(filters_se, 1,
                           padding='same',
                           activation=activation_fn,
                           kernel_initializer=CONV_KERNEL_INITIALIZER,
                           name=name + 'se_reduce')(se)
        se = layers.Conv2D(filters, 1,
                           padding='same',
                           activation='sigmoid',
                           kernel_initializer=CONV_KERNEL_INITIALIZER,
                           name=name + 'se_expand')(se)
        if backend.backend() == 'theano':
            # For the Theano backend, we have to explicitly make
            # the excitation weights broadcastable.
            se = layers.Lambda(
                lambda x: backend.pattern_broadcast(x, [True, True, True, False]),
                output_shape=lambda input_shape: input_shape,
                name=name + 'se_broadcast')(se)
        x = layers.multiply([x, se], name=name + 'se_excite')

    # (4) Output phase
    x = layers.Conv2D(filters_out, 1,
                      padding='same',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name=name + 'project_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=name + 'project_bn')(x)
    if (id_skip is True and strides == 1 and filters_in == filters_out):
        if drop_rate > 0:
            x = layers.Dropout(drop_rate,
                               noise_shape=(None, 1, 1, 1),
                               name=name + 'drop')(x)
        x = layers.add([x, inputs], name=name + 'add')

    return x



debug = 1
# ----------------------------------------------------------------------
def EfficientNet(width_coefficient,
                 depth_coefficient,
                 default_size,
                 dropout_rate=0.2,
                 drop_connect_rate=0.2,
                 depth_divisor=8,
                 activation_fn=swish,
                 blocks_args=DEFAULT_BLOCKS_ARGS,
                 model_name='efficientnet',
                 include_top=True,
                 weights='imagenet',
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
    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    # weights 관한 정보를 imagenet으로부터 받아오는듯
    # weight 오류 처리
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if debug:
        print("weights : ", weights)
    sys.exit()


    # weight가 지정되었을 때.
    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    # Input 형태 취득
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=default_size,
                                      min_size=32,
                                      data_format=backend.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    # input_tensor 초기값이 들어오지 않았을 때 / kears_tensor 가 아닐 때 처리
    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else: 
            img_input = input_tensor

    # #?
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    # 
    def round_filters(filters, divisor=depth_divisor): # depth_divisor : default=8
        """Round number of filters based on depth multiplier."""
        filters *= width_coefficient  # 변환시키려는 filter 수 에 ,  width_계수 곱해주기
        new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)  # 새로운 값은, depth_divisor 통해서 나누기
        # Make sure that round down does not go down by more than 10%.
        if new_filters < 0.9 * filters:  # 기존 filter 수 보다 90%미만이면
            new_filters += divisor  # 새 filter 수에 divisor 를 합쳐준다
        return int(new_filters)

    # 
    def round_repeats(repeats):
        """Round number of repeats based on depth multiplier."""
        return int(math.ceil(depth_coefficient * repeats))

    # <<<<  1. Build stem  >>>>  ----------------------------
    x = img_input
    x = layers.ZeroPadding2D(padding=correct_pad(backend, x, 3),
                             name='stem_conv_pad')(x)
    x = layers.Conv2D(round_filters(32), 3,
                      strides=2,
                      padding='valid',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name='stem_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='stem_bn')(x)
    x = layers.Activation(activation_fn, name='stem_activation')(x)

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
            x = block(x, activation_fn, drop_connect_rate * b / blocks,
                      name='block{}{}_'.format(i + 1, chr(j + 97)), **args)
            b += 1

    # Build top  ----------------------------
    x = layers.Conv2D(round_filters(1280), 1,
                      padding='same',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name='top_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='top_bn')(x)
    x = layers.Activation(activation_fn, name='top_activation')(x)
    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name='top_dropout')(x)
        x = layers.Dense(classes,
                         activation='softmax',
                         kernel_initializer=DENSE_KERNEL_INITIALIZER,
                         name='probs')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D(name='max_pool')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = models.Model(inputs, x, name=model_name)

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            file_suff = '_weights_tf_dim_ordering_tf_kernels_autoaugment.h5'
            file_hash = WEIGHTS_HASHES[model_name[-2:]][0]
        else:
            file_suff = '_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'
            file_hash = WEIGHTS_HASHES[model_name[-2:]][1]
        file_name = model_name + file_suff
        weights_path = keras_utils.get_file(file_name,
                                            BASE_WEIGHTS_PATH + file_name,
                                            cache_subdir='models',
                                            file_hash=file_hash)
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

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



Model = EfficientNet()
sys.exit()

setattr(EfficientNetB0, '__doc__', EfficientNet.__doc__)
setattr(EfficientNetB1, '__doc__', EfficientNet.__doc__)
setattr(EfficientNetB2, '__doc__', EfficientNet.__doc__)
setattr(EfficientNetB3, '__doc__', EfficientNet.__doc__)
setattr(EfficientNetB4, '__doc__', EfficientNet.__doc__)
setattr(EfficientNetB5, '__doc__', EfficientNet.__doc__)
setattr(EfficientNetB6, '__doc__', EfficientNet.__doc__)
setattr(EfficientNetB7, '__doc__', EfficientNet.__doc__)





















#%%

from keras.src.applications.efficientnet import EfficientNet, EfficientNetB0

Model = EfficientNetB0()
Model.summary()

# def EfficientNetB0(include_top=True,
#                    weights='imagenet',
#                    input_tensor=None,
#                    input_shape=None,
#                    pooling=None,
#                    classes=1000,
#                    **kwargs):
#     return EfficientNet(1.0, 1.0, 224, 0.2,
#                         model_name='efficientnet-b0',
#                         include_top=include_top, weights=weights,
#                         input_tensor=input_tensor, input_shape=input_shape,
#                         pooling=pooling, classes=classes,
#                         **kwargs


#%%
Model = EfficientNet(width_coefficient,
                    depth_coefficient,
                    default_size,
                    dropout_rate=0.2,
                    drop_connect_rate=0.2,
                    depth_divisor=8,
                    activation_fn=swish,
                    blocks_args=DEFAULT_BLOCKS_ARGS,
                    model_name='efficientnet',
                    include_top=True,
                    weights='imagenet',
                    input_tensor=None,
                    input_shape=None,
                    pooling=None,
                    classes=1000,
                    **kwargs)


#%% resnet
from keras.src.applications.resnet import ResNet

Model2 = ResNet()









#%% 직접 구현 ===========================================================================================================

#%%
#%%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1' # 특정 GPU 사용 활성화
print(os.environ)

# Library
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys

#Check GPU
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
tf.config.set_visible_devices(physical_devices[0],'GPU')  # TITAN 으로 지정

#%%

def MBConv(x, kernel, channels, strides, iteration, se_ratio=0.25):
    print("MBConv in ===================")
    print("MB Conv input : ", x.shape)
    for i in range(iteration):
        if i == 0:
            expand_ratio = 1
            oup = 1
        else:
            expand_ratio = 6
            oup = 32
        print("iteration starts with x : ", x.shape)
        # Expand Phase
        print(channels,expand_ratio)
        x1 = tf.keras.layers.Conv2D(filters=channels*expand_ratio, kernel_size=kernel, strides=strides,
                                    kernel_initializer='he_normal', activation='relu', padding='same')(x)
        x1 = tf.keras.layers.BatchNormalization()(x1)
        x1 = tf.keras.layers.ReLU()(x1)
        print("x1 shape : ", x1.shape)
        
        # Depthwise Conv Phase
        x2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1,3), strides=(1,1), groups=oup,
                                    kernel_initializer='he_normal', activation='relu', padding='same')(x1)
        x2 = tf.keras.layers.BatchNormalization()(x2)
        x2 = tf.keras.layers.ReLU()(x2)
        print("x2 shape : ", x2.shape)
        
        '''
        # Squeeze & excitation phase (SE)
        x3 = tf.keras.layers.AveragePooling2D(pool_size=1, strides=1, padding='valid')(x2)
        x3 = tf.keras.layers.Conv2D(filters=int(max(1,channels*se_ratio)), kernel_size=(1,1), strides=(1,1),
                                    kernel_initializer='he_normal', activation='relu', padding='same')(x3)
        x3 = tf.keras.activations.silu(x3)
        x3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1,1), strides=(1,1),
                                    kernel_initializer='he_normal', activation='relu', padding='same')(x3)
        x3 = tf.keras.activations.sigmoid(x3)
        print("x3 shape : ", x3.shape)
        '''
        
        # Output phase
        x4 = tf.keras.layers.Conv2D(filters=16, kernel_size=(1,1), strides=(1,1),
                                    kernel_initializer='he_normal', activation='relu', padding='same')(x3)
        x4 = tf.keras.layers.BatchNormalization()(x4)
        x4 = tf.keras.layers.Dropout(0.2)(x4)
        print("x4 shape : ", x4.shape)
        
        
        # Skip connection
        x = tf.keras.layers.Concatenate()([x1, x4])
        print("iter fin, x shape : ", x.shape)
    
    return x


def efficient_dd(inputs):
    
    # Stem
    print("input : ", inputs.shape)
    i1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1,3), strides=(1,2),
                                   kernel_initializer='he_normal', activation='relu', padding='same')(inputs)
    i1 = tf.keras.layers.BatchNormalization()(i1)
    i1 = tf.keras.activations.silu(i1)
    print("i1 shape : ", i1.shape)
    
    # MBConv BLocks
    i2 = MBConv(i1, kernel=(1,3), channels=16, strides=(1,1), iteration=1)   
    i3 = MBConv(i2, kernel=(1,3), channels=24, strides=(1,2), iteration=2)
    sys.exit()
    i4 = MBConv(i3, kernel=(1,5), channels=40, strides=(1,2), iteration=2)
    i5 = MBConv(i4, kernel=(1,3), channels=80, strides=(1,2), iteration=3)
    i6 = MBConv(i5, kernel=(1,5), channels=112, strides=(1,1), iteration=3)
    i7 = MBConv(i6, kernel=(1,5), channels=192, strides=(1,2), iteration=4)
    i8 = MBConv(i7, kernel=(1,3), channels=320, strides=(1,1), iteration=1)
    
    # Head
    i9 = tf.keras.layers.Conv2D(filters=1280, kernel_size=(1,1), strides=(1,1),
                                kernel_initializer='he_normal', activation='relu', padding='same')(i8)
    i9 = tf.keras.layers.BatchNormalization()(i9)
    i9 = tf.keras.layers.silu()(i9)
    
    # Pooling & FC
    i9 = tf.keras.layers.AveragePooling2D()(i9)
    i9 = tf.keras.layers.Dropout(0.2)(i9)
    print("final : ", i9.shape)
    #i9 = tf.keras.layers.Linear
    
    return i9

inputs = tf.keras.layers.Input(shape=(1,1000,3))
outputs = efficient_dd(inputs)
Model = tf.keras.Model(inputs, outputs)










#%% 블로그 버전..

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
import math


#%%

# 모델 관련 하이퍼 파라미터
se_ratio = 4
expand_ratio = 6
width_coefficient = 1.0
depth_coefficient = 1.0
default_resolution = 224
input_channels = 3
depth_divisor= 8 
dropout_rate = 0.2
drop_connect_rate = 0.2

# layer 관련 매개변수
kernel_size = [3,3,5,3,5,5,3]
num_repeat = [1,2,2,3,3,4,1]
output_filters = [16,24,40,80,112,192,320]
strides = [1,2,2,2,1,2,1]
MBConvBlock_1_True  =  [True,False,False,False,False,False,False]

# depth 와 filter 수를 hyper parameter 에 따라 바꿔주는 함수
def round_repeats(repeats, depth_coefficient):
    return int(math.ceil(depth_coefficient * repeats))

def round_filters(filters, width_coefficient, depth_divisor):
    filters *= width_coefficient
    new_filters = int(filters + depth_divisor / 2) // depth_divisor * depth_divisor
    new_filters = max(depth_divisor, new_filters)
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)

# skip 연결판 dropout 
class DropConnect(layers.Layer):
    def __init__(self, drop_connect_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.drop_connect_rate = drop_connect_rate

    def call(self, inputs, training):
        def _drop_connect():
            keep_prob = 1.0 - self.drop_connect_rate

            batch_size = tf.shape(inputs)[0]
            random_tensor = keep_prob
            random_tensor += K.random_uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
            binary_tensor = tf.floor(random_tensor)
            output = tf.math.divide(inputs, keep_prob) * binary_tensor
            return output

        return K.in_train_phase(_drop_connect, inputs, training=training)


# SE Block : 입력에 쓸모없는 부분(?)을 sigmoid를 통해 정규화된 값(0~1)을 곱하여 제거하는 블록
def SEBlock(filters, reduced_filters):
    reduced_filters = int(reduced_filters)
    def _block(inputs):
        x = layers.GlobalAveragePooling2D()(inputs)
        x = layers.Reshape((1,1,x.shape[1]))(x)
        x = layers.Conv2D(reduced_filters, 1, 1)(x)
        x = tf.keras.layers.ReLU()(x)
        x = layers.Conv2D(filters, 1, 1)(x)
        x = layers.Activation('sigmoid')(x)
        x = layers.Multiply()([x, inputs])
        return x
    return _block


# MBConvBlock
def MBConvBlock(x,kernel_size, strides, drop_connect_rate, output_channels, MBConvBlock_1_True=False):
    output_channels = round_filters(output_channels,width_coefficient,depth_divisor)
    if MBConvBlock_1_True:
        block = layers.DepthwiseConv2D(kernel_size, strides,padding='same', use_bias=False)(x)
        block = layers.BatchNormalization()(block)
        block = tf.keras.layers.ReLU()(block)
        block = SEBlock(x.shape[3], x.shape[3]/se_ratio)(block)
        block = layers.Conv2D(output_channels, (1,1), padding='same', use_bias=False)(block)
        block = layers.BatchNormalization()(block)
        return block

    channels = x.shape[3]
    expand_channels = channels * expand_ratio
    block = layers.Conv2D(expand_channels, (1,1), padding='same', use_bias=False)(x)
    block = layers.BatchNormalization()(block)
    block = tf.keras.layers.ReLU()(block)
    block = layers.DepthwiseConv2D(kernel_size, strides,padding='same', use_bias=False)(block)
    block = layers.BatchNormalization()(block)
    block = tf.keras.layers.ReLU()(block)
    block = SEBlock(expand_channels, channels/se_ratio)(block)
    block = layers.Conv2D(output_channels, (1,1), padding='same', use_bias=False)(block)
    block = layers.BatchNormalization()(block)
    if x.shape[3] == output_channels:
        block = DropConnect(drop_connect_rate)(block)
        block = layers.Add()([block, x])
    return block


# EfficientNet
def EffNet(num_classes):
    x_input = layers.Input(shape=(default_resolution, default_resolution, input_channels))    
    x = layers.Conv2D(round_filters(32, width_coefficient, depth_divisor), (3,3), 2,padding='same', use_bias=False)(x_input)
    x = layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    num_blocks_total = sum(num_repeat)
    block_num = 0
    for i in range(len(kernel_size)):
        round_num_repeat = round_repeats(num_repeat[i], depth_coefficient)
        drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
        x = MBConvBlock(x,kernel_size[i],strides[i],drop_rate,output_filters[i],MBConvBlock_1_True = MBConvBlock_1_True[i])
        block_num += 1
        if round_num_repeat > 1:
            for bidx in range(round_num_repeat - 1):
                drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
                x = MBConvBlock(x,kernel_size[i],1,drop_rate,output_filters[i],MBConvBlock_1_True = MBConvBlock_1_True[i])
                block_num += 1
    x = layers.Conv2D(round_filters(1280, width_coefficient, depth_divisor), 1,padding='same',use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(num_classes,activation='softmax')(x)
    model = tf.keras.models.Model(inputs=x_input, outputs=x)
    return model


Model = EffNet(1000)
Model.summary()