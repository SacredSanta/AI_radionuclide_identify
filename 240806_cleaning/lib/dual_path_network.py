'''
최종 수정 : 2024.07.28.
사용자 : 서동휘

<수정 내용> 

<처음>
DPN model 구현을 위한 코드.
'''


#%% init
import tensorflow as tf

# mirrored Strategy
mirrored_strategy = tf.distribute.MirroredStrategy()
# <장치간 통신 option>
# 장치 간 통신을 재정의하려면 tf.distribute.CrossDeviceOps의 인스턴스를 제공하여 cross_device_ops 인수를 사용하면 됩니다. 
# 현재는 기본값인 tf.distribute.NcclAllReduce 이외에 의 두 가지 옵션을 사용합니다.
# - tf.distribute.HierarchicalCopyAllReduce
# - tf.distribute.ReductionToOneDevice


# 이 아래에서 학습하면됨.
with mirrored_strategy.scope():
    pass


# %% my try
def conv2_inner(cur_model):
    x = cur_model
    print("-- init : ", x.shape)
    for _ in range(3):
        x = tf.keras.tf.keras.layers.Conv2D(96, kernel_size=(1,1), strides=2)(x)
        print("---- 1 : ", x.shape)
        x = tf.keras.tf.keras.layers.Conv2D(96, kernel_size=(3,3), strides=2, groups = 32)(x)
        print("---- 2 : ", x.shape)
        x = tf.keras.tf.keras.layers.Conv2D(256, kernel_size=(1,1), strides=2)(x)
        print("---- 3 : ", x.shape)
        

def conv3_inner(cur_model):
    x = cur_model
    for _ in range(4):
        x = tf.keras.tf.keras.layers.Conv2D(96, kernel_size=(3,3), strides=2, activation=softmax)(x)
        x = tf.keras.tf.keras.layers.Conv2D(96, kernel_size=(3,3), strides=2, activation=softmax)(x)
        x = tf.keras.tf.keras.layers.Conv2D(256, kernel_size=(3,3), strides=2, activation=softmax)(x)

def conv4_inner(cur_model):
    x = cur_model
    for _ in range(20):
        x = tf.keras.tf.keras.layers.Conv2D(96, kernel_size=(3,3), strides=2, activation=softmax)(x)
        x = tf.keras.tf.keras.layers.Conv2D(96, kernel_size=(3,3), strides=2, activation=softmax)(x)
        x = tf.keras.tf.keras.layers.Conv2D(256, kernel_size=(3,3), strides=2, activation=softmax)(x)

def conv5_inner(cur_model):
    x = cur_model
    for _ in range(3):
        x = tf.keras.tf.keras.layers.Conv2D(96, kernel_size=(3,3), strides=2, activation=softmax)(x)
        x = tf.keras.tf.keras.layers.Conv2D(96, kernel_size=(3,3), strides=2, activation=softmax)(x)
        x = tf.keras.tf.keras.layers.Conv2D(256, kernel_size=(3,3), strides=2, activation=softmax)(x)




inputs = tf.keras.Input(shape=(32, 32, 3))
print("inputs : ", inputs.shape)
conv1 = tf.keras.tf.keras.layers.Conv2D(64, kernel_size=(7,7), strides=2, padding='same')(inputs)
print("conv1 : ", conv1.shape)
conv2 = conv2_inner(conv1)
#conv3 = conv3_inner(conv2)
#conv4 = conv4_inner(conv3)
#conv5 = conv5_inner(conv4)

outputs = tf.keras.tf.keras.layers.GlobalAveragePooling1D()(conv2)
outputs = tf.keras.tf.keras.layers.Softmax()(outputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs, name="DPN")
model.summary()





# %% chat gpt version
import tensorflow as tf


class myDPN():
    def __init__(
        self,
        name=None,
        input_shape=(224, 224, 3)
    ):
        self.name = name
        self.input_shape = input_shape
        self.model = None
    
    def conv_bn_relu(x, filters, kernel_size, strides=1, padding='same', use_relu=True):
        x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        if use_relu:
            x = tf.keras.layers.Activation('relu')(x)
        return x

    def dual_path_block(x, filters, strides=1):
        shortcut = x
        in_channels = x.shape[-1]
        k, k_r, k_plus = filters

        # Dense Path
        x1 = conv_bn_relu(x, k_r, 1)
        x1 = conv_bn_relu(x1, k_r, 3, strides=strides)
        x1 = conv_bn_relu(x1, k_plus, 1, use_relu=False)

        # Residual Path
        x2 = conv_bn_relu(x, k, 1)
        x2 = conv_bn_relu(x2, k, 3, strides=strides)
        x2 = conv_bn_relu(x2, k_plus, 1, use_relu=False)

        if strides != 1 or in_channels != k_plus:
            shortcut = conv_bn_relu(shortcut, k_plus, 1, strides=strides, use_relu=False)

        x = tf.keras.layers.Concatenate()([x1, x2, shortcut])
        x = tf.keras.layers.Activation('relu')(x)
        return x

    def dpn_block(x, num_blocks, filters, strides=1):
        for i in range(num_blocks):
            x = dual_path_block(x, filters, strides if i == 0 else 1)
        return x

    def create_dpn_92(self, num_classes=1000):
        input = tf.keras.layers.Input(shape=self.input_shape)
        x = conv_bn_relu(input, 64, 7, strides=2, padding='same')
        x = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

        x = dpn_block(x, 3, (128, 32, 256), strides=1)
        x = dpn_block(x, 4, (256, 32, 512), strides=2)
        x = dpn_block(x, 20, (512, 64, 1024), strides=2)
        x = dpn_block(x, 3, (1024, 128, 2048), strides=2)

        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

        model = tf.keras.Model(inputs=input, outputs=output)
        
        self.model = model
    

DPNmodel = myDPN(input_shape=(224,224,3))
DPNmodel.create_dpn_92()
DPNmodel.model.summary()




# %%
