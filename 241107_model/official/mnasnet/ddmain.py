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
sys.path.append("/tf/latest_version/new_AI/docker_data_241016/Tensorflow/Code/tpu/models/official/efficientnet/")
#%%
from mnasnet_model import *
from mnasnet_models import *
import mnas_utils

model_parameters = mnasnet_b1()
Model = MnasNetModel(*model_parameters)

