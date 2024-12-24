#%%
import numpy as np
import matplotlib.pyplot as plt

# %%
source = np.array(['ba', 'cs', 'na', 'background', 'co57', 'th', 'ra', 'am'])

dt = np.load("./1.preprocess_data/")















# %% =================================
# 논문용 지표 확인
# ====================================
import tensorflow as tf
modelname = "241218_set10000_min01040_max01999_all_241202_densenet1d_w1.0_d1.0_r1.0"
Model = tf.keras.models.load_model(f"./2.model/241218data_densenet1d_combi/241218_set10000_min01040_max01999_all/1.0,1.0,1.0/{modelname}.keras")
# %%