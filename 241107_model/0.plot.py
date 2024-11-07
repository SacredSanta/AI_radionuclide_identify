#%%

import numpy as np
import sys
import os
import matplotlib.pyplot as plt
sys.path.append('./')

from lib.modi_hist_extract import modi_hist_extract
from lib.modwt_pkg import *

#%%

data_folder = "./1.preprocess_data/"
data_name = "240923_10to20_3source_10000_xzfile_all.npz"

data = np.load(os.path.join(data_folder, data_name))

data

#%%
data["x_train"][1]

#%%
data2_folder = "./0.dataset/"
data2_name = "240923_10to20_3source_10000_xzfile.npz"

data2 = np.load(os.path.join(data2_folder, data2_name))

data2
#%%
x = data2["x_ori"][7]
x = noisefiltering(x,0,0)

plt.plot(x[0])