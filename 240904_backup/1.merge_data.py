#%% Load data
import numpy as np

path1 = "./1.preprocess_data/240806_30to300sec_7source_5000_all.npz"
data1 = np.load(path1)

path2 = "./1.preprocess_data/240806_4source_3000_xzfile_all.npz"
data2 = np.load(path2)



#%% merge and shuffle data

# 3000 | 1000 | 1000
#             | 3000

# => 3000 ++ 2000 | 1000 ++ 500 | 1000 ++ 500
x_train = np.concatenate((data1["x_train"], data2["x_test"][0:2000]), axis=0)
x_val = np.concatenate((data1["x_val"], data2["x_test"][2000:2500]), axis=0)
x_test = np.concatenate((data1["x_test"], data2["x_test"][2500:3000]), axis=0)

y_train = np.concatenate((data1["y_train"], data2["y_test"][0:2000]), axis=0)
y_val = np.concatenate((data1["y_val"], data2["y_test"][2000:2500]), axis=0)
y_test = np.concatenate((data1["y_test"], data2["y_test"][2500:3000]), axis=0)


#* x ~ y 간 관계 틀어짐. 쓰면 안됨
#np.random.shuffle(total_x)
#np.random.shuffle(total_y)


#%% 
filename = "240827_240806-merged_8000"

np.savez(f"./1.preprocess_data/{filename}_all.npz", 
         x_train=x_train,
         y_train=y_train,
         x_val=x_val,
         y_val=y_val,
         x_test=x_test,
         y_test=y_test)
