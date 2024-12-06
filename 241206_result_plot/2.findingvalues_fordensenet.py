#%% ====================================
# coeff 도출 코드
# =======================================
#%%
import numpy as np

# 0.05 단위로 1 이하의 값을 가지는 a, b, c 배열 생성
values = np.arange(0.1, 1.1, 0.1)

# 모든 조합을 저장할 리스트
results = []

# 조건을 만족하는 조합 찾기
for a in values:
    for b in values:
        for c in values:
            product = a * b * c
            # product 값이 0.5에 가까운 경우를 찾음
            if abs(product - 0.5) <= 0.1:  # 오차 범위 ±0.05
                results.append([round(a,1), round(b,1), round(c,1)])

# 결과 출력
print("조건을 만족하는 (a, b, c) 조합:")
for result in results:
    print(result)
print(len(results))












#%% ====================================
# grid search 최적 조합 정리
# =======================================

#%% coefs 조합중 최고 acc 찾기
import os

# 현재 디렉토리의 폴더 리스트 출력
dirname = '241113_densenet1d_combi/241113_count5000down_3series_merged_12000_all_orispec_normed/block2,4,8,4/2.2,2.9,0.2'
folders = [f.name for f in os.scandir(f'./2.model/{dirname}') if f.is_dir()]
folders

#%%
import pandas as pd

finaldirec = f"./2.model/{dirname}"

df = pd.DataFrame(columns=['coefs', 'acc', 'FLOPs', 'params'])

for folder in folders:
    if 'phi' in folder:  # phi 제외
        continue
    finalfile = os.path.join(finaldirec, folder)
    for filename in os.listdir(finalfile):
        if filename.endswith('.txt'):
            txt_path = os.path.join(finalfile, filename)
            with open(txt_path,'r') as fff:
                lines = fff.readlines()
            params = list(lines[3].split())[2]    
            flops = list(lines[5].split())[2]
            acc = list(lines[10].split())[3][:-1]
            coef = folder
            
            new_row = {'coefs':folder,
                       'acc':float(acc),
                       'FLOPs':int(flops),
                       'params':int(params)}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
#%%
df["Score"] = df['acc']/(35906146/df['FLOPs'])
df
#%%
df = df.sort_values(by='FLOPs', ascending=True)
df
#%%
df_y = df['acc'].to_numpy()
df_x = df['FLOPs'].to_numpy()
#%%
df_xtick = [round(i/1000000) for i in df_x]
df_x_label = [str(round(i/1000000)) for i in df_x]
df_xtick

#%%
df_xtick_new = [i*20 for i in range(0,18)]
df_xtick_new

#%%
import matplotlib.pyplot as plt

plt.figure(figsize=(12,3))
plt.plot(df_xtick, df_y, marker='^', markersize=10, linewidth=3, color='orange')
plt.axhline(y=85, color='red', linestyle='--', label="y=78", alpha=0.8)
plt.axhline(y=78, color='black', linestyle='-.', label="y=78", alpha=0.8)
#plt.axvline(x=1,  color='blue', linestyle='--', label="Default", alpha=0.8)

plt.title("Compound Scaling")
plt.xlabel("FLOPs (x1_000_000)")
plt.xticks(df_xtick_new,df_xtick_new)
plt.ylabel("Accuracy")
plt.yticks([77,82,87])
plt.tight_layout()
plt.gcf().autofmt_xdate()

#%%
df.to_csv(f"{finaldirec}/results.csv")

#%% watch results from 이미 한거
df = pd.read_csv(f"{finaldirec}/results.csv", index_col=0)
df












#%% ====================================
# 모델 확인하기
# =======================================

#%% load model test
import tensorflow as tf

model_path = "./2.model/241113_effinet_combi/241113_count5000down_3series_merged_12000_all_noisefiltered/blocks_arg3/1,1,1/241113_count5000down_3series_merged_12000_all_noisefiltered_241110_effinet1d_w1_d1_r1.keras"
A = tf.keras.models.load_model(model_path)

A.summary()













#%% ====================================
# 결과 정리용
# =======================================
import os

# 현재 디렉토리의 폴더 리스트 출력
dirname = '241113_densenet1d_combi/241113_count5000down_3series_merged_12000_all_orispec_normed/block2,4,8,4/depth'
folders = [f.name for f in os.scandir(f'./2.model/{dirname}') if f.is_dir()]
folders

#%%
import pandas as pd

finaldirec = f"./2.model/{dirname}"

df = pd.DataFrame(columns=['coefs', 'acc', 'FLOPs', 'params'])

for folder in folders:
    if 'phi' in folder:  # phi 제외
        continue
    finalfile = os.path.join(finaldirec, folder)
    for filename in os.listdir(finalfile):
        if filename.endswith('.txt'):
            txt_path = os.path.join(finalfile, filename)
            with open(txt_path,'r') as fff:
                lines = fff.readlines()
            params = list(lines[3].split())[2]    
            flops = list(lines[5].split())[2]
            acc = list(lines[10].split())[3][:-1]
            coef = folder
            
            new_row = {'coefs':folder,
                       'acc':float(acc),
                       'FLOPs':int(flops),
                       'params':int(params)}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            
            

#%%
df = df.sort_values(by='coefs', ascending=True)
df

#%%
dt_y = df["acc"].to_numpy()
dt_y
#%%
dt_x = [round(i*0.1,1) for i in range(6, 31)]
dt_x
#%%
import matplotlib.pyplot as plt

plt.figure(figsize=(12,3))
plt.plot(dt_x, dt_y[:25], marker='^', markersize=10, linewidth=3)
plt.axhline(y=78, color='red', linestyle='--', label="y=78", alpha=0.8)
plt.axvline(x=1,  color='blue', linestyle='--', label="Default", alpha=0.8)

plt.title("Width")
plt.xlabel("Scale coefficient")
plt.xticks(dt_x,dt_x)
plt.ylabel("Accuracy")
plt.yticks([60,70,80,90])

# %%
