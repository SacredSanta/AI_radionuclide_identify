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


#%% ===================================================
# model check
import tensorflow as tf

model_direc = "./2.model/241113_effinet_combi/241113_count5000down_3series_merged_12000_all_orispec_normed/blocks_arg3/depth/1,1.6,1/241113_count5000down_3series_merged_12000_all_orispec_normed_241110_effinet1d_w1_d1.6_r1.keras"
Model = tf.keras.models.load_model(model_direc)
Model.summary()









#%% coefs 조합중 최고 acc 찾기
import os

# 현재 디렉토리의 폴더 리스트 출력
dirname = '241107_densenet1d_dd_combi/241004_10to20sec_3series_merged_orispectrum_normed_noisefiltered/'
folders = [f.name for f in os.scandir(f'./2.model/{dirname}/findingcoef6,12,24,16') if f.is_dir()]
print(folders)

#%%
import pandas as pd

finaldirec = f"./2.model/{dirname}/findingcoef6,12,24,16"

df = pd.DataFrame(columns=['coefs', 'acc', 'FLOPs', 'params'])

for folder in folders:
    finalfile = os.path.join(finaldirec, folder)
    for filename in os.listdir(finalfile):
        if filename.endswith('.txt'):
            txt_path = os.path.join(finalfile, filename)
            with open(txt_path,'r') as fff:
                lines = fff.readlines()
            params = list(lines[2].split())[2]    
            flops = list(lines[4].split())[2]
            acc = list(lines[9].split())[3][:-1]
            coef = folder
            
            new_row = {'coefs':folder,
                       'acc':float(acc),
                       'FLOPs':int(flops),
                       'params':int(params)}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
#%%
df["Score"] = df['acc']/(df['FLOPs']/10000000)
df
#%%
df = df.sort_values(by='acc', ascending=False)
df

#%%
df.to_csv(f"{finaldirec}/results.csv")














#%%

