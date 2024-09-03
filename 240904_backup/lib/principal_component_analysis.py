'''
최종 수정 : 2024.08.06.
사용자 : 서동휘

<수정 내용> 

<처음>
PCA 이해를 위한 연습 코드
'''

#%%
import numpy as np


# %%
#%% ---------------------------------------------------------------
# -                      1.PCA                                   -
# -----------------------------------------------------------------
#%%
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# %%
np.random.seed(0)
X = np.random.rand(10, 5)  # 10개의 샘플과 5개의 특성

# PCA 모델 생성 (주성분의 수를 2로 설정)
pca = PCA(n_components=5)

# PCA 적용
X_pca = pca.fit_transform(X)

# 결과 출력
print("원본 데이터:\n", X)
print("PCA 변환된 데이터:\n", X_pca)

# 주성분 비율
print("설명된 분산의 비율:", pca.explained_variance_ratio_)

# 주성분을 시각화
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA result')
plt.show()



# %% ------------------------------------------------------------

D1 = np.array([170,150,160,180,170]).reshape(5,1)
D2 = np.array([70, 45, 55, 60, 80]).reshape(5,1)
D = np.hstack((D1, D2))
# %%
D[:,0] = D[:,0] - np.mean(D1)
D[:,1] = D[:,1] - np.mean(D2)
# %%
n = len(D1)
SIGMA = (D.transpose() @ D) / n
print(SIGMA)

# 고유값과 고유벡터 계산
eigenvalues, eigenvectors = np.linalg.eig(D)

print("고유값:")
print(eigenvalues)

print("고유벡터:")
print(eigenvectors) # 한개의 column이 하나의 벡터


# %% 
mag = np.sqrt(np.sum(eigenvectors[:,0]**2))
print(mag)
# unit 크기로 주는지 확인
# %%
