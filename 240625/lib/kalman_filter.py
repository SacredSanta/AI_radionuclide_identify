#%%
import numpy as np

# 시간 간격
dt = 1.0

# 상태 전이 행렬
F = np.array([[1, dt],
              [0, 1]])

# 제어 입력 행렬 (없을 경우 B는 생략 가능)
B = np.array([[0.5 * dt**2],
              [dt]])

# 측정 모델 행렬
H = np.array([[1, 0]])

# 프로세스 노이즈 공분산
Q = np.array([[1, 0],
              [0, 1]])

# 측정 노이즈 공분산
R = np.array([[5]])

# 초기 상태 추정
x = np.array([[0],
              [1]])

# 초기 오차 공분산
P = np.array([[1, 0],
              [0, 1]])

# 측정값 (예시)
measurements = [1, 2, 3, 4, 5, 6]

# 칼만 필터 적용
for z in measurements:
    # 예측 단계
    x = F @ x
    P = F @ P @ F.T + Q
    
    # 칼만 이득 계산
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
    
    # 상태 추정치 갱신
    x = x + K @ (z - H @ x)
    
    # 오차 공분산 갱신
    P = (np.eye(2) - K @ H) @ P
    
    print(f"Updated state: {x.flatten()}")
# %%
