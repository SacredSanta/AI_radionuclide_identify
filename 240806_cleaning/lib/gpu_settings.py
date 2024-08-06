'''
최종 수정 : 2024.07.28.
사용자 : 서동휘

<수정 내용> 

<처음>
GPU setting 등을 위한 연습 코드
'''


#%%
import tensorflow as tf
# %%
print(tf.__version__)



#%% GPU 상태 Debug
tf.debugging.set_log_device_placement(True)

# 텐서 생성
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)

print(c)





# %% 멀티 GPU 사용

# GPU 추상화
gpus = tf.config.list_physical_devices('GPU')

# GPU status 확인
print(gpus) 

# GPU 가상화
if gpus:
    try:
        # Create 2 virtual GPUs with 1GB memory each
        tf.config.set_logical_device_configuration(
            gpus[0],  # 분할할 gpu 지정
            [tf.config.LogicalDeviceConfiguration(memory_limit=1024),    # logical 분리 1
             tf.config.LogicalDeviceConfiguration(memory_limit=1024)]    # logical 분리 2
        )
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPU, ", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print("!!!!@@@@@!!!!@@@@", e, "!!!!@@@@@!!!!@@@@")
        
        
        
        
        
        
        
        
# %% 멀티 GPU 사용

# Debug on
tf.debugging.set_log_device_placement(True)

# logical 오브젝트화
gpus = tf.config.list_logical_devices('GPU')

# strategy - 데이터 병렬처리 - MODEL 복사하여 각 GPU에서 실행
strategy = tf.distribute.MirroredStrategy(gpus)

with strategy.scope():
    print("make model")
    inputs = tf.keras.layers.Input(shape=(1,))
    predictions = tf.keras.layers.Dense(1)(inputs)
    model = tf.keras.models.Model(inputs=inputs, outputs=predictions)
    model.compile(loss='mse',
                  optimizer=tf.keras.optimizers.SGD(learning_rate=0.2))
    

# %%
