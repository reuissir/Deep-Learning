import numpy as np
import sys
import os

module_path = os.path.abspath('C:/Users/heess/Deep Learning/DeepLearningFS/DLFS/ch03')
sys.path.append(module_path)

from mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize = True, one_hot_label=True)

print(x_train.shape)
# (60000, 784) --> 784 = 28 * 28
print(t_train.shape)
# (60000, 10) --> 정답 레이블은 10줄짜리 데이터

# 훈련데이터에서 무작위로 10장만 빼내기 --> minibatch 만들기
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

np.random.choice(60000, 10)

# mini-batch용 교차 엔트로피 오차 구현하기

# 인자 y는 신경망의 출력, t는 정답 레이블이다
# y 는 신경망의 예측값들을 가진 numpy array다. 
# y 는 batch_size와 num_classes(출력 값)로 구성 되어있다
# t는 라벨값들을 가진 numpy array다

# one_hot_encoding cross_entropy_error
# t가 0 인 원소는 교체 엔트로피 오차도 0이므로, 그 계산은 무시해도 된다
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    batch_size = y.shape[0]
    # batch size로 나눠주며 정규화히여 이미지 1장당 평균의 교차 엔트로피 오차를 구하기
    return -np.sum(t * np.log(y)) / batch_size
        



# 정답 레이블이 원-핫 인코딩이 아니라 '2'나 '7' 등의 숫자 레이블로 주어졌을 때의 교차 엔트로피
def cross_entropy_error_notonehot(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size

"""
t 와 y를 전부 1행(차원) 리스트로 반환하는 이유는 true distribution y와 predicted value t간의 연산을 수행하기 위해서다.
batch size가 10, 클래스가 총 3개 있다고 가정해보자. 그렇다면 y의 shape은 10, 3이 될 것이다. 
이것을 1차원 리스트로 flatten 한다면 1, 30이 될 것이다. 
"""


 

