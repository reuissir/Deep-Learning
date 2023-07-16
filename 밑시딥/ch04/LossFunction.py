import numpy as np


# 평균 제곱 오차(mean_squared_error)
def mean_squared_error(y, t):
    # 첫 예시 같은 경우 두개의 데이터에 대한 합차곱을 구하기 때문에 0.5로 나눈다
    return 0.5 * np.sum((y-t)**2)

# one-hot-encoding
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

# softmax 출력 결과(가정)
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]

mean_squared_error(np.array(y), np.array(t))

# 교차 엔트로피 오차(cross entropy loss)
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print(cross_entropy_error(np.array(y), np.array(t)))

y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
cross_entropy_error(np.array(y), np.array(t))



