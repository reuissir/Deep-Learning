import numpy as np
import matplotlib.pylab as plt

# step function   
def step_function(x):
    return np.array(x > 0, dtype=np.int)

"""
x = np.array([-1.0, 1.0, 2.0])
y = x > 0
y = y.astype(np.int)
print(y)
"""

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

# sigmoid function

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# x = np.array([-1.0, 1.0, 2.0])
# print(sigmoid(x))

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

# RELU(Rectified Linear Unit)

def relu(x):
    return np.maximum(0, x)

y = relu(x)
print(relu(x))


plt.plot(x, y)
plt.ylim(-1.0, 5.5)
plt.show()

# identity function
def identity_function(x):
    return x

y2 = identity_function(x)

