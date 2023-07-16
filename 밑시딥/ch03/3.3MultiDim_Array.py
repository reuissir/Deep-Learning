import numpy as np

# 1 dimensional matrix
A = np.array([1, 2, 3, 4])
print("A")
print(A)
print("np.ndim(A)")
print(np.ndim(A))
print(A.shape)
print(A.shape[0])
print()

# 2 dimensional matrix
B = np.array([[1, 2], [3, 4], [5, 6]])
print("B")
print(B)
print("np.ndim(B)")
print(np.ndim(B))
print(B.shape)
print()

# Matrix multiplication(matrix dot produt)
A1 = np.array([[1, 2], [3, 4]])
B1 = np.array([[5, 6], [7, 8]])
print("matrix dot product(A, B)")
print(np.dot(A1, B1))

C = np.array([[1, 2], [3, 4]])
print("matrix dot product(A,C)")
np.dot(A, C)

A3 = np.array([[1, 2], [3, 4], [5, 6]])
print("A3.shape")
print(A3.shape)

B0 = np.array([7, 8])
print("B0.shape")
print(B0.shape)

print("A3, B0 matrix dot product")
print(np.dot(A3, B0))

#3.3.3 Dot product within a neural network

X = np.array([1, 2])
print("X.shape")
print(X.shape)

W = np.array([[1, 3, 5], [2, 4, 6]])
print("W.shape")
print(W.shape)

Y = np.dot(X, W)
print("Y: np.dot(X,W)")
print(Y)


o = [[1, 2]]
m = [[1, 3, 5], [2, 4, 6]]
g = [1]

