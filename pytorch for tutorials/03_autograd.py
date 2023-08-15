import torch

# calculate gradient of some function with respect to x

# set argument requires_grad=True
# add AddBackward function that calculates gradient during backpropogation
# function y in respect to x - dy/dx
x = torch.randn(3, requires_grad=True)
print(x)
# AddBackWard
y = x + 2
print(y)
# MulBackWard
z= y*y*2
print(z)
# MeanBackward
# z = z.mean()

v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)

# calculate gradient of z with respect to x
z.backward(v) # dz/dx
# create jacobian matrix with partial derivatives
# multiply jacobian matrix with a gradient vector --> chain rule 
print(x.grad, ": gradients")


# prevent pytorch from tracking the history in computational graph
# x.requires_grad_(False)
# x.detach()
# with torch.no_grad():

x.requires_grad_(False)
print(x)

# create new tensor with the same values that doesn't require the gradient
y = x.detach()
print(y)

with torch.no_grad():
    y = x + 2
    print(y)