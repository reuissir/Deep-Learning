import torch

weights = torch.ones(4, requires_grad=True)

#training loop
for epoch in range(3):
    model_output = (weights*3).sum()
    
    #call gradients
    model_output.backward()
    
    print(weights.grad)
    # empty gradients
    weights.grad.zero_()
    
optimizer = torch.optim.SGD(weights, lr=0.01)
optimizer.step()
optimizer.zero_grad()