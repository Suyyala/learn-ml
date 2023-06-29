# learning torch sum and mean

import torch

x = torch.arange(10, dtype=torch.float32)
print(x)
print(f'sum = {x.sum()}')
print(f'sum vaue = {x.sum().item()}')

print(f'mean = {x.mean()}')

# Creating a 2D tensor
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f'x {x}')

print(f'sum along dim=0 {x.sum(dim=0)}')
print(f'sum along axis=0 {x.sum(axis=0)}')
print(f'sum along axis=1 {x.sum(dim=1)}')

# create 2D tensor
x = torch.arange(12).view(-1, 3).float()
print(f'x {x} shape {x.shape}')
w = torch.ones(3, 1).float()
print(f'w {w} shape {w.shape}')

print(f'matrix multiplication {torch.matmul(x, w)}')

