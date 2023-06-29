# simple matrix multiplication

import torch

x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
w = torch.tensor([[10, 11], [12, 13], [14, 15]])
print(f'x {x} shape {x.shape}')

print(f'w {w} shape {w.shape}')

# vector dot product from scratch
def dot(x, y):
    # check if the dimensions are correct
    print(f'dot product of {x} and {y}')
    print(f'x.shape {x.shape} y.shape {y.shape}')
    assert x.shape[0] == y.shape[0]
    z = 0
    for i in range(x.shape[0]):
        z += x[i] * y[i]
    return z



# matrix multiplication from scratch using dot product
def matmul(x, w):
    y = torch.zeros((x.shape[0], w.shape[1]))
    for i in range(x.shape[0]):
        for j in range(w.shape[1]):
            # i is the row index of x
            # j is the column index of w
            y[i, j] = dot(x[i, :], w[:, j])
    return y

print(f'dot product from scratch {dot(torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]))}')

print(f'matrix multiplication from scratch {matmul(x, w)}')
print(f'matrix multiplication {torch.matmul(x, w)}')