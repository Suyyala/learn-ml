# learnign pytorh tensor operations

import torch

print("create a tensor  with uninitilized values")
x = torch.empty(2, 3)
print(x)

print("create a tensor  with random values")
x = torch.rand(2, 3)
print(x)

print("create a tensor  with zeros")
x = torch.zeros(2, 3)
print(x)


print("tensor of size with ones")
x = torch.ones(2, 3)
print(x)

print("tensor reshaping")
x = torch.arange(5, 10)
x = x.view(5, 1)
x = x.expand(5, 2)
print(x)

print("tensor attributes")
print(x.shape)
print(x.size())
print(x.dtype)
print(x.device)
print(x.layout)

print("tensor operations")
print ("addition")
x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])
z1 = x + y
print(z1)

print ("subtraction")
z2 = x - y
print(z2)

print ("multiplication")
z3 = x * y
print(z3)

print ("division")
z4 = x / y
print(z4)

print ("dot product")
z5 = torch.dot(x, y)
print(z5)

print ("matrix multiplication")
x1 = torch.rand((2, 5))
y1 = torch.rand((5, 3))
z6 = torch.mm(x1, y1)
print(z6)


print("tensor clone")
x = torch.tensor([1, 2, 3])
y = x.clone()
y[0] = 10
print(x)
print(y)


print("tensor concatenation")
x1 = torch.tensor([1, 2, 3])
y1 = torch.tensor([4, 5, 6])
z1 = torch.cat((x1, y1))
print(z1)

print("tensor stacking")
x1 = torch.tensor([1, 2, 3])
y1 = torch.tensor([4, 5, 6])
z1 = torch.stack((x1, y1))
print(z1)

print("tensor indexing")
batch_size = 10
features = 25
x = torch.rand((batch_size, features))
print(x)
print(f'x[0].shape= {x[0].shape}')
print(x[:, 0])
print(x[:, -1])

print("tensor slicing")
x = torch.arange(10)
print(x)
print(x[:5])
print(x[5:])
print(x[4:7])

print("tensor broadcasting")
x1 = torch.ones((5, 5))
x2 = torch.ones((1, 5))
z = x1 - x2
print(z)

print("tensor flattening")
x = torch.arange(10).view(2, 5)
print(x)
print(x.flatten())

print("tensor transpose")
x = torch.arange(10).view(2, 5)
print(x)
print(x.T)

print("tensor elementwise")
x = torch.arange(10)
print(x)
print(x.pow(2))
print(x ** 2)
print(x.sqrt())
print(x.log())
print(x[1:].sum())


print("tensor comparison")
x = torch.arange(10)
print(x)
print(x < 5)
print(x > 5)
print(x == 5)
print(x != 5)
print(torch.sum(x < 5))
print(torch.sum(x > 5))

print("tensor any and all")
print(torch.any(x > 5, dim=0))
print(torch.all(x > 5, dim=0))

print("tensor sorting")
x = torch.randperm(10)
print(x)
print(x.sort(descending=True))

print("tensor max and argmax")
x = torch.randperm(3)
print(x)
print(x.max())
print(x.argmax())

print("tensor reduction")
x = torch.arange(10, dtype=torch.float32)
print(x)
print(x.sum())
print(x.prod())
print(x.mean())
print(x.std())


print("tensor unsqueeze")
x = torch.arange(10).view(2, 5)
print(x)
print(x.unsqueeze(0).shape)
print(x.unsqueeze(1).shape)
print(x.shape)

print("tensor cuda")
print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
    











