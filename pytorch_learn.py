# learnign pytorh tensor operations

import torch

# 1. Create a tensor of size (2,3) with uninitialized values
x = torch.empty(2, 3)
print(x)

# 2. Create a tensor of size (2,3) with random values
x = torch.rand(2, 3)
print(x)

# 3. Create a tensor of size (2,3) with zeros
x = torch.zeros(2, 3)
print(x)

# 4. Create a tensor of size (2,3) with ones
x = torch.ones(2, 3)
print(x)

# 5. Create a tensor of size (3,2) with values from [0, 1) and of type long
x = torch.arange(0, 6).reshape(3, 2).long()

# 6. Create a tensor of size (1,) with value 2
x = torch.tensor([2])
print(x)

# 7. Create a tensor of size (1,2) with values 1 and 2
x = torch.tensor([1, 2])
print(x)

# 8. Create a tensor of size (2,1) with values 1 and 2
x = torch.tensor([[1], [2]])
print(x)

# 9. Create a 3x3 tensor with values ranging from 0 to 8
x = torch.arange(0, 9).reshape(3, 3)
print(x)

# 10. Create a 3x3 identity matrix
x = torch.eye(3)
print(x)

# 11. Create a tensor with values from a uniform distribution on [0, 1)
x = torch.rand(2, 3)
print(x)

# 12. Get the datatype of the tensor x
print(x.dtype)

# 13. Get the size of the tensor x
print(x.size())

# 14. Add the tensor x and y
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
print(x + y)


# 15. Subtract the tensor y from x
print(x - y)

# 16. Multiply the tensor x by y
print(x * y)

# 17. Divide the tensor x by y
print(x / y)

# 18. Get the dot product of the tensor x and y
print(x @ y)

# 19. Get the element-wise product of the tensor x and y
print(x * y)

# 20. sum all the elements in x
print(x.sum())

# 21. sum the elements in the rows of x
print(x.sum(axis=1))

# 22. sum the elements in the columns of x
print(x.sum(axis=0))

# 23. Transpose x
print(x.T)

# 24. Reshape x from (2,3) to (3,2)
print(x.reshape(3, 2))


# 25. Get the maximum value in x
print(x.max())

# 26. Get the minimum value in x
print(x.min())

# 27. Get the maximum value of each row of x
print(x.max(axis=1))

# 28. Get the minimum value of each column in x
print(x.min(axis=0))

# 29. Get the argmax of x
print(x.argmax())

# 30. Get the argmin of x
print(x.argmin())

# 31. Get the argmax of each row of x
print(x.argmax(axis=1))

# 32. Get the argmin of each column in x
print(x.argmin(axis=0))

# 33. Get the square root of x
print(x.sqrt())

# 34. Get the sin of x
print(x.sin())

# 35. Get the natural log of x
print(x.log())

# 36. Get the absolute value of x
print(x.abs())

# 37. Get the exponential of all elements in x
print(x.exp())

# 38. Get the log2 of all elements in x
print(x.log2())

# 39. Get the log10 of all elements in x
print(x.log10())

# 40. Get the floor of all elements in x
print(x.floor())

# 41. Get the ceil of all elements in x
print(x.ceil())

# 42. Clamp all elements in x between 0 and 1
print(x.clamp(0, 1))

# 43. Get the number of non-zero elements in x
print(x.nonzero())

# 44. Get the number of non-zero elements in x
print(x.nonzero())

# 45. Get the unique elements in x
print(x.unique())

# 46. Get the number of unique elements in x
print(x.unique().size())

# 47. Get the number of elements in x
print(x.numel())

# 48. Get the number of dimensions of x
print(x.dim())

# 49. Get the shape of x
print(x.shape)

# 50. Get the size of x
print(x.size())

# 51. Get the number of bytes in x
print(x.numel() * x.element_size())

# 52. Get the number of bytes in x
print(x.numel() * x.element_size())

#  tensor to numpy array
x = torch.tensor([1, 2])
print(x.numpy())

#  numpy array to tensor
import numpy as np

x = np.array([1, 2])
print(torch.from_numpy(x))

# 53. Create a tensor of size (2,3) with all elements equal to 1
x = torch.ones(2, 3)
print(x)

# 54. Create a tensor of size (2,3) with all elements equal to 0
x = torch.zeros(2, 3)
print(x)
