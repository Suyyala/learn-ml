from typing import Any
import torch

# linear layer implementation similar to pytorch

class Linear:
    def __init__(self, input_dim, output_dim, lr=0.01):
        self.W = torch.rand((input_dim, output_dim)) / input_dim ** 0.5
        self.B = torch.zeros(output_dim)
        self.lr = lr
    
    def __call__(self, X) -> Any:
        return self.forward(X)
    
    def forward(self, X):
        return X @ self.W + self.B
    
    def parameters(self):
        return [self.W, self.B]
    

# test linear layer
if __name__ == "__main__":
    linear = Linear(3, 2)
    X = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    print(linear(X))
    print(f'parameters:  {linear.parameters()}')

    
