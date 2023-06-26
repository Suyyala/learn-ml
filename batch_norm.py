# batch normalization implementation in pytorch

import torch

class BatchNorm1d:
    def __init__(self, momentum=0.9, eps=1e-5, training=True) -> None:
        self.eps = eps
        self.momentum = momentum
        # buffers
        self.running_mean = torch.zeros(1)
        self.running_var =torch.ones(1)
        self.training = training
        # parameters
        self.gamma = torch.ones(1)
        self.beta = torch.zeros(1)

    def __call__(self, X):
        if self.training:
            x_mean = torch.mean(X, dim=0, keepdim=True)
            x_var = torch.var(X, dim=0, keepdim=True)
        else:
            x_mean = self.running_mean
            x_var = self.running_var
        X_hat = (X - x_mean) / torch.sqrt(x_var + self.eps)
        out = self.gamma * X_hat + self.beta
        # update buffers
        if self.training:
            with torch.no_grad(): # no gradient calculation
                self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * x_mean
                self.running_var = self.momentum * self.running_var + (1 - self.momentum) * x_var
        return out
    
    def parameters(self):
        # gamma and beta are learnable parameters
        return [self.gamma, self.beta]
    

if __name__ == "__main__":
    bn = BatchNorm1d()
    X = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    print(bn(X))
    print(f'parameters:  {bn.parameters()}')


            
