# support vector machines from scratch using pure pytorch and python

import torch
import torch.optim as optim


# support vector machine class definition
class SVM:
    def __init__(self, n_iter=1000, lr=0.01, C=1.0):
        self.W = None
        self.B = None
        self.n_iter = n_iter
        self.lr = lr
        self.C = C
    
    def forward(self, X):
        return X @ self.W + self.B
    
    def compute_loss(self, X, Y):
        hinge_loss = torch.mean(torch.clamp(1 - Y * self.forward(X), min=0))
        reg_loss = 0.5 * torch.sum(self.W ** 2)
        return hinge_loss + self.C * reg_loss
    
    def fit(self, X, Y):
        self.W = torch.zeros((X.shape[1], 1), requires_grad=True)
        self.B = torch.zeros(1, requires_grad=True)
        
        optimizer = optim.SGD([self.W, self.B], lr=self.lr)
        
        for i in range(self.n_iter):
            optimizer.zero_grad()
            loss = self.compute_loss(X, Y)
            loss.backward()
            optimizer.step()

   

