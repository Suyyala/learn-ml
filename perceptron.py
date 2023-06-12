# sample perceptron implementation using pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Perceptron:
    def __init__(self, n_iter=1000, lr=0.01) -> None:
        self.W = None
        self.B = None
        self.n_iter = n_iter
        self.lr = lr
        self.grad = None
        self.loss = None
        self.optimizer = None

    def loss_mse_dw(self, X, Y):
        return 2 *  torch.mean(((X @ self.W) + self.B - Y) * X, axis=0).reshape(-1, 1)
    
    def loss_mse(self, X, Y):
        return ((X @ self.W) + self.B - Y) ** 2
    
    def loss_mse_db(self, X, Y):
        return 2 * torch.mean((X @ self.W) + self.B - Y, axis=0)
    
    def forward(self, X):
        return torch.sigmoid(X @ self.W + self.B)
    
    def backward(self, X, Y, y_pred):
        dW = torch.mean((y_pred - Y) * X, axis=0).reshape(-1, 1)
        dB = torch.mean(y_pred - Y, axis=0)
        return dW, dB
    
    def fit(self, X, Y):
        self.W = torch.zeros((X.shape[1], 1))
        self.B = torch.zeros(1)
        self.X = X
        self.Y = Y
        self.optimizer = optim.SGD([self.W, self.B], lr=self.lr)
        for i in range(self.n_iter):
            y_pred = self.forward(X)
            self.loss = self.loss_mse(X, Y)
            self.grad = self.backward(X, Y, y_pred)
            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()
    

