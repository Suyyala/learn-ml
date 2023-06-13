# perceptron from scratch using python and pytorch tensors

import torch
import torch.optim as optim


# perceptron class definition
class Perceptron:
    def __init__(self, n_iter=1000, lr=0.01):
        self.W = None
        self.B = None
        self.n_iter = n_iter
        self.lr = lr
    
    def forward(self, X):
        return torch.matmul(X, self.W) + self.B
    
    def compute_loss(self, X, Y):
        Y_hat = self.forward(X)
        return torch.mean((Y_hat - Y) ** 2)
    
    def fit(self, X, Y):
        self.W = torch.zeros((X.shape[1], 1), requires_grad=True)
        self.B = torch.zeros(1, requires_grad=True)
        
        optimizer = optim.SGD([self.W, self.B], lr=self.lr)
        
        for i in range(self.n_iter):
            optimizer.zero_grad()
            loss = self.compute_loss(X, Y)
            loss.backward()
            optimizer.step()

    def predict(self, X):
        return self.forward(X)
    
    def rmse_error(self, y_true, y_pred):
        return torch.sqrt(torch.mean((y_true - y_pred) ** 2))
    
    def accuracy(self, y_true, y_pred):
        return torch.mean((y_true == y_pred).float())
    
    def evaluate(self, X, Y):
        Y_hat = self.predict(X)
        Y_hat[Y_hat > 0.5] = 1
        Y_hat[Y_hat <= 0.5] = 0
        return self.rmse_error(Y, Y_hat), self.accuracy(Y, Y_hat)
    
    def __str__(self):
        return f'W: {self.W}, B: {self.B}'
    
    def __repr__(self):
        return f'W: {self.W}, B: {self.B}'
    

# main function
def main():
    # load the data
    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    Y = torch.tensor([[0], [0], [0], [1]], dtype=torch.float32)
    
    # create a perceptron object
    model = Perceptron(n_iter=1000, lr=0.1)
    
    # train the model
    model.fit(X, Y)
    
    # evaluate the model
    rmse_error, accuracy = model.evaluate(X, Y)
    print(f'RMSE Error: {rmse_error}, Accuracy: {accuracy}')
    
    # print the parameters
    print(model)


# driver function
if __name__ == '__main__':
    main()
    

