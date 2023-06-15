# implementing seq2seq model from scratch in pytorch

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm

# set seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# define hyperparameters
batch_size = 64
n_epochs = 10
learning_rate = 0.001
momentum = 0.9
n_classes = 10
input_size = 28
hidden_size = 128
output_size = 10
n_layers = 1

# load data
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                        download=True, transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

# define model
class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(Seq2Seq, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, X):
        X, _ = self.rnn(X)
        X = self.fc(X[:, -1, :])
        return X
    
    def compute_loss(self, X, Y):
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn(X, Y)
    
    def predict(self, X):
        return torch.argmax(self.forward(X), dim=1)
    
    def accuracy(self, X, Y):
        return torch.mean((self.predict(X) == Y).float())
    
    def evaluate(self, X, Y):
        return self.compute_loss(self.forward(X), Y), self.accuracy(X, Y)
    
    def __str__(self):
        return f'rnn: {self.rnn}, fc: {self.fc}'
    
    def __repr__(self):
        return f'rnn: {self.rnn}, fc: {self.fc}'
    

# instantiate model
model = Seq2Seq(input_size, hidden_size, output_size, n_layers)

# define optimizer
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

# train model
for epoch in range(n_epochs):
    train_loss = 0.0
    train_acc = 0.0
    model.train()
    for X, Y in tqdm(trainloader):
        optimizer.zero_grad()
        loss = model.compute_loss(model.forward(X), Y)
        acc = model.accuracy(model.forward(X), Y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc += acc.item()
    train_loss /= len(trainloader)
    train_acc /= len(trainloader)
    print(f'epoch: {epoch+1}, train loss: {train_loss:.4f}, train acc: {train_acc:.4f}')
    
    test_loss = 0.0
    test_acc = 0.0
    model.eval()
    for X, Y in tqdm(testloader):
        loss = model.compute_loss(model.forward(X), Y)
        acc = model.accuracy(model.forward(X), Y)
        test_loss += loss.item()
        test_acc += acc.item()
    test_loss /= len(testloader)
    test_acc /= len(testloader)
    print(f'epoch: {epoch+1}, test loss: {test_loss:.4f}, test acc: {test_acc:.4f}')

# save model
torch.save(model.state_dict(), 'seq2seq.pt')

# load model
model = Seq2Seq(input_size, hidden_size, output_size, n_layers)

# load weights
model.load_state_dict(torch.load('seq2seq.pt'))

# evaluate model
test_loss = 0.0
test_acc = 0.0
model.eval()
for X, Y in tqdm(testloader):
    loss = model.compute_loss(model.forward(X), Y)
    acc = model.accuracy(model.forward(X), Y)
    test_loss += loss.item()
    test_acc += acc.item()
test_loss /= len(testloader)
test_acc /= len(testloader)
print(f'test loss: {test_loss:.4f}, test acc: {test_acc:.4f}')

# predict
X, Y = next(iter(testloader))
print(model.predict(X))
print(Y)
