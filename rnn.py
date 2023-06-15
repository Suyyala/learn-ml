# recurrent neural network (RNN) class definition using pytorch

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


# recurrent neural network class definition
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNN, self).__init__()
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
    

# hyperparameters
batch_size = 128
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
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

# initialize model
model = RNN(input_size, hidden_size, output_size, n_layers)
print(model)

# initialize optimizer
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

# train the model
for epoch in range(n_epochs):
    for i, (X, Y) in enumerate(trainloader):
        optimizer.zero_grad()
        loss = model.compute_loss(model.forward(X), Y)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f'epoch: {epoch}, batch: {i}, loss: {loss.item()}')

# evaluate the model
with torch.no_grad():
    total_loss = 0
    total_accuracy = 0
    for X, Y in testloader:
        loss, accuracy = model.evaluate(X, Y)
        total_loss += loss
        total_accuracy += accuracy
    print(f'loss: {total_loss / len(testloader)}, accuracy: {total_accuracy / len(testloader)}')

# save the model
torch.save(model.state_dict(), 'rnn.pt')

# load the model
model = RNN(input_size, hidden_size, output_size, n_layers)
model.load_state_dict(torch.load('rnn.pt'))
model.eval()

# evaluate the model
with torch.no_grad():
    # get a random test image
    X, Y = testset[0]
    print(f'true label: {Y}')

    # predict the label
    pred = model.predict(X.unsqueeze(0))
    print(f'predicted label: {pred}')

    # visualize the image
    import matplotlib.pyplot as plt
    plt.imshow(X.squeeze(0), cmap='gray')
    plt.show()



