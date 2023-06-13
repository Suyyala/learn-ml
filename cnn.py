# convolution neural network (CNN) using pytorch

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


# convolution neural network class definition
class CNN(nn.Module):
    def __init__(self, n_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6,
                               kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16,
                               kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, n_classes)
    
    def forward(self, X):
        X = self.conv1(X)
        X = self.relu1(X)
        X = self.maxpool1(X)
        X = self.conv2(X)
        X = self.relu2(X)
        X = self.maxpool2(X)
        X = X.view(-1, 16 * 5 * 5)
        X = self.fc1(X)
        X = self.relu3(X)
        X = self.fc2(X)
        X = self.relu4(X)
        X = self.fc3(X)
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
        return f'conv1: {self.conv1}, conv2: {self.conv2}, fc1: {self.fc1}, fc2: {self.fc2}, fc3: {self.fc3}'
    
    def __repr__(self):
        return f'conv1: {self.conv1}, conv2: {self.conv2}, fc1: {self.fc1}, fc2: {self.fc2}, fc3: {self.fc3}'
    

# main function
def main():
    # load the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                            (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                                shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                                shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
                'ship', 'truck')
    
    # create the model
    model = CNN(n_classes=10)
    print(model)

    # train the model
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            X, Y = data
            optimizer.zero_grad()
            loss = model.compute_loss(model.forward(X), Y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 2000}')
                running_loss = 0.0
        
    # evaluate the model
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            X, Y = data
            outputs = model.forward(X)
            _, predicted = torch.max(outputs.data, 1)
            total += Y.size(0)
            correct += (predicted == Y).sum().item()
    
    print(f'accuracy of the network on the 10000 test images: {100 * correct / total}%')

    # evaluate the model per class
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            X, Y = data
            outputs = model.forward(X)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == Y).squeeze()
            for i in range(4):
                label = Y[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    for i in range(10):
        print(f'accuracy of {classes[i]}: {100 * class_correct[i] / class_total[i]}%')


# driver function
if __name__ == '__main__':
    main()
