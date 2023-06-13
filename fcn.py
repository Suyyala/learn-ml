# fully connected layer implementation in pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F


class FCN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FCN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))
    
    def predict(self, x):
        pred = self.forward(x)
        return torch.round(pred)
    
    def accuracy(self, y_true, y_pred):
        correct = (y_true == y_pred).float()
        return torch.mean(correct)
    
    def train_step(self, x, y):
        pred = self.forward(x)
        loss = F.binary_cross_entropy(pred, y)
        return loss
    
    def val_step(self, x, y):
        pred = self.predict(x)
        loss = F.binary_cross_entropy(pred, y)
        acc = self.accuracy(y, pred)
        return loss, acc
    
    def fit(self, train_dl, val_dl, epochs, lr, opt_func=torch.optim.SGD):
        optimizer = opt_func(self.parameters(), lr)
        history = dict(train_loss=[], train_acc=[], val_loss=[], val_acc=[])
        
        for epoch in range(epochs):
            # training phase
            self.train()
            train_losses = []
            train_accs = []
            
            for batch in train_dl:
                x, y = batch
                loss = self.train_step(x, y)
                acc = self.accuracy(y, self.predict(x))
                train_losses.append(loss)
                train_accs.append(acc)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            # validation phase
            self.eval()
            val_losses = []
            val_accs = []
            
            for batch in val_dl:
                x, y = batch
                loss, acc = self.val_step(x, y)
                val_losses.append(loss)
                val_accs.append(acc)
            
            train_loss = torch.stack(train_losses).mean().item()
            train_acc = torch.stack(train_accs).mean().item()
            val_loss = torch.stack(val_losses).mean().item()
            val_acc = torch.stack(val_accs).mean().item()
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f'Epoch: {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        return history
    
    def __str__(self):
        return f'FCN'
    
    def __repr__(self):
        return f'FCN'
    
    def summary(self):
        return f'Fully Connected Network with 3 hidden layers'
    
    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))
        self.eval()

 
# test the model
if __name__ == '__main__':
    # create a random dataset
    X = torch.randn((1000, 10))
    Y = torch.randint(0, 2, (1000, 1)).float()

    # create a fully connected network
    fcn = FCN(10, 1)

    
    # split the dataset into train and validation sets
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    Y_train, Y_val = Y[:split_idx], Y[split_idx:]
    train_ds = torch.utils.data.TensorDataset(X_train, Y_train)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=32)
    val_ds = torch.utils.data.TensorDataset(X_val, Y_val)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=32)

    # train the model
    history = fcn.fit(train_dl, val_dl, epochs=100, lr=0.01)
    
    # save the model
    fcn.save('fcn.pth')

    # load the model
    fcn.load('fcn.pth')

    # make predictions
    pred = fcn.predict(X)
    print(pred[:10])

    # evaluate the model
    acc = fcn.accuracy(Y, pred)
    print(acc)
