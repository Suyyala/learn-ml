import torch
import torch.nn as nn
import torch.optim as optim


# simple RNN model
# input_size: number of features
# hidden_size: number of hidden units
# output_size: number of classes

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        
        # input to hidden
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        # hidden to output
        self.h2o = nn.Linear(hidden_size, output_size)
        # softmax activation function for output layer
        self.softmax = nn.LogSoftmax(dim=1)
        
        # Criterion and Optimizer
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=lr)

    def forward(self, input, hidden):
        # concatenate input and hidden state
        combined = torch.cat((input, hidden), 1)
        # hidden state with tanh activation
        hidden = torch.tanh(self.i2h(combined))
        # output
        output = self.h2o(hidden)
        # softmax activation function for output layer
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)
    
    def train_step(self, inputs, labels):
        hidden = self.init_hidden()
        self.optimizer.zero_grad()
        
        for i in range(inputs.size()[0]):
            output, hidden = self.forward(inputs[i], hidden)
        
        loss = self.criterion(output, labels)
        loss.backward()
        self.optimizer.step()
        
        return output, loss.item()

    def fit(self, X, y, epochs=10):
        for epoch in range(epochs):
            for inputs, labels in zip(X, y):
                output, loss = self.train_step(inputs, labels)
                print(f"Epoch {epoch} Loss {loss}")
                
    def predict(self, inputs):
        hidden = self.init_hidden()
        for i in range(inputs.size()[0]):
            output, hidden = self.forward(inputs[i], hidden)
        return output


if __name__ == "__main__":
    # define parameters
    input_size = 4
    hidden_size = 10
    output_size = 3
    lr = 0.01
    epochs = 10

    # define model
    model = RNN(input_size, hidden_size, output_size, lr)

    # define data
    X = torch.randn(10, 4)
    y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])

    # train model
    model.fit(X, y, epochs)

    # predict
    print(model.predict(X))