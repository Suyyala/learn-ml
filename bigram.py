# bigram model
import torch

with open('upload/shakespeare.txt', 'r') as f:
    text = f.read()

print(text[:100])

# unique chars in the text file
chars = sorted(list(set(text)))
print(chars)
vocab_size = len(chars)
print(f'Vocabulary size: {vocab_size}')

# create a mapping from char to index and vice versa
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# given a string convert it to indices
encode = lambda s: [char_to_idx[ch] for ch in s]

# given indices convert it to string
decode = lambda idx: ''.join([idx_to_char[i] for i in idx])

print(encode('hello'))
print(decode(encode('hello')))

# lets encode the entire text into a tensor
data = torch.tensor(encode(text), dtype=torch.long)
print(data.size())
print(data.shape, data.dtype)

# create a dataset
def create_dataset(data, seq_length):
    X = []
    y = []
    
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    
    return torch.stack(X), torch.stack(y)

# create a dataset
seq_length = 8
X, y = create_dataset(data, seq_length=seq_length)
print(X.shape, y.shape)
print(X[:2], y[:2])

# split the dataset into train and validation sets
split_idx = int(len(X) * 0.8)
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]
print(X_train.shape, y_train.shape)

# create a dataloader
batch_size = 32
train_ds = torch.utils.data.TensorDataset(X_train, y_train)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size)
val_ds = torch.utils.data.TensorDataset(X_val, y_val)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)


# create a bigram model
class Bigram(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Bigram, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = torch.nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = x.mean(dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def predict(self, x):
        return torch.argmax(self.forward(x), dim=1)
    
    def accuracy(self, x, y):
        return torch.mean((self.predict(x) == y).float())
    
    def generate(self, x, n):
        res = []
        for i in range(n):
            y = self.predict(x).unsqueeze(1)  # Make y a 2D tensor with shape [batch_size, 1]
            res.append(y)
            x = torch.cat([x[:, 1:], y], dim=1)
        return torch.cat(res, dim=1)


# create a model
model = Bigram(vocab_size, embedding_dim=10, hidden_dim=10)
print(model)

# loss function
loss_fn = torch.nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# train the model
epochs = 10
for epoch in range(epochs):
    for i, (X, y) in enumerate(train_dl):
        # forward pass
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            print(f'Epoch: {epoch+1}, Batch: {i+1}/{len(train_dl)}, Loss: {loss.item():.4f}')
    
    # validation loss
    with torch.no_grad():
        val_loss = sum([loss_fn(model(X), y) for X, y in val_dl]) / len(val_dl)
        print(f'Epoch: {epoch+1}, Validation Loss: {val_loss.item():.4f}')

# save the model
torch.save(model.state_dict(), 'models/bigram.pt')

# generate text
with torch.no_grad():
    print(decode(model.generate(torch.tensor([[char_to_idx['t']]], dtype=torch.long), 100).squeeze().tolist()))



    







