import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class TextDataset:
    def __init__(self, path):
        self.text = open(path).read().split()
        self.vocab = list(set(self.text))
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx2word = {idx: word for idx, word in enumerate(self.vocab)}
    
    def __len__(self):
        return len(self.text) - 1
    
    def __getitem__(self, idx):
        input_word_idx = self.word2idx[self.text[idx]]
        target_word_idx = self.word2idx[self.text[idx + 1]]
        return input_word_idx, target_word_idx


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(RNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        x = self.embed(x)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x


def train(model, dataloader, optimizer, criterion, n_epochs):
    for epoch in range(n_epochs):
        for input_word, target_word in dataloader:
            optimizer.zero_grad()
            output = model(input_word.unsqueeze(1))
            loss = criterion(output.squeeze(1), target_word)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item()}")


def predict_next_word(model, dataset, prompt_word):
    model.eval()
    with torch.no_grad():
        input_idx = torch.tensor(dataset.word2idx[prompt_word]).long()
        output = model(input_idx.unsqueeze(0).unsqueeze(1))
        predicted_idx = output.argmax(-1).item()
        return dataset.idx2word[predicted_idx]


# Load dataset
# link to dataset: https://www.gutenberg.org/files/11/11-0.txt
dataset = TextDataset('./data/text/11-0.txt')
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Initialize model
embedding_dim = 128
hidden_dim = 128
model = RNN(vocab_size=len(dataset.vocab), embedding_dim=embedding_dim, hidden_dim=hidden_dim)

# Train model
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
n_epochs = 10
train(model, dataloader, optimizer, criterion, n_epochs)

# Predict next word for a given prompt
prompt_word = "Alice"  # example prompt
predicted_word = predict_next_word(model, dataset, prompt_word)
print(f"Given the prompt word '{prompt_word}', the predicted next word is '{predicted_word}'")
