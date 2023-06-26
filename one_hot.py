# one hot encoding of bigram
import torch

# create a dataset
data = 'hello world'
vocab = ['~', 'h', 'e', 'l', 'o', 'w', 'r', 'd', ' ']
vocab_size = len(vocab)
print(f'Vocabulary size: {vocab_size}')

# create a mapping from char to index and vice versa
char_to_idx = {ch: i for i, ch in enumerate(vocab)}
idx_to_char = {i: ch for i, ch in enumerate(vocab)}

# given a string convert it to indices
encode = lambda s: [char_to_idx[ch] for ch in s]

# given indices convert it to string
decode = lambda idx: ''.join([idx_to_char[i] for i in idx])

# lets encode the entire text into a tensor
data = torch.tensor(encode(data), dtype=torch.long)
print(data.size())
print(data.shape, data.dtype)
print(data)

one_hot_encoded = torch.functional.F.one_hot(data, vocab_size)
print(one_hot_encoded.shape)
print(one_hot_encoded)
