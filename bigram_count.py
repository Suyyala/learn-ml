# make more names
import torch

# load names.txt from disk
with open('upload/names.txt', 'r') as f:
    text = f.read()

print(text[:100])
# find all unique characters in the file
unique_chars = set(text)
unique_chars.remove('\n')
chars = sorted(list(unique_chars))


# add a terminator character as first character
chars = ['~'] + chars


print(chars)

vocab_size = len(chars)
print(f'Vocabulary size: {vocab_size}')

# create a mapping from char to index and vice versa
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}


# given a string convert it to indices
encode = lambda s: [char_to_idx[ch] for ch in s]
decode = lambda idx: ''.join([idx_to_char[i] for i in idx])

# split the text into lines
lines = text.split('\n')


# add terminator character to each line
lines = [ '~' + line + '~' for line in lines]
print(lines[:10])

# create two dimensional tensor to hold count of bigrams
bigram_count = torch.ones(vocab_size, vocab_size, dtype=torch.long)

# iterate through each line
for line in lines:
    # convert line to indices
    line = encode(line)
    # iterate through each character in the line
    for i in range(len(line) - 1):
        # increment the count of bigram
        bigram_count[line[i], line[i+1]] += 1

print(bigram_count.shape)
print(bigram_count[:100, :100])


#create a probability distribution
bigram_prob = bigram_count.float() / bigram_count.sum(dim=1, keepdim=True)
print(bigram_prob.shape)
print(bigram_prob.sum(dim=1))

# create a function to generate names
def generate_name():
    # start with a terminator character
    # sample a character from the distribution
    
    idx = torch.multinomial(bigram_prob[0], num_samples=1).item()
    name = '~'
    # append the character to the name

    # keep generating characters until terminator character is generated
    while True:
        # convert name to indices
        name_encoded = encode(name)
        # get the last character
        last_char = name_encoded[-1]
        # get the probability distribution of next character
        probs = bigram_prob[last_char]
        # sample from the distribution to get a new character
        new_char = torch.multinomial(probs, num_samples=1, replacement=True).item()
        # convert new character to string and append to the name
        if new_char == 0:
            break
        name += idx_to_char[new_char]
        # if terminator character is generated then stop
       
    return name[1:-1]

# evaluate the model
# log likelihood of a name
def negative_log_likelihood(name):
    # convert name to indices
    n  = 0
    name_encoded = encode(name)
    # get the probability of first character
    log_prob = torch.log(bigram_prob[0, name_encoded[0]])
    # iterate through each character in the name
    for i in range(len(name_encoded) - 1):
        # get the probability of next character
        log_prob += torch.log(bigram_prob[name_encoded[i], name_encoded[i+1]])
        n += 1
    return -log_prob.item() / n

# evaluate the model
# probability of a name
print(negative_log_likelihood('sridharp'))

# generate 10 names
for _ in range(10):
    print(generate_name())








