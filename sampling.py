# sampling functions

import torch

def sample_uniform(low, high, size):
    return torch.rand(size) * (high - low) + low


def sample_normal(mean, std, size):
    return torch.randn(size) * std + mean


def sample_bernoulli(prob, size):
    return (torch.rand(size) < prob).float()


def sample_categorical(prob, size):
    return torch.multinomial(prob, size, replacement=True)

# test sampling functions
if __name__ == '__main__':
    torch.manual_seed(0)
    print("category sampling")
    random = torch.rand(10)
    print(random)
    print(sample_categorical(random, 10))

    print("uniform sampling")
    print(sample_uniform(0, 5, 10))
