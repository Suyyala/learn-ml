

# load MNIST dataset from torchvision.datasets
# split into train and test
# train model
# predict
# print accuracy
# plot
# Path: mnist_test.py

import torchvision.datasets as datasets
import torch
import matplotlib.pyplot as plt
from torch.utils.data import random_split, TensorDataset
import torchvision.transforms as transforms

from knn import KNN


# Loading MNIST data
train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())


# Preparing data for k-NN. We need to reshape the images to 1D array
X_train = train_dataset.data.reshape(-1, 28 * 28)
y_train = train_dataset.targets
X_test = test_dataset.data.reshape(-1, 28 * 28)
y_test = test_dataset.targets

X_train = X_train[:5000]
y_train = y_train[:5000]
X_test = X_test[:10]
y_test = y_test[:10]
print(len(X_train), len(X_test))


# train model
print("Training model...")
model = KNN(5)
model.fit(X_train, y_train)
# predict
print("Predicting...")
y_pred_test = model.predict(X_test)

# print accuracy
print(f"Accuracy {torch.sum(y_pred_test == y_test).item() / len(y_test)}")




