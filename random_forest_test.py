# test using random forest
import torch
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from random_forest import RandomForestClassifier


# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target


# Convert the numpy arrays to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Print the shape of the training and testing sets
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

random_forest = RandomForestClassifier(n_trees=10, max_depth=10)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)
print(f"Accuracy {torch.sum(y_pred == y_test).item() / len(y_test)}")
