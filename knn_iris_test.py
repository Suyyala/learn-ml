import torch
from sklearn import datasets
from sklearn.model_selection import train_test_split
from knn import KNN
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Convert the numpy arrays to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shape of the training and testing sets
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# knn algorithm for machine learning
knn = KNN(5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(f"Accuracy {torch.sum(y_pred == y_test).item() / len(y_test)}")


# Visualization
feature_names = iris.feature_names
target_names = iris.target_names

plt.figure(figsize=(14, 5))

# Plot the different pairs of features
for i in range(3):
    plt.subplot(1, 3, i + 1)

    # Training points
    for target in set(y_train):
        selection = y_train == target
        plt.scatter(X_train[selection, i], X_train[selection, i + 1], label=target_names[target], marker='o', alpha=0.7)

    # Test points
    for target in set(y_test):
        selection = y_test == target
        plt.scatter(X_test[selection, i], X_test[selection, i + 1], label=target_names[target] + ' Test', marker='x', alpha=0.7)
    
    plt.xlabel(feature_names[i])
    plt.ylabel(feature_names[i + 1])
    plt.legend()

plt.show()