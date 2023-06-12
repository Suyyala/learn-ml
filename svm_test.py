#svm test
from svm import SVM

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split


# generate data
X, Y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=0)
Y[Y == 0] = -1
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Convert Y_train to be shape (n_samples, 1) instead of (n_samples,)
Y_train = Y_train.reshape(-1, 1)

# train svm
svm = SVM(n_iter=1000, lr=0.01, C=1.0)
svm.fit(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32))

# predict
with torch.no_grad():
    y_pred = svm.forward(torch.tensor(X_test, dtype=torch.float32))
    y_pred = (y_pred > 0).float()
    y_pred[y_pred == 0] = -1

# Compute accuracy
correct = (y_pred.view(-1) == torch.tensor(Y_test, dtype=torch.float32)).float()
accuracy = correct.mean().item()
print("Accuracy:", accuracy)

# plot
#plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_test)
#plt.show()

# Plotting
plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, cmap='autumn')
plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_test, cmap='autumn', marker='x')

# Plot decision boundary and margins
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = svm.forward(torch.tensor(xy, dtype=torch.float32)).detach().numpy().reshape(XX.shape)

# Plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Support Vector Machine Decision Boundary')
plt.show()
