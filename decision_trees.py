# decision tree classifier from scratch using pytorch

import torch

class Node:
    def __init__(self, predicted_class):
        self.left = None
        self.right = None
        self.feature = None
        self.threshold = None
        self.predicted_class = predicted_class


class DecisionTreeClassifier:
    def __init__(self, max_depth=10):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.n_classes = len(y.unique())
        self.n_features = X.size(1)
        self.tree = self._grow_tree(X, y)

    def predict(self, X):
        return torch.stack([self._traverse_tree(inputs, self.tree) for inputs in X])

    def _grow_tree(self, X, y, depth=0):
        # get samples per class
        n_samples_per_class = [torch.sum(y == i) for i in range(self.n_classes)]
        # get the class with the most samples
        predicted_class = torch.argmax(torch.tensor(n_samples_per_class))
        node = Node(predicted_class.item())

        if depth < self.max_depth:
            feature, threshold = self._find_best_split(X, y)
            if feature is not None:
                indices_left = X[:, feature] < threshold
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature = feature
                node.threshold = threshold
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def _find_best_split(self, X, y):
        n_samples = y.size(0)
        if n_samples <= 1:
            return None, None

        num_parent = [torch.sum(y == c) for c in range(self.n_classes)]
        best_entropy = self._calculate_entropy(num_parent, n_samples)
        best_idx, best_thr = None, None

        for idx in range(self.n_features):
            thresholds, classes = zip(*sorted(zip(X[:, idx].tolist(), y.tolist())))
            num_left = [0] * self.n_classes
            num_right = num_parent.copy()

            for i in range(1, n_samples):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                entropy = self._calculate_weighted_entropy(num_left, num_right, i, n_samples)
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if entropy < best_entropy:
                    best_entropy = entropy
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2
        return best_idx, best_thr

    @staticmethod
    def _calculate_entropy(num_list, total):
        return sum(-(n / total) * torch.log2(torch.tensor(n / total, dtype=torch.float32)) for n in num_list if n > 0)

    def _calculate_weighted_entropy(self, num_left, num_right, n_left, n_samples):
        entropy_left = self._calculate_entropy(num_left, n_left)
        entropy_right = self._calculate_entropy(num_right, n_samples - n_left)
        return (n_left * entropy_left + (n_samples - n_left) * entropy_right) / n_samples

    def _traverse_tree(self, inputs, node):
        if node.left is None:
            return torch.tensor(node.predicted_class, dtype=torch.int)

        if inputs[node.feature] < node.threshold:
            return self._traverse_tree(inputs, node.left)
        else:
            return self._traverse_tree(inputs, node.right)

