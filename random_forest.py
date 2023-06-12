from decision_trees import DecisionTreeClassifier
import torch


class RandomForestClassifier:
    def __init__(self, n_trees=100, max_depth=10):
        self.n_trees = n_trees
        self.max_depth = max_depth

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X, y)
            self.trees.append(tree)
    
    def predict(self, X):
        predictions = torch.stack([tree.predict(X) for tree in self.trees])
        return torch.mode(predictions).values
    
    