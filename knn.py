# knn algorithm for machine learning
import torch


class KNN:
    def __init__(self, k=5):
        self.k = k
    
    def fit(self, X, y):
        self.X = X
        self.y = y
    
    def predict(self, X):
        y_pred = []
        for x in X:
            distances = torch.sqrt(torch.sum((self.X - x) ** 2, axis=1))
            top_k = torch.topk(distances, self.k, largest=False)
            top_k_y = self.y[top_k.indices]
            y_pred.append(torch.mode(top_k_y).values)
        return torch.stack(y_pred)