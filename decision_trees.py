# decision tree classifier from scratch using pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class DecisionTreeClassifier:
    def __init__(self, max_depth=10):
        self.max_depth = max_depth
        