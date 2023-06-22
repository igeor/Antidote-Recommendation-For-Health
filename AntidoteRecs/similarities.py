import numpy as np
import pandas as pd

def cosine_similarity(x, y):
    ## x and y are numpy arrays of shape [1, n]
    return np.dot(x, y.T) / (np.linalg.norm(x) * np.linalg.norm(y))

def euclidean_similarity(x, y):
    ## x and y are numpy arrays of shape [1, n]
    return 1 / (1 + np.linalg.norm(x - y))