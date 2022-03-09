import numpy as np


def euclidean_distance(x, y):
    squared_matrix = np.sum(x ** 2, axis=1)[:, np.newaxis] + np.sum(y ** 2, axis=1)[np.newaxis, :] - 2 * (x@y.T)
    return np.sqrt(squared_matrix)

def cosine_distance(x, y):
    return 1 - (x @ y.T) / np.sqrt(np.sum(x ** 2, axis=1)[:, np.newaxis] * np.sum(y ** 2, axis=1)[np.newaxis, :])
