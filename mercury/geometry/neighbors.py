import numpy as np
import sklearn.neighbors


def nearest_neighbors(reference, query):
    kdtree = sklearn.neighbors.KDTree(reference)
    indices = kdtree.query(query, return_distance=False)
    return indices[:, 0]


def average_distance(reference, query):
    indices = nearest_neighbors(reference, query)
    return np.linalg.norm(reference[indices] - query).mean()
