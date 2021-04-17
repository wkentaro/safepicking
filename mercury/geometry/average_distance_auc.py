import numpy as np
import sklearn.neighbors


def average_distance_auc(reference, query, max_threshold=0.01, plot=False):
    kdtree = sklearn.neighbors.KDTree(reference)
    distances, _ = kdtree.query(query, k=1)

    x = np.linspace(0, max_threshold)
    y = [(distances <= xi).sum() / distances.size for xi in x]
    auc = sklearn.metrics.auc(x, y) / max_threshold

    if plot:
        import matplotlib.pyplot as plt

        plt.subplot(121)
        plt.hist(distances)
        plt.subplot(122)
        plt.plot(x, y)
        plt.show()

    return auc
