import numpy as np
from sklearn.neighbors import NearestNeighbors
from knn.nearest_neighbors import NearestNeighborsFinder


class KNNClassifier:
    def __init__(self, n_neighbors, algorithm='my_own', metric='euclidean', weights='uniform'):
        if algorithm == 'my_own':
            finder = NearestNeighborsFinder(n_neighbors=n_neighbors, metric=metric)
        elif algorithm in ('brute', 'ball_tree', 'kd_tree',):
            finder = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm, metric=metric)
        else:
            raise ValueError("Algorithm is not supported", metric)

        if weights not in ('uniform', 'distance'):
            raise ValueError("Weighted algorithm is not supported", weights)

        self._finder = finder
        self._weights = weights

    def fit(self, X, y=None):
        self._finder.fit(X)
        self._labels = np.asarray(y)
        return self

    def _predict_precomputed(self, indices, distances):
        with np.errstate(divide='ignore'):
            dinv = np.nan_to_num(1 / distances)

        distinct_labels = np.sort(np.array(list(set(self._labels))))
        neigh_labels = self._labels[indices]

        if self._weights == 'uniform':
            return (neigh_labels[:, :, np.newaxis] == distinct_labels).sum(
                axis=1).argmax(axis=1)
        else:
            return (dinv[:, :, np.newaxis] * (neigh_labels[:, :, np.newaxis] == distinct_labels)).sum(
                axis=1).argmax(axis=1)

    def kneighbors(self, X, return_distance=False):
        return self._finder.kneighbors(X, return_distance=return_distance)

    def predict(self, X):
        distances, indices = self.kneighbors(X, return_distance=True)
        return self._predict_precomputed(indices, distances)


class BatchedKNNClassifier(KNNClassifier):
    '''
    Нам нужен этот класс, потому что мы хотим поддержку обработки батчами
    в том числе для классов поиска соседей из sklearn
    '''

    def __init__(self, n_neighbors, algorithm='my_own', metric='euclidean', weights='uniform', batch_size=None):
        KNNClassifier.__init__(
            self,
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            weights=weights,
            metric=metric,
        )
        self._batch_size = batch_size

    def kneighbors(self, X, return_distance=False):
        if self._batch_size is None or self._batch_size >= X.shape[0]:
            return super().kneighbors(X, return_distance=True)
        ind = []
        for x in np.array_split(X, len(X)/self._batch_size + 1, axis=0):
            ind.append(super().kneighbors(x, return_distance=return_distance))
        res = np.concatenate(ind, axis=1)

        return res[0], res[1].astype(int)
