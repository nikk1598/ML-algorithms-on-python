import numpy as np
from sklearn.model_selection import KFold, BaseCrossValidator
from sklearn.metrics import accuracy_score
from knn.classification import BatchedKNNClassifier


def knn_cross_val_score(X, y, k_list, scoring, cv=None, **kwargs):
    y = np.asarray(y)

    if scoring == "accuracy":
        scorer = accuracy_score
    else:
        raise ValueError("Unknown scoring metric", scoring)

    if cv is None:
        cv = KFold(n_splits=5)
    elif not isinstance(cv, BaseCrossValidator):
        raise TypeError("cv should be BaseCrossValidator instance", type(cv))

    knn = BatchedKNNClassifier(np.amax(k_list), algorithm=kwargs['algorithm'],
                               metric=kwargs['metric'], weights=kwargs['weights'], batch_size=5)

    scores = {x: [] for x in k_list}
    for train_index, test_index in cv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        knn.fit(X_train, y_train)
        distances, indices = knn.kneighbors(X_test, return_distance=True)
        for x in k_list:
            scores[x].append(scorer(knn._predict_precomputed(indices[:, 0:x], distances[:, 0:x]), y_test))
    return scores
