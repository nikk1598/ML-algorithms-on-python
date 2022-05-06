import numpy as np
from collections import defaultdict
import time
import random


class LinearModel:
    def __init__(
        self,
        loss_function,
        batch_size=None,
        step_alpha=1,
        step_beta=0,
        tolerance=1e-5,
        max_iter=1000,
        random_seed=153,
        **kwargs
    ):
        """
        Parameters
        ----------
        loss_function : BaseLoss inherited instance
            Loss function to use
        batch_size : int
        step_alpha : float
        step_beta : float
            step_alpha and step_beta define the learning rate behaviour
        tolerance : float
            Tolerace for stop criterio.
        max_iter : int
            Max amount of epoches in method.
        """
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.random_seed = random_seed
        self.w = 0

    def fit(self, X, y, w_0=None, trace=False, X_val=None, y_val=None):
        """

        Parameters
        —------—
        X : numpy.ndarray or scipy.sparse.csr_matrix
        2d matrix, training set.
        y : numpy.ndarray
        1d vector, target values.
        w_0 : numpy.ndarray
        1d vector in binary classification.
        2d matrix in multiclass classification.
        Initial approximation for SGD method.
        trace : bool
        If True need to calculate metrics on each iteration.
        X_val : numpy.ndarray or scipy.sparse.csr_matrix
        2d matrix, validation set.
        y_val: numpy.ndarray
        1d vector, target values for validation set.

        Returns
        —---—
        : dict
        Keys are 'time', 'func', 'func_val'.
        Each key correspond to list of metric values after each training epoch.
        """

        history = defaultdict(list)

        if self.batch_size == None:
            self.batch_size = X.shape[0]

        if w_0 is None:
             self.w =  np.random.random_sample(X.shape[1])
        else:
            self.w = np.copy(w_0)

        w_prev = np.zeros(X.shape[1])

        for i in range(1, self.max_iter +1):
            start_time = time.time()
            

            if (self.w - w_prev).dot(self.w - w_prev) <= self.tolerance:
                print('Method completed with num of iterations: ', i)
                break

            w_prev = np.copy(self.w)

            random_indices_for_train = random.sample(range(0, X.shape[0]), X.shape[0])
            j = 0

            while j <= X.shape[0] - self.batch_size:
                X_to_use = X[random_indices_for_train][j:j+self.batch_size]
                y_to_use = y[random_indices_for_train][j:j+self.batch_size]

                grad = self.loss_function.grad(X_to_use, y_to_use, self.w)
                learning_rate = self.step_alpha / (i ** self.step_beta)
                self.w -= learning_rate * grad
                j += self.batch_size

            if trace == True:
                 history['time'].append(time.time() - start_time)
                 history['func'].append(self.loss_function.func(X, y, self.w))
                 if X_val is None:
                     history['func_val'].append(None)
                 else:
                     history['func_val'].append(self.loss_function.func(X_val, y_val, self.w))

        if trace == True:
            return history

    def predict(self, X, threshold=0):
        """

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, test set.
        threshold : float
            Chosen target binarization threshold.

        Returns
        -------
        : numpy.ndarray
            answers on a test set
        """

        prediction = X.dot(self.w)
        prediction[prediction < threshold] = -1
        prediction[prediction >= threshold] = 1
        return prediction

    def get_optimal_threshold(self, X, y):
        """
        Get optimal target binarization threshold.
        Balanced accuracy metric is used.

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, validation set.
        y : numpy.ndarray
            1d vector, target values for validation set.

        Returns
        -------
        : float
            Chosen threshold.
        """
        if self.loss_function.is_multiclass_task:
            raise TypeError('optimal threhold procedure is only for binary task')

        weights = self.get_weights()
        scores = X.dot(weights)
        y_to_index = {-1: 0, 1: 1}

        # for each score store real targets that correspond score
        score_to_y = dict()
        score_to_y[min(scores) - 1e-5] = [0, 0]
        for one_score, one_y in zip(scores, y):
            score_to_y.setdefault(one_score, [0, 0])
            score_to_y[one_score][y_to_index[one_y]] += 1

        # ith element of cum_sums is amount of y <= alpha
        scores, y_counts = zip(*sorted(score_to_y.items(), key=lambda x: x[0]))
        cum_sums = np.array(y_counts).cumsum(axis=0)

        # count balanced accuracy for each threshold
        recall_for_negative = cum_sums[:, 0] / cum_sums[-1][0]
        recall_for_positive = 1 - cum_sums[:, 1] / cum_sums[-1][1]
        ba_accuracy_values = 0.5 * (recall_for_positive + recall_for_negative)
        best_score = scores[np.argmax(ba_accuracy_values)]
        return best_score

    def get_weights(self):
        """
        Get model weights

        Returns
        -------
        : numpy.ndarray
            1d vector in binary classification.
            2d matrix in multiclass classification.
            Initial approximation for SGD method.
        """
        return self.w

    def get_objective(self, X, y):
        """
        Get objective.

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix.
        y : numpy.ndarray
            1d vector, target values for X.

        Returns
        -------
        : float
        """
        return self.loss_function.func(X, y, self.get_weights())
