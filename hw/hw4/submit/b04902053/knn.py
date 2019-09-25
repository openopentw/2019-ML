""" k Nearest Neighbor. """

import numpy as np

class KNN:
    """ k Nearest Neighbor """
    def __init__(self, k):
        self.k = k
        self.x = self.y = None

    def train(self, x_train, y_train):
        """ Train on x_train and y_train. """
        self.x = x_train
        self.y = y_train

    def test(self, x_test):
        """ Test on x_test. """
        # calculate distance
        dist = self.x[None, :, :] - x_test[:, None, :]
        dist = (dist ** 2).sum(2)

        # find the k closest x
        max_k_idx = np.argpartition(dist, self.k)[:, :self.k]

        return np.sign(self.y[max_k_idx].sum(1))
