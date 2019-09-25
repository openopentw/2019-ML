""" Nearest Neighbor. (g_uniform) """

import numpy as np

class NN:
    """ Nearest Neighbor. (g_uniform) """
    def __init__(self, gamma):
        self.gamma = gamma
        self.x = self.y = None

    def train(self, x_train, y_train):
        """ Train on x_train and y_train. """
        self.x = x_train
        self.y = y_train

    def test(self, x_test):
        """ Test on x_test. """
        dist = self.x[None, :, :] - x_test[:, None, :]
        dist = np.exp(-self.gamma * (dist ** 2).sum(2))
        return np.sign((self.y * dist).sum(1))
