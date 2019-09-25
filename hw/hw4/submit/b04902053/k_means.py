""" k-Means """

import numpy as np

class KMeans:
    """ k-Means """
    def __init__(self, k, seed=0):
        self.k = k
        self.data = self.mu_list = self.s_idx = None
        np.random.seed(seed)

    def fit(self, data):
        """ Fit on data. """
        self.data = data

        # init
        mu_idx_list = np.random.choice(self.data.shape[0], self.k)
        self.mu_list = self.data[mu_idx_list]
        self.s_idx = np.zeros(self.data.shape[0], int)

        while True:
            # optimize S
            dist = self.data[:, None, :] - self.mu_list[None, :, :]
            dist = (dist ** 2).sum(2)
            s_idx = np.argmin(dist, axis=-1)

            # if converge: break
            if (s_idx == self.s_idx).all():
                break
            self.s_idx = s_idx

            # optimize mu
            for part_idx in range(self.k):
                idcs = self.s_idx == part_idx
                if idcs.any():
                    self.mu_list[part_idx] = self.data[idcs].mean(0)
                else:
                    self.mu_list[part_idx] = 0

    def calc_err(self):
        """ Calculate the error of the clustering. """
        err = 0
        for part_idx in range(self.k):
            dist = self.data[self.s_idx == part_idx] - self.mu_list[part_idx]
            dist = (dist ** 2).sum()
            err += dist
        err /= self.data.shape[0]
        return err
