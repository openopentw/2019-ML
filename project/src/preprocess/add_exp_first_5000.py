""" Just use the first 5000-dim data. """

import numpy as np

def preprocess(x):
    """ Preprocess on x-data. """
    return np.concatenate((x, np.exp(x[:, :5000])), axis=1)
