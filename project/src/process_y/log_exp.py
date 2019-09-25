""" log and exp the data. """

import numpy as np

def preprocess(y):
    """ Preprocess on y-data. """
    return np.log(y)

def postprocess(y):
    """ Postprocess on y-data. """
    return np.exp(y)
