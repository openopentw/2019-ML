""" Reduce the dimension lower by pca. """

import numpy as np
from sklearn.decomposition import PCA

def preprocess(x_train, x_test, param):
    """ Preprocess on x-data. """
    pca = PCA(**param)
    x = np.concatenate((x_train, x_test))
    pca.fit(x)
    return pca.transform(x_train), pca.transform(x_test)
