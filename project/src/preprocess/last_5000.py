""" Just use the first 5000-dim data. """

def preprocess(x):
    """ Preprocess on x-data. """
    return x[:, 5000:]
