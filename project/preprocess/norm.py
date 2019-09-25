""" normalize the data. """

def preprocess(x):
    """ Preprocess on x-data. """
    return (x - x.mean(axis=0)) / x.std(axis=0)
