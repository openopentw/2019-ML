""" Only use the first 5000-dim data and normalize the them. """

def preprocess(x):
    """ Preprocess on x-data. """
    x = x[:, :5000]
    return (x - x.mean(axis=0)) / x.std(axis=0)
