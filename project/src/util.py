""" This module contains utility functions. """

import argparse
import time

import numpy as np

def get_data(x_train_path, y_train_path, x_test_path, verbosity=1):
    """ Load and split data."""
    if verbosity >= 1:
        start = time.time()

    # load data
    x_train = np.load(x_train_path)['arr_0']
    y_train = np.load(y_train_path)['arr_0']
    x_test = np.load(x_test_path)['arr_0']

    if verbosity >= 1:
        print('Spending {} seconds to load data.'.format(time.time() - start))

    # print('x_train:', x_train.shape)
    # print(x_train)
    # print('y_train:', y_train.shape)
    # print(y_train)
    # print('x_test:', x_test.shape)
    # print(x_test)

    return x_train, y_train, x_test

def parse_args():
    """ Parse the args and return it. """
    parser = argparse.ArgumentParser()
    parser.add_argument('Config', help='The path to the config file.') # dict
    parser.add_argument('-v', '--verbosity', action='count', default=0,
                        help='increase output verbosity')
    args = parser.parse_args()
    return args
