""" This script train the project of ml 2019. """

import argparse
import json
import time

import numpy as np
from tqdm import trange
from sklearn import svm
from sklearn.model_selection import GridSearchCV

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

    if verbosity >= 2:
        print('x_train:', x_train.shape)
        print(x_train)
        print('y_train:', y_train.shape)
        print(y_train)
        print('x_test:', x_test.shape)
        print(x_test)

    return x_train, y_train, x_test

def main():
    """ Main function. """
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('X_train', help='The path to the "X_train.npz".') # (47500, 10000)
    parser.add_argument('Y_train', help='The path to the "Y_train.npz".') # (2500, 10000)
    parser.add_argument('X_test', help='The path to the "X_test.npz".') # (47500, 3)
    parser.add_argument('Output', help='The path to the output file.') # (2500, 3)
    parser.add_argument('-v', '--verbosity', action='count', default=0,
                        help='increase output verbosity')
    args = parser.parse_args()
    verbosity = args.verbosity

    # get data
    x_train, y_train, x_test = get_data(args.X_train, args.Y_train, args.X_test, verbosity)

    y_pred = np.zeros((x_test.shape[0], y_train.shape[1]))
    for target_dim in trange(y_train.shape[1]):
        # prepare input data
        if target_dim > 0:
            y_train_add = y_train[:, :target_dim].reshape(x_train.shape[0], target_dim)
            x_train_target = np.concatenate((x_train, y_train_add), axis=1)
            y_test_add = y_pred[:, :target_dim].reshape(x_test.shape[0], target_dim)
            x_test_target = np.concatenate((x_test, y_test_add), axis=1)
        else:
            x_train_target = x_train
            x_test_target = x_test
        y_train_target = y_train[:, target_dim]

        # grid search pool
        paras = {
            'kernel': ['rbf', 'linear', 'poly'],
            'C': [1.5, 10],
            'gamma': [1e-7, 1e-4],
            'epsilon': [0.1, 0.2, 0.5, 0.3],
        }

        if verbosity >= 1:
            start = time.time()

        # grid search to get best param
        svr = svm.SVR()
        gscv = GridSearchCV(svr, paras, cv=3, verbose=1)
        gscv.fit(x_train_target, y_train_target)
        best_params = gscv.best_params_
        json.dump(best_params, open(
            './svm_best_params_dim_{}.json'.format(target_dim), 'w'))

        if verbosity >= 1:
            print('Spending {} seconds to do the grid search.'.format(
                time.time() - start))
            start = time.time()

        # train & pred
        clf = svm.SVR(**best_params)
        clf.fit(x_train_target, y_train_target)
        y_pred_target = clf.predict(x_test_target)
        y_pred[:, target_dim] = y_pred_target

        if verbosity >= 1:
            print('Spending {} seconds to do the train & pred.'.format(
                time.time() - start))

    np.savetxt(args.Output, y_pred, delimiter=',')

if __name__ == '__main__':
    main()
