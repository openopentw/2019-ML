""" This script train the project of ml 2019. """

import importlib
import json
import os

import numpy as np

from util import get_data, parse_args

def main():
    """ Main function. """
    # parse args & get data
    args = parse_args()
    config = json.load(open(args.Config))
    paths = config['paths']

    # load model
    model = importlib.import_module('model.{}'.format(config['model']))

    # load data
    x_train, y_train, x_test = get_data(
        paths['X_train'], paths['Y_train'], paths['X_test'], args.verbosity
    )

    # preprocess on x
    if 'preprocess' in config:
        preprocess = importlib.import_module('preprocess.{}'.format(config['preprocess']))
        x_train = preprocess.preprocess(x_train)
        x_test = preprocess.preprocess(x_test)

    # preprocess on y
    if 'process_y' in config:
        process_y = importlib.import_module('process_y.{}'.format(config['process_y']))
        y_train = process_y.preprocess(y_train)

    # train & pred
    y_pred = model.train_n_test(
        x_train,
        y_train,
        np.concatenate((x_train, x_test)),
        config,
        args.verbosity
    )

    # postprocess on y
    if 'process_y' in config:
        y_pred = process_y.postprocess(y_pred)

    y_train_pred = y_pred[:x_train.shape[0]]
    y_test_pred = y_pred[x_train.shape[0]:]

    # output
    if 'Output_train' in config:
        np.savetxt(paths['Output_train'], y_train_pred, delimiter=',')
    np.savetxt(paths['Output'], y_test_pred, delimiter=',')

if __name__ == '__main__':
    main()
