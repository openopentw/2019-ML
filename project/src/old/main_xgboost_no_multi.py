""" This script train the project of ml 2019. """

import argparse
import json
import time

import numpy as np
from tqdm import trange
import xgboost as xgb

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

def main():
    """ Main function. """
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('X_train', help='The path to the "X_train.npz".') # (47500, 10000)
    parser.add_argument('Y_train', help='The path to the "Y_train.npz".') # (2500, 10000)
    parser.add_argument('X_test', help='The path to the "X_test.npz".') # (47500, 3)
    parser.add_argument('Param', help='The path to the param file.') # dict
    parser.add_argument('Output', help='The path to the output file.') # (2500, 3)
    parser.add_argument('-v', '--verbosity', action='count', default=0,
                        help='increase output verbosity')
    args = parser.parse_args()

    # get data
    x_train, y_train, x_test = get_data(args.X_train, args.Y_train, args.X_test, args.verbosity)
    param = json.load(open(args.Param))

    # parameters
    param.update({
        # 'objective': 'reg:squarederror',
        'nthread': 8,
        'eval_metric': 'mae',
    })
    print('param =', param)
    max_num_round = 10000

    y_pred = np.zeros((x_test.shape[0], y_train.shape[1]))
    for target_dim in trange(y_train.shape[1]):
        # convert data into xgb-format
        dtrain = xgb.DMatrix(x_train, label=y_train[:, target_dim])
        dtest = xgb.DMatrix(x_test)

        # cross validation
        cv_output = xgb.cv(param,
                           dtrain,
                           max_num_round,
                           early_stopping_rounds=20,
                           verbose_eval=25)
        if args.verbosity >= 1:
            print('best num_round = ', len(cv_output))
        num_round = len(cv_output)

        # train & pred
        bst = xgb.train(param, dtrain, num_round)
        y_pred_target = bst.predict(dtest)
        y_pred[:, target_dim] = y_pred_target

    np.savetxt(args.Output, y_pred, delimiter=',')

if __name__ == '__main__':
    main()
