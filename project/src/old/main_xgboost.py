""" This script train the project of ml 2019. """

import json

import numpy as np
from tqdm import tqdm
import xgboost as xgb

from util import get_data, parse_args

def main():
    """ Main function. """
    # parse args & get data
    args = parse_args()
    config = json.load(open(args.Config))
    paths = config['paths']
    x_train, y_train, x_test = get_data(paths['X_train'], paths['Y_train'], paths['X_test'], args.verbosity)

    # get & set parameters
    param = {
        # 'objective': 'reg:squarederror',
        'nthread': 8,
        'eval_metric': 'mae',
    }
    if 'param' in config:
        param.update(config['param'])
    max_num_round = config.get('max_num_round', 10000)
    add_pred_to_train = "pred_order" in config
    pred_order = config.get("pred_order", list(range(y_train.shape[1])))
    if verbosity >= 1:
        print('param =', param)
        print('max_num_round =', max_num_round)
        print('pred_order =', pred_order)

    # train & pred
    y_pred = np.zeros((x_test.shape[0], y_train.shape[1]))
    for target_dim in tqdm(pred_order):
        # prepare input data
        if add_pred_to_train and target_dim > 0:
            y_train_add = y_train[:, :target_dim].reshape(x_train.shape[0], target_dim)
            x_train_target = np.concatenate((x_train, y_train_add), axis=1)
            y_test_add = y_pred[:, :target_dim].reshape(x_test.shape[0], target_dim)
            x_test_target = np.concatenate((x_test, y_test_add), axis=1)
        else:
            x_train_target = x_train
            x_test_target = x_test
        y_train_target = y_train[:, target_dim]

        # convert data into xgb-format
        dtrain = xgb.DMatrix(x_train_target, label=y_train_target)
        dtest = xgb.DMatrix(x_test_target)

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

    np.savetxt(paths['Output'], y_pred, delimiter=',')

if __name__ == '__main__':
    main()
