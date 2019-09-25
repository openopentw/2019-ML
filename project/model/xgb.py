""" This module train & test by xgboost. """

import numpy as np
from tqdm import tqdm
import xgboost as xgb

def gen_eta(eta_config_list):
    """ Generate a function that returns the eta by the config. """
    def eta(round_, _):
        """ Return the eta by the round number. """
        for i, eta_config in enumerate(eta_config_list[:-1]):
            if (eta_config['round'] <= round_
                    < eta_config_list[i + 1]['round']):
                return eta_config['eta']
        return eta_config_list[-1]['eta']
    return eta

def train_n_test(x_train, y_train, x_test, config, verbosity):
    """ Train & Test. """
    # get & set parameters
    param = {
        # 'objective': 'reg:squarederror',
        'nthread': 8,
        'eval_metric': 'mae',
    }
    if 'param' in config:
        param.update(config['param'])
    max_num_round = config.get('max_num_round', 10000)
    train_on_pred = ("train_on_pred" in config) and (config["train_on_pred"])
    add_pred_to_train = "pred_order" in config
    pred_order = config.get("pred_order", list(range(y_train.shape[1])))
    if verbosity >= 1:
        print('param =', param)
        print('max_num_round =', max_num_round)
        print('pred_order =', pred_order)

    # callbacks
    callbacks_cv = None
    callbacks_train = None
    if 'adaptive_eta' in config:
        learning_rates = config['adaptive_eta']
        callbacks_cv = [xgb.callback.reset_learning_rate(gen_eta(learning_rates))]
        callbacks_train = [xgb.callback.reset_learning_rate(gen_eta(learning_rates))]
        if verbosity >= 1:
            print('callbacks =', config['adaptive_eta'])

    # train & pred
    if train_on_pred:
        y_train_pred = np.zeros_like(y_train)
    y_pred = np.zeros((x_test.shape[0], y_train.shape[1]))
    for target_dim in tqdm(pred_order):
        # prepare input data
        if add_pred_to_train and target_dim > 0:
            if train_on_pred:
                y_train_add = y_train_pred[:, :target_dim].reshape(x_train.shape[0], target_dim)
            else:
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
        dtrain_test = xgb.DMatrix(x_train_target)
        dtest = xgb.DMatrix(x_test_target)

        # cross validation
        cv_output = xgb.cv(param,
                           dtrain,
                           max_num_round,
                           early_stopping_rounds=20,
                           verbose_eval=25,
                           callbacks=callbacks_cv)
        if verbosity >= 1:
            print('best num_round = ', len(cv_output))
        num_round = len(cv_output)

        # train & pred
        bst = xgb.train(param, dtrain, num_round, callbacks=callbacks_train)
        y_pred_target = bst.predict(dtest)
        y_pred[:, target_dim] = y_pred_target
        if train_on_pred:
            y_pred_target = bst.predict(dtrain_test)
            y_train_pred[:, target_dim] = y_pred_target

    return y_pred
