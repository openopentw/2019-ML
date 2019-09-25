import numpy as np
from tqdm import tqdm
import lightgbm

def train_n_test(x_train, y_train, x_test, config, verbosity):
    # get & set parameters
    param = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'num_threads': 8,
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

        # convert data into lightgbm-format
        dtrain = lightgbm.Dataset(x_train_target, label=y_train_target)
        dtrain_test = lightgbm.Dataset(x_train_target)
        dtest = lightgbm.Dataset(x_test_target)

        # cross validation
        cv_output = lightgbm.cv(param,
                                dtrain,
                                max_num_round,
                                early_stopping_rounds=20,
                                verbose_eval=25)
        if verbosity >= 1:
            print('best num_round = ', len(cv_output))
        num_round = len(cv_output)

        # train & pred
        bst = lightgbm.train(param, dtrain, num_round)
        y_pred_target = bst.predict(dtest)
        y_pred[:, target_dim] = y_pred_target
        if train_on_pred:
            y_pred_target = bst.predict(dtrain_test)
            y_train_pred[:, target_dim] = y_pred_target

    return y_pred
