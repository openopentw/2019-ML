import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm

def train_n_test(x_train, y_train, x_test, config, verbosity):
    # get & set parameters
    param = {}
    if 'param' in config:
        param.update(config['param'])
    train_on_pred = ("train_on_pred" in config) and (config["train_on_pred"])
    add_pred_to_train = "pred_order" in config
    pred_order = config.get("pred_order", list(range(y_train.shape[1])))
    if verbosity >= 1:
        print('param =', param)
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

        # train & pred
        svr = svm.SVR(**param)
        bst = svr.fit(x_train_target, y_train_target)
        y_pred_target = svr.predict(x_test_target)
        y_pred[:, target_dim] = y_pred_target
        if train_on_pred:
            y_pred_target = svr.predict(x_train_target)
            y_train_pred[:, target_dim] = y_pred_target

    return y_pred
