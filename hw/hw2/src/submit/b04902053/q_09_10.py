""" This script do the experiments of questions 9 & 10. """

import argparse

import numpy as np
from sklearn.linear_model import RidgeClassifier

TRAIN_NUM = 400
LMBD_LIST = [0.05, 0.5, 5, 50, 500]

def get_data(data_path):
    """ Load and split data."""
    # load data
    raw_data = np.genfromtxt(data_path)
    data = np.ones((raw_data.shape[0], raw_data.shape[1] + 1)) # add x_0 = 1
    data[:, :-2] = raw_data[:, :-1]
    data[:, -1] = raw_data[:, -1] == 1

    # split x/y & split train/test
    train = data[:TRAIN_NUM]
    test = data[TRAIN_NUM:]
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = test[:, :-1]
    y_test = test[:, -1]
    return x_train, y_train, x_test, y_test

def main():
    """ Main function. """
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='The path to the data. (hw2_lssvm_all.dat)')
    args = parser.parse_args()

    data_path = args.data

    # get data
    x_train, y_train, x_test, y_test = get_data(data_path)

    # run linear ridge and print errors
    np.random.seed(0)
    e_in_list = []
    e_out_list = []
    for lmbd in LMBD_LIST:
        clf = RidgeClassifier(lmbd)
        clf.fit(x_train, y_train)
        e_in = (clf.predict(x_train) != y_train).mean()
        e_in_list.append(e_in)
        e_out = (clf.predict(x_test) != y_test).mean()
        e_out_list.append(e_out)

    # print errors
    print('E_in:', e_in_list)
    min_e_in = min(e_in_list)
    argmin_e_in = [LMBD_LIST[idx] for idx, e_in in enumerate(e_in_list) if e_in == min_e_in]
    print('min lambda:', argmin_e_in)
    print('corresponding E_in:', min_e_in)
    print('')
    print('E_out:', e_out_list)
    min_e_out = min(e_out_list)
    argmin_e_out = [LMBD_LIST[idx] for idx, e_out in enumerate(e_out_list) if e_out == min_e_out]
    print('min lambda:', argmin_e_out)
    print('correspondoutg E_out:', min_e_out)

if __name__ == '__main__':
    main()
