""" This script do the experiments of questions 13. """

import argparse

import numpy as np
import matplotlib.pyplot as plt

ITERS = 300

def get_data(train_path, test_path):
    """ Load and split data."""
    # load data
    train = np.genfromtxt(train_path)
    test = np.genfromtxt(test_path)

    # split x/y
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = test[:, :-1]
    y_test = test[:, -1]
    return x_train, y_train, x_test, y_test

def quick_stump_train(x_train, y_train, u): # pylint: disable=invalid-name
    """ Obtain best g by given u. """
    u_sum = u.sum()
    best_err = u_sum + 100
    best_para = {
        's': None,
        'i': None,
        'theta': None,
    }
    for i in range(x_train.shape[1]):
        sorted_idx = x_train[:, i].argsort()
        sorted_x = x_train[sorted_idx, i]
        sorted_y = y_train[sorted_idx]
        sorted_u = u[sorted_idx]

        # theta = -infty
        err = ((sorted_y != 1) * sorted_u).sum()
        if err < best_err:
            best_err = err
            best_para['s'] = 1
            best_para['i'] = i
            best_para['theta'] = sorted_x[0] - 1
        if u_sum - err < best_err:
            best_err = u_sum - err
            best_para['s'] = -1
            best_para['i'] = i
            best_para['theta'] = sorted_x[0] - 1

        # other thetas
        for theta_idx in range(x_train.shape[0] - 1):
            err += sorted_u[theta_idx] * sorted_y[theta_idx]
            if err < best_err:
                best_err = err
                best_para['s'] = 1
                best_para['i'] = i
                best_para['theta'] = (sorted_x[theta_idx] + sorted_x[theta_idx + 1]) / 2
            if u_sum - err < best_err:
                best_err = u_sum - err
                best_para['s'] = -1
                best_para['i'] = i
                best_para['theta'] = (sorted_x[theta_idx] + sorted_x[theta_idx + 1]) / 2

    return best_para

def stump_train(x_train, y_train, u): # pylint: disable=invalid-name
    """ Obtain best g by given u. """
    u_sum = u.sum()
    best_err = u_sum + 100
    best_para = {
        's': None,
        'i': None,
        'theta': None,
    }
    for i in range(x_train.shape[1]):
        sorted_idx = x_train[:, i].argsort()
        sorted_x = x_train[sorted_idx, i]
        sorted_y = y_train[sorted_idx]
        sorted_u = u[sorted_idx]

        # theta_idx = 0
        err = ((sorted_y != 1) * sorted_u).sum()
        if err < best_err:
            best_err = err
            best_para['s'] = 1
            best_para['i'] = i
            best_para['theta'] = sorted_x[0] - 1
        if u_sum - err < best_err:
            best_err = u_sum - err
            best_para['s'] = -1
            best_para['i'] = i
            best_para['theta'] = sorted_x[0] - 1

        for theta_idx in range(1, x_train.shape[0]):
            err = (((sorted_y[:theta_idx] != -1) * sorted_u[:theta_idx]).sum()
                   + ((sorted_y[theta_idx:] != 1) * sorted_u[theta_idx:]).sum())
            if err < best_err:
                best_err = err
                best_para['s'] = 1
                best_para['i'] = i
                best_para['theta'] = (sorted_x[theta_idx - 1] + sorted_x[theta_idx]) / 2
            if u_sum - err < best_err:
                best_err = u_sum - err
                best_para['s'] = -1
                best_para['i'] = i
                best_para['theta'] = (sorted_x[theta_idx - 1] + sorted_x[theta_idx]) / 2

    return best_para

def stump_test(x_test, para):
    """ Test the result by given para. """
    thres = np.ones(x_test.shape[0]) * para['theta']
    return para['s'] * ((x_test[:, para['i']] > thres) * 2 - 1)

def main():
    """ Main function. """
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('train', help='The path to the training data.')
    parser.add_argument('test', help='The path to the testing data.')
    args = parser.parse_args()

    # get data
    x_train, y_train, x_test, y_test = get_data(args.train, args.test)

    # train
    e_in_list = []
    alpha_list = []
    u = np.ones(x_train.shape[0]) / x_train.shape[0] # pylint: disable=invalid-name
    for _ in range(ITERS):
        para = stump_train(x_train, y_train, u)
        y_train_pred = stump_test(x_train, para)

        # update u
        epsilon = ((y_train_pred != y_train) * u).sum() / u.sum()
        scaling_factor = np.sqrt((1 - epsilon) / epsilon)
        u[y_train_pred != y_train] *= scaling_factor
        u[y_train_pred == y_train] /= scaling_factor

        # get e_in, e_out, alpha
        e_in = (y_train_pred != y_train).mean()
        e_in_list.append(e_in)
        alpha_list.append(np.log(scaling_factor))

    # plot
    plt.plot(e_in_list, label='Ein(g)')
    # plt.ylim((-0.1, 1))
    plt.xlabel('t')
    plt.legend()
    plt.show()
    print(e_in_list[0])
    print(alpha_list[0])
    print(e_in_list[-1])

if __name__ == '__main__':
    main()
