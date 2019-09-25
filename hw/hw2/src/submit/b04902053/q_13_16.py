""" This script do the experiments of questions 13. """

import argparse

import matplotlib.pyplot as plt
import numpy as np

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

def stump_train(x_train, y_train, u):
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

def stump_test(x_test, para):
    """ Test the result by given para. """
    thres = np.ones(x_test.shape[0]) * para['theta']
    return para['s'] * ((x_test[:, para['i']] > thres) * 2 - 1)

def main():
    """ Main function. """
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('train', help='The path to the training data. (hw2_adaboost_train.dat)')
    parser.add_argument('test', help='The path to the testing data. (hw2_adaboost_test.dat)')
    parser.add_argument('-o', '--output_fig', action='store_false',
                        help='Output image to a file. (default is output to screen)')
    args = parser.parse_args()

    # get data
    x_train, y_train, x_test, y_test = get_data(args.train, args.test)

    # train
    ein_g_list = []
    G_train_pred = np.zeros_like(y_train)
    ein_G_list = []
    G_pred = np.zeros_like(y_test)
    eout_G_list = []
    u_list = []
    alpha_list = []
    u = np.ones(x_train.shape[0]) / x_train.shape[0]
    for _ in range(ITERS):
        para = stump_train(x_train, y_train, u)
        y_train_pred = stump_test(x_train, para)
        y_pred = stump_test(x_test, para)

        # update u
        epsilon = ((y_train_pred != y_train) * u).sum() / u.sum()
        scaling_factor = np.sqrt((1 - epsilon) / epsilon)
        u[y_train_pred != y_train] *= scaling_factor
        u[y_train_pred == y_train] /= scaling_factor

        # get ein, e_out, alpha
        ein = (y_train_pred != y_train).mean()
        ein_g_list.append(ein)
        alpha = np.log(scaling_factor)
        alpha_list.append(alpha)
        u_list.append(u.sum())

        # accumulate G(x)
        G_train_pred += alpha * y_train_pred
        ein_G_list.append((np.sign(G_train_pred) != y_train).mean())
        G_pred += alpha * y_pred
        eout_G_list.append((np.sign(G_pred) != y_test).mean())

    # Q13
    plt.plot(ein_g_list)
    plt.title('$t$ versus $E_{in}(g_{t})$')
    plt.xlabel('$t$')
    plt.ylabel('$E_{in}(g_{t})$')
    if args.output_fig:
        plt.show()
    else:
        plt.savefig('./q_13')
    plt.clf()
    print(ein_g_list[-1])

    # Q14
    plt.plot(ein_G_list)
    plt.title('$t$ versus $E_{in}(G_{t})$')
    plt.xlabel('$t$')
    plt.ylabel('$E_{in}(G_{t})$')
    if args.output_fig:
        plt.show()
    else:
        plt.savefig('./q_14')
    plt.clf()
    print(ein_G_list[-1])

    # Q15
    plt.plot(u_list)
    plt.title('$t$ versus $U_{t}$')
    plt.xlabel('$t$')
    plt.ylabel('$U_{t}$')
    if args.output_fig:
        plt.show()
    else:
        plt.savefig('./q_15')
    plt.clf()
    print(u_list[-1])

    # Q16
    plt.plot(eout_G_list)
    plt.title('$t$ versus $E_{out}(G_{t})$')
    plt.xlabel('$t$')
    plt.ylabel('$E_{out}(G_{t})$')
    if args.output_fig:
        plt.show()
    else:
        plt.savefig('./q_16')
    plt.clf()
    print(eout_G_list[-1])

if __name__ == '__main__':
    main()
