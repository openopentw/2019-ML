""" This script do the question 13 ~ 14 of ml-hw4. """

import argparse

import matplotlib.pyplot as plt
import numpy as np

from nn import NN

def get_data(train_path, test_path):
    """ Get the data. """
    train = np.genfromtxt(train_path)
    test = np.genfromtxt(test_path)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = test[:, :-1]
    y_test = test[:, -1]
    return x_train, y_train, x_test, y_test

def main():
    """ Main function. """
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('train', help='hw4_train.dat')
    parser.add_argument('test', help='hw4_test.dat')
    parser.add_argument('-o', '--output_to_png', default=False, action='store_true',
                        help='Output image to files. (default is display on screen)')
    args = parser.parse_args()

    # get data
    x_train, y_train, x_test, y_test = get_data(args.train, args.test)

    # train & test
    gamma_list = [0.001, 0.1, 1, 10, 100]
    ein_list = []
    eout_list = []
    for gamma in gamma_list:
        nn = NN(gamma)
        nn.train(x_train, y_train)
        y_train_pred = nn.test(x_train)
        ein_list.append((y_train_pred != y_train).mean())
        y_test_pred = nn.test(x_test)
        eout_list.append((y_test_pred != y_test).mean())

    # plot

    # q_11
    plt.scatter(gamma_list, ein_list)
    plt.title('$E_{in}(g_{uniform})$ vs. $gamma$')
    plt.xlabel('$gamma$')
    plt.ylabel('$E_{in}(g_{uniform})$')
    if args.output_to_png:
        plt.savefig('q_13')
    else:
        plt.show()
    plt.clf()

    # q_12
    plt.scatter(gamma_list, eout_list)
    plt.title('$E_{out}(g_{uniform})$ vs. $gamma$')
    plt.xlabel('$gamma$')
    plt.ylabel('$E_{out}(g_{uniform})$')
    if args.output_to_png:
        plt.savefig('q_14')
    else:
        plt.show()
    plt.clf()

if __name__ == '__main__':
    main()
