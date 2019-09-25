""" This script do the question 11 ~ 12 of ml-hw4. """

import argparse

import matplotlib.pyplot as plt
import numpy as np

from knn import KNN

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
                        help='Output image to files. (default is output to screen)')
    args = parser.parse_args()

    # get data
    x_train, y_train, x_test, y_test = get_data(args.train, args.test)

    # train & test
    k_list = [1, 3, 5, 7, 9]
    ein_list = []
    eout_list = []
    for k in k_list:
        knn = KNN(k)
        knn.train(x_train, y_train)
        y_train_pred = knn.test(x_train)
        ein_list.append((y_train_pred != y_train).mean())
        y_test_pred = knn.test(x_test)
        eout_list.append((y_test_pred != y_test).mean())

    # plot

    # q_11
    plt.scatter(k_list, ein_list)
    plt.title('$E_{in}(g_{k-nbor})$ vs. $k$')
    plt.xlabel('$k$')
    plt.ylabel('$E_{in}(g_{k-nbor})$')
    if args.output_to_png:
        plt.savefig('q_11')
    else:
        plt.show()
    plt.clf()

    # q_12
    plt.scatter(k_list, eout_list)
    plt.title('$E_{out}(g_{k-nbor})$ vs. $k$')
    plt.xlabel('$k$')
    plt.ylabel('$E_{out}(g_{k-nbor})$')
    if args.output_to_png:
        plt.savefig('q_12')
    else:
        plt.show()
    plt.clf()

if __name__ == '__main__':
    main()
