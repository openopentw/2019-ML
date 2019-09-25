""" This script do the experiments of questions 14 ~ 16. """

import argparse
import importlib

import matplotlib.pyplot as plt
import numpy as np

from model import RF

if importlib.util.find_spec('tqdm'):
    from tqdm import tqdm
else:
    def tqdm(x):
        """ An empty function that has the same spec as tqdm. """
        return x

def get_data(train_path, test_path):
    """ Get the training & test data, and split them. """
    train = np.genfromtxt(train_path)
    test = np.genfromtxt(test_path)

    x_train = train[:, :-1] # shape = (100, 2)
    y_train = train[:, -1]  # shape = (100,)
    x_test = test[:, :-1]   # shape = (1000, 2)
    y_test = test[:, -1]    # shape = (1000,)

    return x_train, y_train, x_test, y_test

def main():
    """ Main function. """
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('train', help='The path to the training data. (hw3_train.dat)')
    parser.add_argument('test', help='The path to the testing data. (hw3_test.dat)')
    parser.add_argument('-o', '--output_to_png', action='store_false',
                        help='Output image to a file. (default is output to screen)')
    parser.add_argument('--num_tree', '--nt', type=int, default=30000,
                        help='Number of trees in the random forest. (default 30000)')
    parser.add_argument('--boots_rate', '--bt', type=float, default=0.8,
                        help='Bootstrap ratio in the random forest. (default 0.8)')
    args = parser.parse_args()

    x_train, y_train, x_test, y_test = get_data(args.train, args.test)

    rf = RF(args.num_tree, args.boots_rate)
    rf.train(x_train, y_train)

    # Q14
    ein_list = []
    for tree in tqdm(rf.tree_list):
        ein_list.append((tree.test(x_train) != y_train).mean())

    # plot
    plt.hist(ein_list)
    plt.title('$t$ versus $E_{in}(g_{t})$')
    plt.xlabel('$t$')
    plt.ylabel('$E_{in}(g_{t})$')
    if args.output_to_png:
        plt.show()
    else:
        plt.savefig('./q_14')
    plt.clf()

    # Q15 & Q16
    train_pred_sum = np.zeros(x_train.shape[0])
    test_pred_sum = np.zeros(x_test.shape[0])
    ein_list = []
    eout_list = []
    for tree in tqdm(rf.tree_list):
        train_pred_sum += tree.test(x_train)
        test_pred_sum += tree.test(x_test)
        ein_list.append((np.sign(train_pred_sum) != y_train).mean())
        eout_list.append((np.sign(test_pred_sum) != y_test).mean())

    # Q15 - plot
    plt.plot(ein_list)
    plt.title('$t$ versus $E_{in}(G_{t})$')
    plt.xlabel('$t$')
    plt.ylabel('$E_{in}(G_{t})$')
    if args.output_to_png:
        plt.show()
    else:
        plt.savefig('./q_15')
    plt.clf()

    # Q16 - plot
    plt.plot(ein_list, label='$E_{in}(G_{t})$')
    plt.plot(eout_list, label='$E_{out}(G_{t})$')
    plt.title('$t$ versus $E_{in}(G_{t})$ and $E_{out}(G_{t})$')
    plt.xlabel('$t$')
    plt.ylabel('0/1 error')
    plt.legend()
    if args.output_to_png:
        plt.show()
    else:
        plt.savefig('./q_16')
    plt.clf()

if __name__ == '__main__':
    main()
