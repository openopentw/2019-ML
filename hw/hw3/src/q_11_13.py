""" This script do the experiments of questions 11 ~ 13. """

import argparse

import matplotlib.pyplot as plt
import numpy as np

from model import DTree

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
    args = parser.parse_args()

    x_train, y_train, x_test, y_test = get_data(args.train, args.test)

    dtree = DTree()
    dtree.train(x_train, y_train)

    # Q11
    print('Tree:')
    print(dtree)

    # Q12
    print('E_in =', (dtree.test(x_train) != y_train).mean())
    print('E_out =', (dtree.test(x_test) != y_test).mean())

    # Q13
    ein_list = []
    eout_list = []
    for height in range(1, dtree.height):
        ein = (y_train != dtree.test(x_train, height)).mean()
        eout = (y_test != dtree.test(x_test, height)).mean()
        ein_list.append(ein)
        eout_list.append(eout)

    # plotting
    plt.plot(ein_list, label='$E_{in}(g_{h})$')
    plt.plot(eout_list, label='$E_{out}(g_{h})$')
    plt.title('$h$ versus $E_{in}(g_{h})$ and $E_{out}(g_{h})$')
    plt.xlabel('$h$')
    plt.ylabel('0/1 error')
    plt.xticks(range(dtree.height - 1), range(1, dtree.height))
    plt.legend()
    if args.output_to_png:
        plt.show()
    else:
        plt.savefig('./q_13')
    plt.clf()

if __name__ == '__main__':
    main()
