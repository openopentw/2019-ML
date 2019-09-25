""" This script do the classifying and plotting of question 14. """

import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

def main():
    """ Main function. """
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('train', help='The path to the training data.')
    parser.add_argument('-o', '--output', default='./q_14',
                        help='The path to the outputing picture.')
    args = parser.parse_args()

    train_path = args.train
    output_path = args.output

    # load data
    train = np.genfromtxt(train_path)
    x_train = train[:, 1:]
    y_train = train[:, 0] == 4

    # run svm and plot
    log_c_list = [-5, -3, -1, 1, 3]
    e_in_list = []
    for log_c in log_c_list:
        clf = svm.SVC(kernel='poly', degree=2, coef0=1, gamma=1, C=10 ** log_c)
        clf.fit(x_train, y_train)
        e_in = (clf.predict(x_train) != y_train).mean()
        e_in_list.append(e_in)

    plt.plot(log_c_list, e_in_list)
    plt.title('$E_{in}$ vs $log_{10}C$')
    plt.xlabel('$log_{10}C$')
    plt.ylabel('$E_{in}$')
    plt.savefig(output_path)

if __name__ == '__main__':
    main()
