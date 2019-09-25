""" This script do the classifying and plotting of question 13. """

import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

def main():
    """ Main function. """
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('train', help='The path to the training data.')
    parser.add_argument('-o', '--output', default='./q_13',
                        help='The path to the outputing picture.')
    args = parser.parse_args()

    train_path = args.train
    output_path = args.output

    # load data
    train = np.genfromtxt(train_path)
    x_train = train[:, 1:]
    y_train = train[:, 0] == 2

    # run svm and plot
    log_c_list = [-5, -3, -1, 1, 3]
    w_norm_list = []
    for log_c in log_c_list:
        clf = svm.SVC(kernel='linear', C=10 ** log_c)
        clf.fit(x_train, y_train)
        weight = clf.coef_[0]
        w_norm_list.append(np.sqrt((weight ** 2).sum()))

    plt.plot(log_c_list, w_norm_list)
    plt.title('$||w||$ vs $log_{10}C$')
    plt.xlabel('$log_{10}C$')
    plt.ylabel('$||w||$')
    plt.savefig(output_path)

if __name__ == '__main__':
    main()
