""" This script do the classifying and plotting of question 15. """

import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

def main():
    """ Main function. """
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('train', help='The path to the training data.')
    parser.add_argument('-o', '--output', default='./q_15',
                        help='The path to the outputing picture.')
    args = parser.parse_args()

    train_path = args.train
    output_path = args.output

    # load data
    train = np.genfromtxt(train_path)
    x_train = train[:, 1:]
    y_train = train[:, 0] == 0

    # run svm and plot
    log_c_list = [-2, -1, 0, 1, 2]
    dis_list = []
    for log_c in log_c_list:
        clf = svm.SVC(kernel='rbf', gamma=80, C=10 ** log_c)
        clf.fit(x_train, y_train)
        weight = clf.dual_coef_[0].dot(clf.support_vectors_)
        dis_list.append(1 / np.sqrt((weight ** 2).sum()))

    plt.plot(log_c_list, dis_list)
    plt.title('Distance vs $log_{10}C$')
    plt.xlabel('$log_{10}C$')
    plt.ylabel('Distance')
    plt.savefig(output_path)

if __name__ == '__main__':
    main()
