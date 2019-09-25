""" This script do the classifying and plotting of question 16. """

import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

def main():
    """ Main function. """
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('train', help='The path to the training data.')
    parser.add_argument('-o', '--output', default='./q_16',
                        help='The path to the outputing picture.')
    args = parser.parse_args()

    train_path = args.train
    output_path = args.output

    # load data
    train = np.genfromtxt(train_path)
    x_train = train[:, 1:]
    y_train = train[:, 0] == 0

    best_gamma_cnt = [0] * 5
    for _ in range(100):
        # split train and vali
        indices = np.arange(x_train.shape[0])
        np.random.shuffle(indices)
        x_vali_train = x_train[indices][1000:]
        y_vali_train = y_train[indices][1000:]
        x_vali = x_train[indices][:1000]
        y_vali = y_train[indices][:1000]

        # run svm and collect best gamma
        log_gamma_list = [-2, -1, 0, 1, 2]
        e_val_list = []
        for log_gamma in log_gamma_list:
            clf = svm.SVC(kernel='rbf', gamma=10 ** log_gamma, C=0.1)
            clf.fit(x_vali_train, y_vali_train)
            e_val = (clf.predict(x_vali) != y_vali).mean()
            e_val_list.append(e_val)
        best_gamma_idx = np.array(e_val_list).argmin()
        best_gamma_cnt[best_gamma_idx] += 1

    plt.bar(log_gamma_list, best_gamma_cnt)
    plt.title('Best $log_{10}\\gamma$ Counts')
    plt.savefig(output_path)

if __name__ == '__main__':
    main()
