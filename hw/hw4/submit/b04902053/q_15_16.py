""" This script do the question 15 ~ 16 of ml-hw4. """

import argparse

import matplotlib.pyplot as plt
import numpy as np

from k_means import KMeans

def get_data(data_path):
    """ Get the data. """
    data = np.genfromtxt(data_path)
    return data

def main():
    """ Main function. """
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='hw4_nolabel_train.dat')
    parser.add_argument('-t', '--trial', type=int, default=500,
                        help='experiment times (default = 500)')
    parser.add_argument('-o', '--output_to_png', default=False, action='store_true',
                        help='Output image to files. (default is display on screen)')
    args = parser.parse_args()

    # get data
    data = get_data(args.data)

    # fit
    k_list = [2, 4, 6, 8, 10]
    avg_list = []
    var_list = []
    for k in k_list:
        err_list = []
        k_means = KMeans(k)
        for _ in range(args.trial):
            k_means.fit(data)
            err_list.append(k_means.calc_err())
        err_list = np.array(err_list)
        avg_list.append(err_list.mean())
        var_list.append(err_list.var())

    # plot
    plt.scatter(k_list, avg_list)
    plt.title('Average of $E_{in}$ vs. $k$')
    plt.xlabel('$k$')
    plt.ylabel('Average of $E_{in}$')
    if args.output_to_png:
        plt.savefig('q_15')
    else:
        plt.show()
    plt.clf()

    # plot
    plt.scatter(k_list, var_list)
    plt.title('Variance of $E_{in}$ vs. $k$')
    plt.xlabel('$k$')
    plt.ylabel('Variance of $E_{in}$')
    if args.output_to_png:
        plt.savefig('q_16')
    else:
        plt.show()
    plt.clf()

if __name__ == '__main__':
    main()
