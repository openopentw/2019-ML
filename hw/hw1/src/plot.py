""" This script do the plotting of question 13. """

import matplotlib.pyplot as plt
import numpy as np

def main():
    """ Main function. """
    log_c_list = [-5, -3, -1, 1, 3]
    w_list = np.array([
        [-1.15282419e-05, -1.33000000e-06],
        [-0.00115282, -0.000133],
        [-0.00251775, -0.00017466],
        [-0.00169859, -0.00016728],
        [-0.0262514, -0.00075591],
    ])
    w_norm_list = np.sqrt((w_list ** 2).sum(1))

    plt.plot(log_c_list, w_norm_list)
    plt.title('$||w||$ vs $log_{10}C$')
    plt.xlabel('$log_{10}C$')
    plt.ylabel('$||w||$')
    plt.savefig('./q_13')

if __name__ == '__main__':
    main()
