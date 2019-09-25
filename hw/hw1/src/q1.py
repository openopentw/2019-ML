""" The code for solving ml hw1 q1. """

import numpy as np

def main():
    """ Main function. """
    xy_list = np.array([[1, 0, -1],
                        [0, 1, -1],
                        [0, -1, -1],
                        [-1, 0, 1],
                        [0, 2, 1],
                        [0, -2, 1],
                        [-2, 0, 1]])
    x_list = xy_list[:,:2]
    y_list = xy_list[:,2]

if __name__ == '__main__':
    main()
