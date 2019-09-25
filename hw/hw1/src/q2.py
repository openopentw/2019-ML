""" The code for solving ml hw1 q2. """

import numpy as np
from sklearn import svm

def poly_2_kernel(x1, x2):
    return (1 + np.dot(x1, x2.T)) ** 2

def poly_2_phi(x):
    return np.array([1,
                     np.sqrt(2) * x[0],
                     np.sqrt(2) * x[1],
                     x[0] ** 2,
                     x[0] * x[1],
                     x[1] * x[0],
                     x[1] ** 2])

def main():
    x = np.array([[1, 0], [0, 1], [0, -1], [-1, 0], [0, 2], [0, -2], [-2, 0]])
    y = np.array([-1, -1, -1, 1, 1, 1, 1])
    clf = svm.SVC(kernel=poly_2_kernel, C=1e10)
    clf.fit(x, y)

    alpha = np.zeros(7)
    alpha[clf.support_] = y[clf.support_] * clf.dual_coef_[0]
    print(alpha)

    print(clf.support_)

    ######
    # q2
    ######

    # generate phi(x)
    phi_x = np.ones((x.shape[0], 7))
    phi_x[:,1] = np.sqrt(2) * x[:,0]
    phi_x[:,2] = np.sqrt(2) * x[:,1]
    phi_x[:,3] = x[:,0] ** 2
    phi_x[:,4] = 2 * x[:,0] * x[:,1]
    phi_x[:,5] = 2 * x[:,1] * x[:,0]
    phi_x[:,6] = x[:,1] ** 2

    w = ((alpha * y).reshape(alpha.size, 1) * phi_x).sum(0)
    print(w)
    b = y[1] - np.dot(w, phi_x[1])
    print(b)

    plot_x = np.linspace(-3, 11, 100)
    plot_y = w * poly_2_phi(x)

if __name__ == '__main__':
    main()
