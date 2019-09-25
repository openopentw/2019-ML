import numpy as np
from itertools import product

train_data = np.loadtxt('./data/hw2_adaboost_train.dat')
x_train, y_train = train_data[:,:-1], train_data[:,-1]

test_data = np.loadtxt('./data/hw2_adaboost_test.dat')
x_test, y_test = test_data[:,:-1], test_data[:,-1]

sorted_idx = [np.argsort(x_train[:, i]) for i in range(x_train.shape[1])]
all_decision_stumps = []
for s, i, n in product([1, -1], range(x_train.shape[1]), range(x_train.shape[0])):
    if n == 0:
        theta = -1e10
    else:
        theta = (x_train[sorted_idx[i][n], i] + x_train[sorted_idx[i][n - 1], i]) / 2
    all_decision_stumps.append((s, i, theta))

u = np.ones(x_train.shape[0]) / x_train.shape[0]
g, alpha, ein_G, eout_G, ein_g, eout_g, U = [], [], [], [], [], [], [u.sum()]
for t in range(300):
    best = {'e': 1e10}
    for s, i, theta  in all_decision_stumps:
        predict = s * np.sign(x_train[:, i] - theta)
        e = (u * (predict != y_train)).sum()
        if e < best['e']:
            best = {'e': e, 'param': (s, i, theta)}
    
    s, i, theta = best['param']
    predict = s * np.sign(x_train[:, i] - theta)
    epsilon = (u * (predict != y_train)).sum() / u.sum()
    delta = np.sqrt((1 - epsilon) / epsilon)
    u[np.where(predict == y_train)] /= delta
    u[np.where(predict != y_train)] *= delta

    train_predict = np.sign(sum([(s * np.sign(x_train[:, i] - theta)) for (s, i, theta), a in zip(g, alpha)]))
    test_predict = np.sign(sum([(s * np.sign(x_test[:, i] - theta)) for (s, i, theta), a in zip(g, alpha)]))

    alpha.append(np.log(delta))
    ein_g.append((np.sign(s * np.sign(x_train[:, i] - theta)) != y_train).mean())
    eout_g.append((np.sign(s * np.sign(x_test[:, i] - theta)) != y_test).mean())
    ein_G.append((train_predict != y_train).mean())
    eout_G.append((test_predict != y_test).mean())
    U.append(u.sum())
    g.append((s, i, theta))

import matplotlib.pyplot as plt

# p11~12
plt.plot(ein_g, label='Ein(g)')
plt.ylim((-0.1, 1))
plt.xlabel('t')
plt.legend()
plt.show()
print(ein_g[0], alpha[0])

# p13
plt.plot(ein_G, label='Ein(G)')
plt.ylim((-0.1, 1))
plt.xlabel('t')
plt.legend()
plt.show()
print(ein_G[-1])

# p14
plt.plot(U, label='U')
plt.ylim((-0.1, 1))
plt.xlabel('t')
plt.legend()
plt.show()
print(U[1], U[-1])

# p15
plt.plot(eout_g, label='Eout(g)')
plt.ylim((-0.1, 1))
plt.xlabel('t')
plt.legend()
plt.show()
print(eout_g[0])

# p16
plt.plot(eout_G, label='Eout(G)')
plt.ylim((-0.1, 1))
plt.xlabel('t')
plt.legend()
plt.show()
print(eout_G[-1])
