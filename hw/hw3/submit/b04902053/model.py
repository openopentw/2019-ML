""" This module contains the classes of Decision Tree & Random Forest. """

import importlib

import numpy as np

if importlib.util.find_spec('tqdm'):
    from tqdm import tqdm
else:
    def tqdm(x):
        """ An empty function that has the same spec as tqdm. """
        return x

# # too slow
# def calc_gini(y):
#     """ Calculate gini index of the dataset given. """
#     gini_pos = (y == 1).mean()
#     return 2 * gini_pos * (1 - gini_pos)

class CalcGini:
    """ Calculate gini index of the dataset given. """
    def __init__(self, y):
        self.y = y
        self.pos_sum = (y == 1).sum()

        self.iter = 0
        self.iter_pos_sum = 0

    def __call__(self):
        return self.next_iter()

    def next_iter(self):
        """ Calculate gini index of the dataset given. """
        if self.y[self.iter] == 1:
            self.iter_pos_sum += 1
        self.iter += 1

        part_1_gini_pos = self.iter_pos_sum / self.iter
        part_1_gini = 2 * part_1_gini_pos * (1 - part_1_gini_pos)
        # 1 - (x^2 + (1-x)^2) = 1 - (1 - 2x + 2x^2) = 2x(1-x)

        part_2_gini_pos = (self.pos_sum - self.iter_pos_sum) / (self.y.shape[0] - self.iter)
        part_2_gini = 2 * part_2_gini_pos * (1 - part_2_gini_pos)

        gini = self.iter * part_1_gini + (self.y.shape[0] - self.iter) * part_2_gini
        return gini

class DStump():
    """ This class train & test a decision stump by gini index. """
    def __init__(self):
        self.i = None
        self.theta = None

    def __str__(self):
        return '(x[{}] > {})'.format(self.i, self.theta)

    def train(self, x, y):
        """ Train the decision stump. """
        best_gini = y.shape[0] + 5
        best_param = {'i': None, 'theta': None}
        for i in range(x.shape[1]):
            sort_idcs = x[:, i].argsort()
            y_sorted = y[sort_idcs]
            x_sorted = x[sort_idcs, i]
            calc_gini = CalcGini(y_sorted)
            for before_theta_id in range(x.shape[0] - 1):
                gini = calc_gini()
                if gini < best_gini:
                    best_gini = gini
                    theta = (x_sorted[before_theta_id] + x_sorted[before_theta_id + 1]) / 2
                    best_param.update({'i': i, 'theta': theta})
        self.i = best_param['i']
        self.theta = best_param['theta']

    def test(self, x):
        """ Test the decision stump. """
        y = np.ones(x.shape[0])
        y[x[:, self.i] < self.theta] *= -1
        return y

class TreeNode(DStump):
    """ This class describe a node in decision tree. """
    def __init__(self):
        self.neg_node = None
        self.pos_node = None
        self.is_leaf = False
        self.const = None
        super().__init__()

class DTree():
    """ This class train & test a decision tree. """
    def __init__(self):
        self.root = TreeNode()
        self.height = None

    def _str(self, node, is_from_pos_list):
        preceding_str = ''
        for i, is_from_pos in enumerate(is_from_pos_list):
            if i == len(is_from_pos_list) - 1:
                if is_from_pos:
                    preceding_str += '└──'
                else:
                    preceding_str += '┌──'
            elif is_from_pos_list[i + 1] != is_from_pos:
                preceding_str += '│  '
            else:
                preceding_str += '   '

        if node.is_leaf:
            return preceding_str + '({})\n'.format('+1' if node.const == 1 else '-1')

        ret_str = ''
        ret_str += self._str(node.neg_node, is_from_pos_list + [False])
        ret_str += preceding_str + str(node)
        ret_str += '\n'
        ret_str += self._str(node.pos_node, is_from_pos_list + [True])

        return ret_str

    def __str__(self):
        return self._str(self.root, [])

    def _train(self, x, y):
        """ Train the decision tree and return the node. """
        node = TreeNode()

        if np.unique(y).size == 1:
            node.is_leaf = True
            node.const = y[0]
            return node, 1

        node.const = 1 if (y == 1).mean() > 0.5 else -1

        node.train(x, y)
        pred = node.test(x)

        node.neg_node, neg_height = self._train(x[pred == -1], y[pred == -1])
        node.pos_node, pos_height = self._train(x[pred == 1], y[pred == 1])
        return node, max(neg_height, pos_height) + 1

    def train(self, x, y):
        """ Train the decision tree. """
        self.root, self.height = self._train(x, y)

    def _test(self, x, node, height):
        """ Test the decision tree by the given node. """
        y = np.zeros(x.shape[0])

        if node.is_leaf or height == 1:
            return y + node.const

        pred = node.test(x)
        y[pred == -1] = self._test(x[pred == -1], node.neg_node, height - 1)
        y[pred == 1] = self._test(x[pred == 1], node.pos_node, height - 1)

        return y

    def test(self, x, height=None):
        """ Test the decision tree. """
        if height is None:
            return self._test(x, self.root, self.height)
        return self._test(x, self.root, height)

class RF():
    """ This class train & test a random forest. """
    def __init__(self, num_tree, boots_rate, seed=0):
        self.tree_list = []
        for _ in range(num_tree):
            self.tree_list.append(DTree())
        self.boots_rate = boots_rate
        np.random.seed(seed)

    def train(self, x, y):
        """ Train the random forest. """
        boot_num = int(x.shape[0] * self.boots_rate)
        for tree in tqdm(self.tree_list):
            boots_choice = np.random.randint(0, x.shape[0], boot_num)
            tree.train(x[boots_choice], y[boots_choice])

    def test(self, x):
        """ Test the random forest. """
        pred = np.zeros(x.shape[0])
        for tree in self.tree_list:
            pred += tree.test(x)
        return np.sign(pred)
