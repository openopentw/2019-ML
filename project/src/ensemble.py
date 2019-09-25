""" This script simply ensemble the existing csv files by hand-crafted weights. """

import argparse
import json

import numpy as np

PRED_SHAPE = (2500, 3)

def main():
    """ Main function. """
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='The path to the config file.')
    args = parser.parse_args()

    config = json.load(open(args.config))

    # weighted sum
    y_preds = np.zeros((len(config['file_list']), *PRED_SHAPE))
    weights = np.zeros(len(config['file_list']))
    for i, file_ in enumerate(config['file_list']):
        y_preds[i] = np.genfromtxt(file_['filename'], delimiter=',')
        weights[i] = file_['weight']

    # do things about combination & adjust
    combination = config.get('combination', None)
    adjust = config.get('adjust', [])

    # combination
    if combination == 'norm':
        y_pred = np.average(y_preds, axis=0, weights=weights)
    elif combination == 'min':
        y_pred = y_preds.min(axis=0)
    else: # sum
        pred_size = PRED_SHAPE[0] * PRED_SHAPE[1]
        y_pred = np.dot(weights, y_preds.reshape(-1, pred_size)).reshape(PRED_SHAPE)

    # adjust
    if 'scaling' in adjust:
        y_pred *= adjust['scaling']
    if 'exp' in adjust:
        y_pred **= adjust['exp']
    if 'scaling_decay' in adjust:
        y_pred_max = np.sign(y_pred).max(axis=0)
        y_pred_min = np.sign(y_pred).min(axis=0)
        decay_rate = adjust['scaling_decay'] * (1 / (y_pred_max - y_pred_min))
        y_pred *= decay_rate ** (y_pred_max - np.sign(y_pred))
    if 'scaling_exp_decay' in adjust:
        y_pred_max = np.sign(y_pred).max(axis=0)
        y_pred_min = np.sign(y_pred).min(axis=0)
        decay_rate = adjust['scaling_exp_decay'] ** (1 / (y_pred_max - y_pred_min))
        y_pred *= decay_rate ** (y_pred_max - np.sign(y_pred))
    if 'scaling_relu' in adjust:
        thres = np.percentile(np.sign(y_pred), adjust['scaling_relu'], axis=0)
        y_pred[np.sign(y_pred) < thres] = 0

    np.savetxt(config['output'], y_pred, delimiter=',')

if __name__ == '__main__':
    main()
