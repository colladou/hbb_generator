#!/usr/bin/env python3
from file_names import s_file_names
from file_names import bg_file_names
import h5py
from os.path import join, isfile, isdir
import numpy as np
from sklearn import metrics
import os
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser()
    h = dict(help='default: %(default)s')
    parser.add_argument('file_dir', default='outputs', nargs='?')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('-d', '--discriminant', default='predictions', **h)
    return parser.parse_args()

def get_predictions_and_weights(name_list, file_directory='outputs',
                                discriminant='predictions'):
    pred_list = []
    weights_list = []
    for fname in name_list:
        fpath = join('outputs', fname)
        if not isfile(fpath):
            print('{} not found, skipping...'.format(fname))
            continue
        with h5py.File(fpath, 'r') as h5file:
            pred_list.append(np.asarray(h5file[discriminant]))
            weights_list.append(np.asarray(h5file['weights']))
    return np.hstack(pred_list), np.hstack(weights_list)

args = get_args()
if args.test:
    # we only save one file for tests, so we use it for both signal and bg
    s_file_names = bg_file_names

s_predictions, s_weights = get_predictions_and_weights(
    s_file_names, args.file_dir, args.discriminant)
bg_predictions, bg_weights = get_predictions_and_weights(
    bg_file_names, args.file_dir, args.discriminant)

s_test_y = np.ones_like(s_predictions) * 1
bg_test_y = np.ones_like(bg_predictions) * 0

print(s_predictions.shape, s_test_y.shape, s_weights.shape)
print(bg_predictions.shape, bg_test_y.shape, bg_weights.shape)

predictions = np.concatenate((s_predictions, bg_predictions))
weights = np.concatenate((s_weights, bg_weights))
test_y = np.concatenate((s_test_y, bg_test_y))

assert predictions.shape[0] == s_predictions.shape[0] + bg_predictions.shape[0]
assert weights.shape[0] == s_weights.shape[0] + bg_weights.shape[0]
assert test_y.shape[0] == s_test_y.shape[0] + bg_test_y.shape[0]

predictions = predictions.reshape(predictions.shape[0],)
weights = weights.reshape(weights.shape[0],)

print(predictions.shape, test_y.shape, weights.shape)

assert predictions.shape[0] == test_y.shape[0], predictions.shape[0]

auc_dir = 'auc'
feature = 'hl_tracks'
if not isdir(auc_dir):
    os.mkdir(auc_dir)
np.save(join(auc_dir, "%s_test_y.npy" % feature), test_y)
np.save(join(auc_dir,"%s_test_predictions.npy" % feature), predictions)
print("saved params")

print(test_y.shape, predictions.shape)

print('Calculating unweighted AUC')
print(metrics.roc_auc_score(test_y, predictions))

print('Calculating weighted AUC')
w_auc = metrics.roc_auc_score(test_y, predictions, sample_weight=weights)
print(w_auc)
np.savetxt('./auc/auc_%s.csv' % feature, [w_auc,], delimiter=',')

fpr, tpr, _ = metrics.roc_curve(test_y, predictions, sample_weight=weights)

np.save('./auc/tpr_%s.npy' % (feature), tpr)
np.save('./auc/fpr_%s.npy' % (feature), fpr)

with h5py.File(join('auc', 'roc_{}.h5'.format(feature)), 'w') as roc_file:
    roc_file.create_dataset('tpr', data=tpr)
    roc_file.create_dataset('fpr', data=fpr)
