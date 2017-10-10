from __future__ import print_function

import os
# You can set these as environment variables, i.e.
# > CUDA_VISIBLE_DEVICES=0 KERAS_BACKEND=tensorflow python calculate_auc_numpy.py
# os.environ['CUDA_VISIBLE_DEVICES'] = "%i" % 0
# os.environ['KERAS_BACKEND'] = "tensorflow"

import keras
from keras.models import model_from_json
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers.core import Activation, Dense, Dropout
from sklearn import metrics
import sys
import numpy as np
import h5py
import math
from generator import my_generator
from generator import get_num_samples, get_weights
from os.path import isdir, join
import argparse

# be able to predict only on smaller number of samples of the file

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices={'julian','dan','local'},
                        default='local', nargs='?')
    return parser.parse_args()

def get_model_name(feature):
    return '%s_model.h5' % feature

def get_predictions_from_file_list(model, file_names, feature, load_path='./', sub_sample=100):
    predictions = None
    weights = None
    for file_name in file_names:
        print("Making predictions for: ", file_name)
        file_name = load_path + file_name
        with h5py.File(file_name, 'r') as open_file:
            num_samples = int(get_num_samples(open_file) * (sub_sample/100.0))
            steps = math.ceil(num_samples/(batch_size*1.0))
            file_weights = get_weights(open_file)
            file_weights = file_weights[0:steps*batch_size]
        gen = my_generator(file_name, feature, batch_size)
        print("running generator in {} steps".format(steps))
        file_predictions = model.predict_generator(gen, steps)
        if predictions is None:
            predictions = file_predictions
            weights = file_weights
        else:
            predictions = np.vstack((predictions, file_predictions))
            weights = np.hstack((weights, file_weights))
        assert predictions.shape[0] == weights.shape[0], [predictions.shape[0], weights.shape[0]]
    return [predictions, weights]

args = get_args()

feature = 'hl_tracks'

model_name = get_model_name(feature)
model = keras.models.load_model("./models/" + model_name)

batch_size = 5000

s_file_names = ['d301488_j1.h5', 'd301489_j2.h5', 'd301490_j3.h5', 'd301491_j4.h5',
                 'd301492_j5.h5', 'd301493_j6.h5', 'd301494_j7.h5', 'd301495_j8.h5',
                 'd301496_j9.h5', 'd301497_j10.h5', 'd301498_j11.h5', 'd301499_j12.h5',
                 'd301500_j13.h5', 'd301501_j14.h5', 'd301502_j15.h5', 'd301503_j16.h5',
                 'd301504_j17.h5', 'd301505_j18.h5', 'd301506_j19.h5', 'd301507_j20.h5']
bg_file_names = [#'d361021_j27.h5',
                'd361022_j28.h5', 'd361023_j29.h5','d361024_j30.h5',
                'd361025_j31.h5', 'd361026_j32.h5', 'd361027_j33.h5', 'd361028_j34.h5',
                'd361029_j35.h5', 'd361030_j36.h5', 'd361031_j37.h5', 'd361032_j38.h5']

if args.mode == 'local':
    load_path = 'data/'
    s_file_names = ['d301488_j1.h5']
    bg_file_names = ['d361022_j28.h5']

elif args.mode == 'julian':
    load_path = '/baldig/physicsprojects/atlas/hbb/raw_data/v_3/'

elif args.mode == 'dan':
    load_path = '/home/dguest/bookmarks/hbb/hbb/v3/data/'

s_predictions, s_weights = get_predictions_from_file_list(model, s_file_names, feature, load_path)
s_test_y = np.ones_like(s_predictions) * 0

bg_predictions, bg_weights = get_predictions_from_file_list(model, bg_file_names, feature, load_path, sub_sample=10)
bg_test_y = np.ones_like(bg_predictions) * 1

print(s_predictions.shape, s_test_y.shape, s_weights.shape)
print(bg_predictions.shape, bg_test_y.shape, bg_weights.shape)

predictions = np.vstack((s_predictions, bg_predictions))
weights = np.hstack((s_weights, bg_weights))
test_y = np.vstack((s_test_y, bg_test_y))

assert predictions.shape[0] == s_predictions.shape[0] + bg_predictions.shape[0]
assert weights.shape[0] == s_weights.shape[0] + bg_weights.shape[0]
assert test_y.shape[0] == s_test_y.shape[0] + bg_test_y.shape[0]

predictions = predictions.reshape(predictions.shape[0],)
weights = weights.reshape(weights.shape[0],)

print(predictions.shape, test_y.shape, weights.shape)

assert predictions.shape[0] == test_y.shape[0], predictions.shape[0]

auc_dir = 'auc'
if not isdir(auc_dir):
    os.mkdir(auc_dir)
np.save(join(auc_dir, "%s_test_%s_y.npy" % (feature, model_name)), test_y)
np.save(join(auc_dir,"%s_test_%s_predictions.npy" % (feature, model_name)), predictions)
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

