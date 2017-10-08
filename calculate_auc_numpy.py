from __future__ import print_function

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "%i" % 0
os.environ['KERAS_BACKEND'] = "tensorflow"

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

# dynamically change label based on file list
# be able to predict only on smaller number of samples of the file

def get_model_name(feature):
    if len(sys.argv) == 2:
        return str(sys.argv[1])
    else:
        return '%s_model.h5' % feature 

def get_predictions_from_file_list(model, file_names, feature, load_path='./'):
    predictions = None
    weights = None

    for file_name in file_names:
        file_name = load_path + file_name
        with h5py.File(file_name, 'r') as open_file:
            steps = math.ceil(get_num_samples(open_file)/(batch_size*1.0))
            file_weights = get_weights(open_file)
        gen = my_generator(file_name, feature, batch_size)
        file_predictions = model.predict_generator(gen, steps, verbose=0)
        if predictions is None:
            predictions = file_predictions
            weights = file_weights
        else:
            predictions = np.vstack((predictions, file_predictions))
            weights = np.hstack((weights, file_weights))
        assert predictions.shape[0] == weights.shape[0], weights.shape[0]
    return [predictions, weights]


feature = 'hl_tracks'

model_name = get_model_name(feature)
model = keras.models.load_model("./models/" + model_name)

batch_size = 100

bg_file_names = ['d301488_j1.h5', 'd301489_j2.h5', 'd301490_j3.h5', 'd301491_j4.h5',
               'd301492_j5.h5', 'd301493_j6.h5', 'd301494_j7.h5', 'd301495_j8.h5',
               'd301496_j9.h5', 'd301497_j10.h5', 'd301498_j11.h5', 'd301499_j12.h5',
               'd301500_j13.h5', 'd301501_j14.h5', 'd301502_j15.h5', 'd301503_j16.h5',
               'd301504_j17.h5', 'd301505_j18.h5', 'd301506_j19.h5', 'd301507_j20.h5']
s_file_names = [#'d361021_j27.h5', 
                'd361022_j28.h5', 'd361023_j29.h5','d361024_j30.h5', 
                'd361025_j31.h5', 'd361026_j32.h5', 'd361027_j33.h5', 'd361028_j34.h5',
                'd361029_j35.h5', 'd361030_j36.h5', 'd361031_j37.h5', 'd361032_j38.h5']

file_names = ['d301488_j1.h5', 'd301489_j2.h5']
load_path = '/baldig/physicsprojects/atlas/hbb/raw_data/v_3/'

s_predictions, s_weights = get_predictions_from_file_list(model, s_file_names, feature, load_path)
s_test_y = np.ones_like(predictions)

bg_predictions, bg_weights = get_predictions_from_file_list(model, bg_file_names, feature, load_path)
bg_test_y = np.ones_like(predictions) * 0

print(s_predictions.shape, s_test_y.shape, s_weights.shape)
print(bg_predictions.shape, bg_test_y.shape, bg_weights.shape)

predictions = np.hstack((s_predictions, bg_predictions))
weights = np.hstack((s_weights, bg_weights))
test_y = np.hstack((s_test_y, bg_test_y))

assert predictions.shape[0] == s_predictions.shape[0] + bg_predictions.shape[0]
assert weights.shape[0] == s_weights.shape[0] + bg_weights.shape[0]
assert test_y.shape[0] == s_test_y.shape[0] + bg_test_y.shape[0]

predictions = predictions.reshape(predictions.shape[0],)
weights = weights.reshape(weights.shape[0],)

print(predictions.shape, test_y.shape, weights.shape)

assert predictions.shape[0] == test_y.shape[0], predictions.shape[0]

np.save("./auc/%s_test_%s_y.npy" % (feature, model_name), test_y)
np.save("./auc/%s_test_%s_predictions.npy" % (feature, model_name), predictions)
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

