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
file_names = ['d301488_j1.h5', 'd301489_j2.h5']
load_path = '/baldig/physicsprojects/atlas/hbb/raw_data/v_3/'

predictions, weights = get_predictions_from_file_list(model, file_names, feature, load_path)
label = 1
test_y = np.ones_like(predictions) * label

predictions = predictions.reshape(predictions.shape[0],)
weights = weights.reshape(weights.shape[0],)

print(predictions.shape, test_y.shape, weights.shape)

assert predictions.shape[0] == test_y.shape[0], predictions.shape[0]

np.save("./auc/%s_test_%s_y.npy" % (feature, model_name), test_y)
np.save("./auc/%s_test_%s_predictions.npy" % (feature, model_name), predictions)
print("saved params")

print(test_y.shape, predictions.shape)

assert 1==0, "finished"

print('Calculating unweighted AUC')
print(metrics.roc_auc_score(test_y, predictions))

print('Calculating weighted AUC')
w_auc = metrics.roc_auc_score(test_y, predictions, sample_weight=weights)
print(w_auc)
np.savetxt('./auc/auc_%s.csv' % feature, [w_auc,], delimiter=',')

fpr, tpr, _ = metrics.roc_curve(test_y, predictions, sample_weight=weights)

np.save('./auc/tpr_%s.npy' % (feature), tpr)
np.save('./auc/fpr_%s.npy' % (feature), fpr)

