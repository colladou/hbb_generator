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
import sys
import numpy as np
import h5py
import math
from generator import my_generator
from generator import get_num_samples, get_weights
import argparse
from os.path import join, isdir, basename

# the file lists are in a common file now
from file_names import s_file_names, bg_file_names

# be able to predict only on smaller number of samples of the file

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices={'julian','dan','local'},
                        default='local', nargs='?')
    return parser.parse_args()

def get_model_name(feature):
    return '%s_model.h5' % feature

def get_predictions_from_file_list(model, file_names, feature, load_path='./', sub_sample=100, out_file_path='outputs'):
    if not isdir(out_file_path):
        os.mkdir(out_file_path)
    predictions = None
    weights = None
    for file_name in file_names:
        print("Making predictions for: ", file_name)
        file_name = join(load_path,file_name)
        with h5py.File(file_name, 'r') as open_file:
            num_samples = int(get_num_samples(open_file) * (sub_sample/100.0))
            steps = math.ceil(num_samples/(batch_size*1.0))
            file_weights = get_weights(open_file)
            file_weights = file_weights[0:steps*batch_size]

        out_path = join(out_file_path, basename(file_name))
        with h5py.File(out_path, 'w') as out_file:
            predictions = out_file.create_dataset(
                'predictions', (0,), maxshape=(None,),
                chunks=(batch_size,), dtype=float)
            weights = out_file.create_dataset(
                'weights', (0,), maxshape=(None,),
                chunks=(batch_size,), dtype=float)
            offset = 0
            for batch in my_generator(file_name, feature, batch_size):
                new_offset = offset + batch.shape[0]
                # build a slice object
                batch_slice = slice(offset, new_offset)
                predictions.resize(new_offset, 0)
                batch_prediction = model.predict(batch)
                predictions[batch_slice] = batch_prediction[0]
                weights.resize(new_offset, 0)
                weights[batch_slice] = file_weights[batch_slice]
                offset = new_offset

args = get_args()

feature = 'hl_tracks'

model_name = get_model_name(feature)
model = keras.models.load_model("./models/" + model_name)

batch_size = 1000


if args.mode == 'local':
    load_path = 'data/'
    s_file_names = s_file_names[0:1]
    bg_file_names = bg_file_names[0:1]

elif args.mode == 'julian':
    load_path = '/baldig/physicsprojects/atlas/hbb/raw_data/v_3/'

elif args.mode == 'dan':
    load_path = '/home/dguest/bookmarks/hbb/hbb/v3/data/'

get_predictions_from_file_list(model, s_file_names, feature, load_path)
get_predictions_from_file_list(model, bg_file_names, feature, load_path, sub_sample=10)

