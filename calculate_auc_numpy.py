#!/usr/bin/env python3
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
from os.path import join, isdir, basename, dirname, abspath
import inspect

# the file lists are in a common file now
from file_names import s_file_names, bg_file_names


# be able to predict only on smaller number of samples of the file

def ttyprint(*args, **kwargs):
    if sys.stdout.isatty():
        print(*args, **kwargs)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices={'julian','dan','local'},
                        default='local', nargs='?')
    parser.add_argument('-f', '--files', nargs='*')
    parser.add_argument('-n', '--file-number', type=int,
                        help="only run on one file, given by this index")
    parser.add_argument('--get-n-files', action='store_true',
                        help='print number of files this will use and exit')
    return parser.parse_args()

def get_model_name(feature):
    return '%s_model.h5' % feature

def get_predictions_from_file_list(model, file_names, feature,
                                   load_path='./',
                                   sub_sample=100,
                                   out_file_path='outputs',
                                   mean_and_std_path='models'):
    try:
        os.mkdir(out_file_path)
    except FileExistsError:
        print('{} already exists, not remaking'.format(out_file_path))
    predictions = None
    weights = None
    for file_name in file_names:
        print("Counting samples in for {}... ".format(file_name), end='')
        file_name = join(load_path,file_name)
        with h5py.File(file_name, 'r') as open_file:
            num_samples = int(get_num_samples(open_file) * (sub_sample/100.0))
            steps = math.ceil(num_samples/(batch_size*1.0))
            file_weights = get_weights(open_file)
            file_weights = file_weights[0:steps*batch_size]
        print("{} found, processing".format(num_samples))

        out_path = join(out_file_path, basename(file_name))
        with h5py.File(out_path, 'w') as out_file:
            predictions = out_file.create_dataset(
                'predictions', (0,), maxshape=(None,),
                chunks=(batch_size,), dtype=float)
            weights = out_file.create_dataset(
                'weights', (0,), maxshape=(None,),
                chunks=(batch_size,), dtype=float)
            offset = 0
            for batch in my_generator(file_name, feature, batch_size,
                                      max_samples=num_samples,
                                      mean_and_std_path=mean_and_std_path):
                new_offset = offset + batch.shape[0]
                # build a slice object
                batch_slice = slice(offset, new_offset)
                predictions.resize(new_offset, 0)
                batch_prediction = model.predict(batch)
                predictions[batch_slice] = batch_prediction[:,0]
                weights.resize(new_offset, 0)
                weights[batch_slice] = file_weights[batch_slice]
                offset = new_offset
                ttyprint('.',end='', flush=True)
            ttyprint()

args = get_args()

feature = 'hl_tracks'

model_name = get_model_name(feature)
# some things have to make reference to this package
this_dir = abspath(dirname(inspect.getfile(inspect.currentframe())))
mean_and_std_path = join(this_dir, "models")
model = keras.models.load_model(join(this_dir, "models", model_name))

batch_size = 1000


if args.mode == 'local':
    load_path = 'data/'
    s_file_names = s_file_names[0:1]
    bg_file_names = bg_file_names[0:1]

elif args.mode == 'julian':
    load_path = '/baldig/physicsprojects/atlas/hbb/raw_data/v_3/'

elif args.mode == 'dan':
    load_path = '/home/dguest/bookmarks/hbb/hbb/v3/data/'

if args.files:
    load_path = ''
    s_file_names = args.files
    bg_file_names = []

if args.file_number is not None:
    # set the file lists such that only one file is used
    fid = args.file_number
    s_file_names = s_file_names[fid:fid+1]
    fid -= len(s_file_names)
    bg_file_names = bg_file_names[fid:fid+1]

if args.get_n_files:
    n_files_total = len(s_file_names) + len(bg_file_names)
    print('running on {} files'.format(n_files_total))
    exit(1)

get_predictions_from_file_list(model, s_file_names, feature, load_path,
                               mean_and_std_path=mean_and_std_path)
get_predictions_from_file_list(model, bg_file_names, feature, load_path,
                               mean_and_std_path=mean_and_std_path,
                               sub_sample=10)

