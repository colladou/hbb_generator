#!/usr/bin/env python3

import h5py
import argparse
import numpy as np
from itertools import product
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('-o', '--output-dir', default='plots')
    return parser.parse_args()

def compute_roc(h5file, discrim, variable, window):
    signal = h5file['signal']['{}_vs_{}'.format(variable, discrim)]
    bg = h5file['background']['{}_vs_{}'.format(variable, discrim)]
    n_bins = signal.shape[0] - 2
    axis_range = np.linspace(*bg.attrs['axis_range'][0], n_bins)
    low, high = window
    valid = (axis_range < high) & (axis_range > low)
    sig_squash = signal[valid, :].sum(axis=1)
    bg_squash = bg[valid, :].sum(axis=1)
    tpr = sig_squash / sig_squash.sum()
    fpr = bg_squash / bg_squash.sum()
    return tpr, fpr

WINDOWS = [
    ('jets_pt', 250, 400),
    ('jets_pt', 400, 800),
    ('jets_pt', 800, 1000),
    ('jets_pt', 1000, 1500),
    ('jets_pt', 1500, 2000),
]
DISCRIM = ['baseline', 'predictions']

def run():
    args = get_args()
    rates = {}
    with h5py.File(args.input_file, 'r') as h5_file:
        for discrim, window in product(DISCRIM, WINDOWS):
            var, *bounds = window
            tpr, fpr = compute_roc(h5_file, discrim, var, bounds)
            title = r'{}: {} -- {}'.format(*window)


if __name__ == '__main__':
    run()
