#!/usr/bin/env python3

import h5py
import argparse
from os.path import join
from os import mkdir
import numpy as np
from itertools import product
from argparse import ArgumentParser
from mpl import Canvas

def get_args():
    parser = ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('-o', '--output-dir', default='plots')
    parser.add_argument('-e', '--ext', default='.pdf',
                        choices={'.pdf', '.png'})
    return parser.parse_args()

def compute_roc(h5file, discrim, variable, window):
    signal = h5file['signal']['{}_vs_{}'.format(variable, discrim)]
    bg = h5file['background']['{}_vs_{}'.format(variable, discrim)]
    n_bins = signal.shape[0] - 2
    axis_range = np.linspace(*bg.attrs['axis_range'][0], n_bins+1)
    print(axis_range.shape)
    low, high = window
    valid = (axis_range[1:] < high) & (axis_range[:-1] > low)
    print(valid)
    sig_squash = signal[valid, :].sum(axis=0)
    bg_squash = bg[valid, :].sum(axis=0)
    tpr = sig_squash[::-1].cumsum() / sig_squash.sum()
    fpr = bg_squash[::-1].cumsum() / bg_squash.sum()
    return tpr, fpr

def draw_roc(can, tpr, fpr, title):
    valid = tpr > 0.2
    rej = 1/fpr[valid]
    can.ax.plot(tpr[valid], rej, label=title)

WINDOWS = [
    ('jets_pt', 250, 400),
    ('jets_pt', 400, 800),
    ('jets_pt', 800, 1000),
    ('jets_pt', 1000, 1500),
    ('jets_pt', 1500, 2000),
    ('jets_pt', 250, 2000),
]
DISCRIM = ['baseline', 'predictions']

def run():
    args = get_args()
    ext = args.ext.lstrip('.')
    rates = {}
    with h5py.File(args.input_file, 'r') as h5_file:
        for window in WINDOWS:
            path = join(args.output_dir, '{}-{}-{}.{}'.format(*window, ext))
            with Canvas(path) as can:
                for discrim in DISCRIM:
                    var, *bounds = window
                    tpr, fpr = compute_roc(h5_file, discrim, var, bounds)
                    title = r'{}: {} -- {} GeV'.format(discrim,*window[1:])
                    draw_roc(can, tpr, fpr, title)
                can.ax.set_ylabel('Light Jet Rejection')
                can.ax.set_xlabel(r'$H$ Jet Efficiency')
                can.ax.legend()
                can.ax.set_yscale('log')

if __name__ == '__main__':
    run()
