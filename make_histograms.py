#!/usr/bin/env python3
"""
Script to make histograms from datasets in which predictions have
been recorded.
"""
# help for various options
_test_help = 'run quick job to test histogram building'

from file_names import s_file_names
from file_names import bg_file_names
import h5py
from os.path import join, isfile, isdir
from collections import defaultdict
import numpy as np
import os
from argparse import ArgumentParser
from itertools import combinations



def get_args():
    parser = ArgumentParser(description=__doc__)
    h = dict(help='default: %(default)s')
    parser.add_argument('file_dir', default='outputs', nargs='?')
    parser.add_argument('-o', '--out-file', default='hists.h5', **h)
    parser.add_argument('--test', action='store_true',help=_test_help)
    return parser.parse_args()

RANGES = [
    ('jets_pt', 100, 0, 3000),
    ('jets_mass', 100, 0, 300),
    ('baseline', 2000, -1.1, 1.1),
    ('predictions', 2000, 0, 1),
]
RANGES_DICT = {n: (b, l, h) for n, b, l, h in RANGES}

AXES = [(RANGES[m],RANGES[n]) for m,n in combinations(range(len(RANGES)),2)]

def make_histograms(h5_file):
    """
    Make 2d histograms from an h5 file
    returns: a dictionary of 2d histograms, with axes given by AXES
    """
    ds_names = ['baseline','predictions']
    arrays = { x: np.asarray(h5_file[x]) for x in ds_names }
    extra_names = ['jets_pt', 'jets_mass']
    extra = { x: np.asarray(h5_file['extra'][x]) for x in extra_names}
    arrays.update(extra)
    weights = np.asarray(h5_file['weights'])
    histograms = {}
    for axes in AXES:
        bins_list = []
        names = []
        for name, nbins, low, high in axes:
            bins = np.concatenate(
                ([-np.inf], np.linspace(low, high, nbins+1), [np.inf]) )
            bins_list.append(bins)
            names.append(name)
        values = np.stack([arrays[x] for x in names], 1)
        histogram = np.histogramdd(values, bins=bins_list, weights=weights)[0]
        histograms[tuple(names)] = histogram
    return histograms

def aggregage_histograms(file_list):
    aggregated_hists = defaultdict(lambda: 0)
    for fpath in file_list:
        with h5py.File(fpath, 'r') as h5file:
            hists = make_histograms(h5file)
        for hname, hist in hists.items():
            aggregated_hists[hname] += hist
    return aggregated_hists

def add_ds(group, hist_tup):
    """
    Adds a histogram to a group, plus some attributes which indicate
    the axes etc
    group: an h5 group
    hist: a tupple ( (axis1_name, axis2_name) , numpy_array )
    """
    ax_1, ax_2 = hist_tup[0]
    ds_name = '{}_vs_{}'.format(ax_1, ax_2)
    attrs = []
    axis_dtype = np.dtype([
        ('low', float),
        ('high', float),
    ])
    for name in (ax_1, ax_2):
        bins, low, high = RANGES_DICT[name]
        attr_ds = np.array( (low, high), dtype=axis_dtype)
        attrs.append( attr_ds)
    ds = group.create_dataset(ds_name, data=hist_tup[1])
    ds.attrs['axis_range'] = attrs
    ds.attrs['axis_name'] = [a.encode('utf8') for a in [ax_1, ax_2]]

def run():
    args = get_args()
    sig_paths = [join(args.file_dir, nm) for nm in s_file_names]
    bg_paths = [join(args.file_dir, nm) for nm in bg_file_names]
    if args.test:
        bg_paths = bg_paths[0:1]
        sig_paths = bg_paths
    with h5py.File(args.out_file, 'w') as h5_file:
        sig_hists = aggregage_histograms(sig_paths)
        sig = h5_file.create_group('signal')
        for nm_tup, hist in sig_hists.items():
            add_ds(sig, (nm_tup, hist) )
        bg_hists = aggregage_histograms(bg_paths)
        bg = h5_file.create_group('background')
        for nm_tup, hist in bg_hists.items():
            add_ds(bg, (nm_tup, hist) )

if __name__ == '__main__':
    run()
