#!/usr/bin/env python3

"""
Build a file that specifies the variables used
"""

from argparse import ArgumentParser
import sys
import numpy as np
from math import isinf
import json

from generator import get_variable_names

def get_args():
    d = dict(help='Default: %(default)s', nargs='?')
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('mean_vector')
    parser.add_argument('std_vector')
    parser.add_argument('set_name', default='hl_tracks', **d)
    return parser.parse_args()

def run():
    args = get_args()
    offset_and_scale = zip(-np.load(args.mean_vector),
                           1/np.load(args.std_vector))
    norm_iterator = iter(offset_and_scale)
    var_names, merge_order = get_variable_names(args.set_name)
    out_dict = {
        'inputs': [], 'class_labels': ['HbbScore']
    }
    for category in merge_order:
        variables = var_names[category]
        for variable in variables:
            compound_name = f'{category}_{variable}'
            offset, scale = next(norm_iterator)
            if isinf(scale):
                assert offset == 0.0
                scale = 1.0
            entry = {
                'name': compound_name,
                'offset': float(offset),
                'scale': float(scale)
            }
            out_dict['inputs'].append(entry)
    json.dump(out_dict, sys.stdout, indent=2)

if __name__ == '__main__':
    run()
