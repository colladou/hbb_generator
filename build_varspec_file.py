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

ntup_to_edm_taggers = {
    'ip2d': 'IP2D',
    'ip3d': 'IP3D',
    '_jf': '_JetFitter',
    '_sv1': '_SV1',
}
jf_ntup_to_edm_vars = {
    '_dr': '_dRFlightDir',
    '_efc': '_energyFraction',
    '_m': '_mass',
    '_n2t': '_N2Tpair',
    '_ntrkAtVx': '_nTracksAtVtx',
    '_nvtx': '_nVTX',
    '_nvtx1t': '_nSingleTracks',
    '_sig3d': '_significance3d',
    '_deta': '_deltaeta',
    '_dphi': '_deltaphi',
}
sv_ntup_to_edm_vars = {
    '_dR': '_deltaR',
    '_efc': '_efracsvx',
    '_Lxyz': '_L3d',
    '_Lxy': '_Lxy',
    '_m': '_massvx',
    '_n2t': '_N2Tpair',
    '_ntrkv': '_NGTinSvx',
    '_normdist': '_normdist',
}

energy_corrections = {
    'mass': 1e3,
    'massvx': 1e3,
    'pt': 1e3,
}

def correct_entry(entry):
    for old, new in ntup_to_edm_taggers.items():
        entry['name'] = entry['name'].replace(old, new)
    if 'IP2D' in entry['name'] or 'IP3D' in entry['name']:
        for old, new in sv_ntup_to_edm_vars.items():
            if entry['name'].endswith(old):
                entry['name'] = entry['name'].replace(old,new)
    if 'JetFitter' in entry['name']:
        for old, new in jf_ntup_to_edm_vars.items():
            if entry['name'].endswith(old):
                entry['name'] = entry['name'].replace(old,new)
    for key in energy_corrections:
        if entry['name'].endswith(key):
            entry['offset'] *= energy_corrections[key]
            entry['scale'] *= energy_corrections[key]

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
                scale = 0.0
            entry = {
                'name': compound_name,
                'offset': float(offset),
                'scale': float(scale)
            }
            correct_entry(entry)
            out_dict['inputs'].append(entry)
    json.dump(out_dict, sys.stdout, indent=2)

if __name__ == '__main__':
    run()
