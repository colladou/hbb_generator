#!/usr/bin/env python3

from argparse import ArgumentParser
import numpy as np
from matplotlib.backends.backend_pdf import FigureCanvas
from matplotlib.figure import Figure
from os.path import join, isdir
import os
from h5py import File

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('roc_datasets', nargs='+')
    parser.add_argument('-o', '--output-dir', default='plots')
    return parser.parse_args()

def run():
    args = parse_args()

    # set up the canvas
    fig = Figure( (5, 5*3/4) )
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(1,1,1)

    # now add the roc curves
    for ds_name in args.roc_datasets:
        with File(ds_name, 'r') as h5_file:
            tpr = np.asarray(h5_file['tpr'])
            fpr = np.asarray(h5_file['fpr'])
            # we only care about tpr > 0.6 or so
            valid = tpr > 0.4
        ax.plot(tpr[valid], 1/fpr[valid], '-', label=ds_name)
    ax.legend()
    ax.set_yscale('log')
    plot_name = join(args.output_dir, 'roc.pdf')
    if not isdir(args.output_dir):
        os.mkdir(args.output_dir)
    canvas.print_figure(plot_name)


if __name__ == '__main__':
    run()
