#!/usr/bin/env python

import argparse
import os
import sys
import astropy.io.fits as pyfits
import json

import numpy
import scipy
import sklearn
import matplotlib.pyplot as plt
import glob
import pandas
import itertools
import logging



class BenchSpek(object):

    config = None
    raw_directory = "."
    master_bias = None
    master_flat = None
    master_comp = None

    def __init__(self, json_file, raw_dir=None):
        self.logger = logging.getLogger('BenchSpek')

        self.json_file = json_file
        self.read_config()
        if (raw_dir is not None and os.path.isdir(raw_dir)):
            self.raw_dir = raw_dir = raw_dir

    def read_config(self):
        self.logger.info(self.json_file)
        with open(self.json_file, "r") as f:
            self.config = json.load(f)

    def basic_reduction(self, filelist, bias=None, flat=None, op=numpy.mean):
        _list = []
        for fn in filelist:
            _fn = os.path.join(self.raw_dir, fn)
            hdulist = pyfits.open(_fn)
            data = hdulist[0].data.astype(float)
            if (bias is not None):
                data -= bias
            if (flat is not None):
                data /= flat
            _list.append(data)
        stack = numpy.array(_list)
        combined = op(stack, axis=0)
        return combined

    def make_master_bias(self, save=None):
        self.logger.info("Creating master bias")
        self.master_bias = self.basic_reduction(
            filelist=self.config['bias'],
            bias=None, flat=None, op=numpy.median)
        print(self.master_bias.shape)
        if (save is not None):
            self.logger.info("Writing master bias to %s", save)
            pyfits.PrimaryHDU(data=self.master_bias).writeto(save, overwrite=True)

    def make_master_flat(self, save=None):
        self.logger.info("Creating master flat")
        _list = []
        for fn in self.config['flat']:
            _fn = os.path.join(self.raw_dir, fn)
        hdulist = pyfits.open(_fn)
        data = hdulist[0].data.astype(float)
        if (self.master_flat is not None):
            data -= self.master_bias
        _list.append(data)
        stack = numpy.array(_list)
        self.master_flat = numpy.mean(stack, axis=0)
        print(self.master_flat.shape)
        if (save is not None):
            self.logger.info("Writing master flat to %s", save)
            pyfits.PrimaryHDU(data=self.master_flat).writeto(save, overwrite=True)

    def make_master_comp(self, save=None):
        self.logger.info("Creating master comp")
        self.master_comp = self.basic_reduction(
            filelist=self.config['comp'],
            bias=self.master_bias, flat=None,
            op=numpy.median
        )
        print(self.master_comp.shape)
        if (save is not None):
            self.logger.info("Writing master comp to %s", save)
            pyfits.PrimaryHDU(data=self.master_comp).writeto(save, overwrite=True)


    def reduce(self, save=False):

        _master_bias_fn = "master_bias.fits" if save else None
        self.make_master_bias(save=_master_bias_fn)

        _master_flat_fn = "master_flat.fits" if save else None
        self.make_master_flat(save=_master_flat_fn)

        _master_comp_fn = "master_comp.fits" if save else None
        self.make_master_comp(save=_master_comp_fn)

if __name__ == '__main__':

#    logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config',
                        type=str, default='setup.json')
    parser.add_argument('--rawdir', dest='raw_dir',
                        type=str, default='raw/')
    args = parser.parse_args()

    benchspec = BenchSpek(args.config, args.raw_dir)
    # print(json.dumps(benchspec.config, indent=2))

    benchspec.reduce(save=True)
