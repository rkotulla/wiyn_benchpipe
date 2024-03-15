#!/usr/bin/env python

import argparse
import os
import sys
import astropy.io.fits as pyfits
import json
import scipy.ndimage

import numpy
import scipy
import scipy.signal
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

    def trace_fibers_raw(self, flat=None, save=None):
        if (flat is None):
            flat = self.master_flat

        self.full_y = numpy.arange(flat.shape[0])
        self.n_fibers = 82

        #
        # do a background subtraction first to increase contrast
        #
        # first step, reject outliers by median-filtering ALONG fibers
        # (9,1) works well for sparsepak (9 px wide across fibers, 1px long along fibers)
        self.logger.debug("Preparing frame for fiber tracing")
        median_filter_1d = scipy.ndimage.median_filter(
            input=flat, size=(9, 1),
        )
        min_filter = scipy.ndimage.minimum_filter(
            input=median_filter_1d,  # masterflat,
            size=(5, 30)
        )
        # Now fit a linear slope to the background
        left_edge = 80 ## adjust this for binning, assuming 4x3
        right_edge = 570
        w = 10

        left = numpy.mean(min_filter[:, left_edge - w:left_edge + w], axis=1).reshape((-1, 1))
        right = numpy.mean(min_filter[:, right_edge - w:right_edge + w], axis=1).reshape((-1, 1))
        slope = (right - left) / (right_edge - left_edge)
        iy, ix = numpy.indices(flat.shape)
        gradient_2d = (ix - left_edge) * slope + left

        # subtract the modeled background
        bgsub = flat - gradient_2d
        bgsub[bgsub < 0] = 0

        # Now trace the fibers
        self.logger.debug("Tracing ridge-lines of each fiber")
        dy = 5 ## adjust for binning
        traces = pandas.DataFrame()
        all_peaks = []
        all_traces_y = []
        for y in range(dy, bgsub.shape[0], 2 * dy):
            prof = numpy.nanmean(bgsub[y - dy:y + dy, :], axis=0)
            peak_intensity = numpy.mean(prof)
            # print(y, peak_intensity)

            peaks, peak_props = scipy.signal.find_peaks(prof, height=0.5 * peak_intensity, distance=3)
            if (peaks.shape[0] != self.n_fibers):   # adjust for other instruments -- 82 is for sparsepak
                print(y, "off, #=%d" % (peaks.shape[0]))
                continue

            all_peaks.append(peaks)
            all_traces_y.append(y)

        centers = numpy.array(all_peaks)

        # derive the average spacing between fibers
        # we need this to determine the boundaries of the fibers at the
        # far-left and far-right edges
        avg_peak2peak_spacing = numpy.mean(numpy.diff(centers, axis=1), axis=0).reshape((1, -1))
        avg_peak2peak_vertical = numpy.mean(numpy.diff(centers, axis=1), axis=1).reshape((-1, 1))
        # print(avg_peak2peak_spacing)
        self.logger.info("Average fiber spacing: %f pixels",
                         numpy.mean(avg_peak2peak_spacing))

        leftmost_peak = numpy.min(centers, axis=1)
        rightmost_peak = numpy.max(centers, axis=1)

        # invert masterflat to search for the minima between the cells
        self.logger.debug("Tracing valley lines that limit fibers")
        inverted = -1. * bgsub
        all_troughs = []
        for i, y in enumerate(all_traces_y):
            prof = numpy.nanmean(inverted[y - dy:y + dy, :], axis=0)
            peak_intensity = numpy.min(prof)
            # print(y, peak_intensity)

            peaks, peak_props = scipy.signal.find_peaks(prof, height=0.5 * peak_intensity, distance=3)

            _left = leftmost_peak[i]
            _right = rightmost_peak[i]
            good = (peaks > _left) & (peaks < _right)
            good_peaks = peaks[good]
            # print(y, peak_intensity, peaks.shape, good_peaks.shape)
            all_troughs.append(good_peaks)

        all_troughs = numpy.array(all_troughs)

        # figure out the outer edge of the left & rightmost fibers
        far_left = centers[:, 0].reshape((-1, 1)) - 0.5 * avg_peak2peak_vertical
        print(far_left.shape)
        far_right = centers[:, -1].reshape((-1, 1)) + 0.5 * avg_peak2peak_vertical
        all_lefts = numpy.hstack([far_left, all_troughs])
        all_rights = numpy.hstack([all_troughs, far_right])

        # Now we have the coarsely sampled position along the fiber, upscale this to
        # full frame
        self.logger.debug("Upsampling fiber traces to full resolution")
        y_dim = self.full_y.shape[0]
        fullres_left    = numpy.full((y_dim, self.n_fibers), fill_value=numpy.NaN)
        fullres_right   = numpy.full((y_dim, self.n_fibers), fill_value=numpy.NaN)
        fullres_centers = numpy.full((y_dim, self.n_fibers), fill_value=numpy.NaN)
        for fiber_id in range(self.n_fibers):
            for meas, full in zip([all_lefts, all_rights, centers], [fullres_left, fullres_right, fullres_centers]):
                polyfit = numpy.polyfit(all_traces_y, meas[:, fiber_id], deg=2)
                full[:, fiber_id] = numpy.polyval(polyfit, self.full_y)

        self.logger.info("All done tracing fibers in original pixel space")


    def reduce(self, save=False):

        _master_bias_fn = "master_bias.fits" if save else None
        self.make_master_bias(save=_master_bias_fn)

        _master_flat_fn = "master_flat.fits" if save else None
        self.make_master_flat(save=_master_flat_fn)

        _master_comp_fn = "master_comp.fits" if save else None
        self.make_master_comp(save=_master_comp_fn)

        self.trace_fibers_raw(flat=self.master_flat)

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
