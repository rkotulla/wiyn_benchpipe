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
import scipy.ndimage
import sklearn
import matplotlib.pyplot as plt
import glob
import pandas
import itertools
import logging

import fibertraces
from fibertraces import *


class BenchSpek(object):

    config = None
    raw_directory = "."
    master_bias = None
    master_flat = None
    master_comp = None
    n_fibers = 82
    linelist = []

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


    def read_reference_linelist(self):
        opt = self.config['linelist']
        if (opt.endswith(".fits")):
            # read file, identify lines
            pass
        else:
            # read lines from file
            linelist = []
            self.logger.info("Reading line list from text file (%s)", opt)
            with open(opt, "r") as ll:
                lines = ll.readlines()
                for l in lines:
                    wl = float(l.strip().split(" ")[0])
                    linelist.append(wl)
            self.linelist = numpy.array(linelist)
        self.logger.info("Found %d reference lines for wavelength calibration",
                         self.linelist.shape[0])
        return self.linelist

    def find_lines(self, spec, threshold=1300, distance=5):
        # first, apply running minimum filter to take out continuum
        mins = scipy.ndimage.minimum_filter(input=spec, size=20, mode='constant', cval=0)
        # since the direct minimum is too edge-y, let's smooth it
        cont = scipy.ndimage.gaussian_filter1d(input=mins, sigma=10)

        contsub = spec - cont
        # min_peak_height = 300
        peaks, peak_props = scipy.signal.find_peaks(contsub, height=threshold, distance=distance)
        return contsub, peaks

    def find_wavelength_solution(self, spec, lambda_central=None, dispersion=None, min_lines=3, make_plots=True):

        self.logger.info("Finding wavelength solution")

        if (lambda_central is None):
            lambda_central = self.config['central_wavelength']
        if (dispersion is None):
            dispersion = self.config['dispersion']

        print("spec shape:", spec.shape)

        # find peaks
        contsub, peaks = self.find_lines(spec, threshold=1300, distance=5)
        self.logger.info("Found %d peaks" % (peaks.shape[0]))
        full_y = numpy.arange(spec.shape[0])

        if (make_plots):
            fig, ax = plt.subplots(figsize=(13, 5))
            ax.plot(full_y, contsub, lw=0.5)
            # ax.plot(wl, contsub, lw=0.5)
            ylabelpos = 20000
            for p in peaks:
                ax.axvline(x=p, ymin=0.0, ymax=0.8, lw=0.2, color='red', alpha=0.5)
                ax.text(p, ylabelpos, "%d" % (p), rotation='vertical', ha='center')
            #     if (reflines is not None):
            #         ax.scatter(reflines, numpy.ones_like(reflines)*2500, marker="|")
            # ax.set_yscale('log')
            ax.set_ylim(0, 25000)
            fig.savefig("reference_spectrum.png", dpi=300)

        # generate a tree for the reference lines
        reflines = self.linelist
        ref2d = numpy.array([reflines, reflines]).T
        # print(ref2d.shape)
        ref_tree = scipy.spatial.KDTree(
            data=ref2d
        )

        # scan the range
        var_wl, n_wl = 0.002, 100
        var_disp, n_disp = 0.05, 100
        scan_wl = numpy.linspace(lambda_central * (1. - var_wl), lambda_central * (1 + var_wl), n_wl)
        scan_disp = numpy.linspace(dispersion * (1. - var_disp), dispersion * (1. + var_disp), n_disp)
        self.logger.debug("Scanned central wavelength range: %.3f ... %.3f" % (scan_wl[0], scan_wl[-1]))
        self.logger.debug("Scanned dispersion range: %.3f ... %.3f" % (scan_disp[0], scan_disp[-1]))

        central_y = full_y[full_y.shape[0] // 2]
        peaks2d = numpy.array([peaks, peaks]).T
        results = []
        # print(ref2d.shape)
        for cw, disp in itertools.product(scan_wl, scan_disp):
            wl = cw + (peaks2d - central_y) * disp
            # print(wl.shape)
            # break

            #
            # find matches based on simple solution
            #
            d, i = ref_tree.query(wl, k=1, p=1, distance_upper_bound=4)
            n_good_line_matches = numpy.sum(i < ref_tree.n)

            #
            # assume we correctly matched all matched lines, let's do a proper fit and recount matches
            #
            testfit = [0, 0, 0]
            if (n_good_line_matches >= min_lines):
                match = (i < ref_tree.n)
                # variance = numpy.var()

                #
                # Find which lines match
                #
                ref_wl = reflines[i[match]]
                lines_y = peaks[match] - central_y
                #         print(ref_wl)
                #         print(lines_y)
                #         print(testfit)

                #
                # Perform a proper dispersion solution fit using the matched line pairs
                #
                testfit = numpy.polyfit(lines_y, ref_wl, deg=2)
                wl_postfit = numpy.polyval(testfit, peaks2d - central_y)
                d2, i2 = ref_tree.query(wl_postfit, k=1, p=1, distance_upper_bound=2)
                n_good_line_matches2 = numpy.sum(i2 < ref_tree.n)
                ref_wl_refined = reflines[i2[i2 < ref_tree.n]]

                #
                # Using the proper fit, calculate the calibrated position for each line
                # (based on the on-detector Y position and the wavelength solution we just derived )
                #
                mylines = wl_postfit[i2 < ref_tree.n, 0]
                #             print(mylines.shape, ref_wl.shape, wl_postfit.shape)
                # mylines_wl = numpy.polyval(testfit, mylines - central_y)

                #
                # and finally figure out the quality of the solution based on the observed dispersion
                #
                delta_wl = mylines - ref_wl_refined
                var_wl = numpy.std(delta_wl)
                # print(delta_wl, var_wl)
            else:
                n_good_line_matches2 = 0
                var_wl = 999

            results.append(
                [cw, disp, n_good_line_matches, n_good_line_matches2, var_wl, testfit[0], testfit[1], testfit[2]])
            # break

        results = numpy.array(results)
        numpy.savetxt("results.dump", results)

        i_most_matches = numpy.argmax(results[:, 3])
        print(i_most_matches)
        print("most matches", results[i_most_matches])

        all_best_matches = results[(results[:, 3] >= results[i_most_matches, 3])]

        i_smallest_scatter = numpy.argmin(all_best_matches[:, 4])
        best_solution = all_best_matches[i_smallest_scatter]
        print(best_solution)

        return best_solution  # results[i_most_matches]

    def reduce(self, save=False):

        _master_bias_fn = "master_bias.fits" if save else None
        self.make_master_bias(save=_master_bias_fn)

        _master_flat_fn = "master_flat.fits" if save else None
        self.make_master_flat(save=_master_flat_fn)

        _master_comp_fn = "master_comp.fits" if save else None
        self.make_master_comp(save=_master_comp_fn)

        self.logger.info("Tracing fibers")
        # self.trace_fibers_raw(flat=self.master_flat)


        self.logger.info("Extracting fiber spectra from master flat")
        raw_tracers = fibertraces.SparsepakFiberSpecs()
        raw_tracers.find_trace_fibers(self.master_flat)
        # comp_spectra = raw_tracers.extract_fiber_spectra(
        #     imgdata=self.master_comp,
        #     weights=self.master_flat,
        # )

        # self.flat_spectra = self.extract_spectra_raw(imgdata=self.master_flat, weights=self.master_flat)
        self.flat_spectra = raw_tracers.extract_fiber_spectra(
            imgdata=self.master_flat, weights=self.master_flat)
        # print(self.flat_spectra)
        numpy.savetxt("flat_spectra2.dat", self.flat_spectra)

        self.logger.info("Extracting fiber spectra from master comp")
        # self.comp_spectra = self.extract_spectra_raw(imgdata=self.master_comp, weights=self.master_flat)
        self.comp_spectra = raw_tracers.extract_fiber_spectra(
            imgdata=self.master_comp, weights=self.master_flat)
        numpy.savetxt("comp_spectra2.dat", self.comp_spectra)

        self.read_reference_linelist()

        # find wavelength solution for one "reference" fiber
        self.ref_fiberid = 41
        _wavelength_solution = self.find_wavelength_solution(self.comp_spectra[:, self.ref_fiberid])
        self.wavelength_solution = _wavelength_solution[-3:]
        print("wavelength solution:", self.wavelength_solution)

        # Now re-identify lines across all other fiber traces



if __name__ == '__main__':

#    logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)
    logging.basicConfig(level=logging.DEBUG)

    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config',
                        type=str, default='setup.json')
    parser.add_argument('--rawdir', dest='raw_dir',
                        type=str, default='raw/')
    args = parser.parse_args()

    benchspec = BenchSpek(args.config, args.raw_dir)
    # print(json.dumps(benchspec.config, indent=2))

    benchspec.reduce(save=True)
