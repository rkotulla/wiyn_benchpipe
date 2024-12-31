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
import scipy.interpolate
import scipy.optimize
import sklearn
import matplotlib.pyplot as plt
import glob
import pandas
import itertools
import logging

from specutils import Spectrum1D
import astropy.units as u
from specutils.manipulation import FluxConservingResampler, LinearInterpolatedResampler, SplineInterpolatedResampler

import fibertraces
from fibertraces import *
from grating import *

import warnings
#with warnings.catch_warnings():
print("Disabling warnings")
warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
warnings.filterwarnings('ignore', r'divide by zero encountered in divide')
warnings.filterwarnings('ignore', r'invalid value encountered in divide')
warnings.simplefilter("ignore", RuntimeWarning)

def gauss(x, center, sigma, amplitude):
    return amplitude * numpy.exp(-(x - center) ** 2 / sigma ** 2)


def _fit_gauss(p, x, flux):
    model = gauss(x, center=p[0], sigma=p[1], amplitude=p[2])
    diff = model - flux
    noise = 100
    return (diff / noise) ** 2



class BenchSpek(object):

    config = None
    raw_dir  = None
    master_bias = None
    master_flat = None
    master_comp = None
    n_fibers = 82
    linelist = []
    output_wl_min = None
    output_wl_max = None
    output_dispersion = None

    def __init__(self, json_file, raw_dir=None):
        self.logger = logging.getLogger('BenchSpek')

        self.json_file = json_file
        self.read_config()
        if (raw_dir is not None and os.path.isdir(raw_dir)):
            self.raw_dir = raw_dir

    def read_config(self):
        self.logger.info(self.json_file)
        with open(self.json_file, "r") as f:
            self.config = json.load(f)
        if (self.raw_dir is None):
            try:
                self.raw_dir = self.config['raw_directory']
            except:
                self.raw_dir = "."

        # ensure output directories exist
        if (not os.path.isdir(self.config['cals_directory'])):
            self.logger.info("Creating CALS directory: %s" % (self.config['cals_directory']))
            os.makedirs(self.config['cals_directory'])
        else:
            self.logger.info("CALS directory (%s) exists" % (self.config['cals_directory']))
        if (not os.path.isdir(self.config['out_directory'])):
            self.logger.info("Creating OUTPUT directory: %s" % (self.config['out_directory']))
            os.makedirs(self.config['output_directory'])
        else:
            self.logger.info("OUTPUT directory (%s) already exists" % (self.config['out_directory']))

    def write_FITS(self, hdulist, filename, overwrite=True):
        self.logger.info("Writing FITS file (%s)" % (filename))
        hdulist.writeto(filename, overwrite=overwrite)

    def write_cals_FITS(self, hdulist, filename, **kwargs):
        full_fn = os.path.join(self.config['cals_directory'], filename)
        self.write_FITS(hdulist, full_fn, **kwargs)

    def write_results_FITS(self, hdulist, filename, **kwargs):
        full_fn = os.path.join(self.config['out_directory'], filename)
        self.write_FITS(hdulist, full_fn, **kwargs)

    def basic_reduction(self, filelist, bias=None, flat=None, op=numpy.mean):
        _list = []
        header = None
        for fn in filelist:
            _fn = os.path.join(self.raw_dir, fn)
            hdulist = pyfits.open(_fn)
            if (header is None):
                header = hdulist[0].header
            data = hdulist[0].data.astype(float)
            if (bias is not None):
                data -= bias
            if (flat is not None):
                data /= flat
            _list.append(data)
        stack = numpy.array(_list)
        combined = op(stack, axis=0)
        return combined, header

    def make_master_bias(self, save=None):
        self.logger.info("Creating master bias")
        self.master_bias, _ = self.basic_reduction(
            filelist=self.config['bias'],
            bias=None, flat=None, op=numpy.median)
        print(self.master_bias.shape)
        if (save is not None):
            self.logger.info("Writing master bias to %s", save)
            self.write_cals_FITS(pyfits.PrimaryHDU(data=self.master_bias), filename=save)
            # pyfits.PrimaryHDU(data=self.master_bias).writeto(save, overwrite=True)

    def make_master_flat(self, save=None):
        self.logger.info("Creating master flat")
        _list = []
        for fn in self.config['flat']:
            _fn = os.path.join(self.raw_dir, fn)
            hdulist = pyfits.open(_fn)
            data = hdulist[0].data.astype(float)
            if (self.master_bias is not None):
                data -= self.master_bias
            _list.append(data)
        stack = numpy.array(_list)
        self.master_flat = numpy.median(stack, axis=0)
        # print(self.master_flat.shape)
        # if (save is not None):
        #     self.logger.info("Writing master flat to %s", save)
        #     pyfits.PrimaryHDU(data=self.master_flat).writeto(save, overwrite=True)

        # normalize flat-field
        avg_flat = numpy.nanmean(self.master_flat, axis=1).reshape((-1,1))
        print(avg_flat.shape)
        self.master_flat /= avg_flat
        numpy.savetxt("masterflat_norm.txt", avg_flat)
        if (save is not None):
            self.logger.info("Writing master flat to %s", save)
            self.write_cals_FITS(pyfits.PrimaryHDU(data=self.master_flat), filename=save)


    def make_master_comp(self, save=None):
        self.logger.info("Creating master comp")
        self.master_comp, self.comp_header = self.basic_reduction(
            filelist=self.config['comp'],
            bias=self.master_bias, flat=self.master_flat,
            op=numpy.median
        )
        self.logger.debug("MasterComp dimensions: %s" % (str(self.master_comp.shape)))
        # print(self.master_comp.shape)
        # print(self.comp_header)
        if (save is not None):
            self.logger.info("Writing master comp to %s", save)
            pyfits.PrimaryHDU(data=self.master_comp, header=self.comp_header).writeto(save, overwrite=True)


    def read_reference_linelist(self):
        opt = self.config['linelistx']
        linelist = []
        if (opt.endswith(".fits")):
            # read file, identify lines
            self.logger.info("Reading line list from FITS file (%s)", opt)
            self.logger.warning("NOT IMPLEMENTED")
            pass
        else:
            # read lines from file
            with open(opt, "r") as ll:
                lines = ll.readlines()
                for l in lines:
                    wl = float(l.strip().split(" ")[0])
                    linelist.append(wl)
            linelist = numpy.array(linelist)
        self.logger.info("Found %d reference lines for wavelength calibration",
                         linelist.shape[0])

        fine_linelist = pandas.DataFrame()
        fine_linelist['peak_pos'] = self.linelist
        fine_linelist['center'] = self.linelist
        fine_linelist['sigma'] = 1.0
        fine_linelist['peak_pos'] = 1.0

        # all positions read from file are already converted to actual calibrated wavelengths
        fine_linelist['cal_peak_pos'] = fine_linelist['peak_pos']
        fine_linelist['cal_center'] = fine_linelist['center']

        self.linelist = fine_linelist

        return self.linelist

    def find_refined_lines(self, spec):
        return

    def find_lines(self, spec, threshold=1300, distance=5):
        # first, apply running minimum filter to take out continuum
        mins = scipy.ndimage.minimum_filter(input=spec, size=20, mode='constant', cval=0)
        # since the direct minimum is too edge-y, let's smooth it
        cont = scipy.ndimage.gaussian_filter1d(input=mins, sigma=10)

        contsub = spec - cont
        # min_peak_height = 300
        peaks, peak_props = scipy.signal.find_peaks(contsub, height=threshold, distance=distance)
        return contsub, peaks

    def fine_line_centroiding(self, spec, line_pos):
        # print(spec.shape)
        # print(line_pos)

        fig, ax = plt.subplots(figsize=(14, 4))
        x = numpy.arange(spec.shape[0])
        ax.plot(x, spec, lw=0.5)
        # ax.set_xlim((x1,x2))
        ax.set_xlim((10e3, 12e3))
        ax.set_ylim((0, 1e5))

        # in_window = (line_pos > x1) & (line_pos<x2)
        # line_pos = line_pos[in_window]

        peak_flux = spec[line_pos]
        ax.scatter(line_pos, peak_flux + 3e3, marker="|")

        window_size = 5
        fine_lines = pandas.DataFrame()  # numpy.full((line_pos.shape[0],5), numpy.nan)
        for i, line in enumerate(line_pos):
            _left = int(numpy.floor(line - window_size))
            _right = int(numpy.ceil(line + window_size))
            w_x = x[_left:_right + 1]
            w_spec = spec[_left:_right + 1]
            weighted = numpy.sum(w_x * w_spec) / numpy.sum(w_spec)

            # fine_lines[i,1] = weighted
            ax.axvline(x=weighted, lw=0.2)

            fit_results = scipy.optimize.leastsq(
                func=_fit_gauss,
                x0=[line, 3, peak_flux[i]],
                args=(w_x, w_spec)
            )
            # print(line, weighted, fit_results[0])

            m_gauss = gauss(w_x, fit_results[0][0], fit_results[0][1], fit_results[0][2])
            ax.plot(w_x, m_gauss)

            fine_lines.loc[i, 'peak_pos'] = line
            fine_lines.loc[i, 'center'] = fit_results[0][0]
            fine_lines.loc[i, 'sigma'] = fit_results[0][1]
            fine_lines.loc[i, 'amplitude'] = fit_results[0][2]

        return fine_lines

        # ax.scatter(fine_lines, numpy.ones_like(line_pos)*23e3, marker="|")


    def get_refined_lines_from_spectrum(self, spec, distance=5, window_size=8, filter=True, return_contsub=False):
        # find a background approximation for the spectrum
        mins = scipy.ndimage.minimum_filter(input=spec, size=20, mode='constant', cval=0)
        cont = scipy.ndimage.gaussian_filter1d(input=mins, sigma=10)

        # prepare dataframe for the results
        line_inventory = pandas.DataFrame()


        contsub = spec-cont
        x = numpy.arange(spec.shape[0])

        supersample = 5
        markers = ['x', 'o', '+', 'o']
        fine_x = numpy.arange(int(supersample*contsub.shape[0]), dtype=float)/supersample

        # find the detection thresholds
        stats = numpy.nanpercentile(contsub, [16,50,84])
        _med = stats[1]
        _sigma = 0.5*(stats[2]-stats[0])
        self.logger.debug("continuum subtracted spec: median=%f, sigma=%f" % (_med, _sigma))

        width_buffer=1.5
        thresholds = numpy.array([10,5,3,2,1]) * _sigma
        # thresholds = numpy.array([20,5,2]) * _sigma
        for iteration, threshold in enumerate(thresholds):  # range(5):

            self.logger.debug("*" * 25 + "\n\n   ITERATION %d\n\n" % (iteration + 1) + "*" * 25)
            added_new_line = False

            gf = numpy.zeros_like(contsub, dtype=float)
            diffslopes_left = numpy.pad(numpy.diff(contsub), (0, 1), mode='constant', constant_values=0)
            diffslopes_right = numpy.pad(numpy.diff(contsub), (1, 0), mode='constant', constant_values=0)

            peaks, peak_props = scipy.signal.find_peaks(contsub, height=threshold, distance=distance)

            # fig, ax = plt.subplots(figsize=(25, 5))

            next_cs = contsub.copy()
            for i, line in enumerate(peaks):

                # check if we already have a line at this position
                if (len(line_inventory.index) > 0):
                    gc = line_inventory['gauss_center'].to_numpy()
                    gw = line_inventory['gauss_width'].to_numpy()
                    gw[gw > distance] = distance
                    match = (line > (gc - width_buffer * gw)) & (line < (gc + width_buffer * gw))
                    # print(match)
                    if (numpy.sum(match) > 0):
                        # self.logger.debug("already found a line at this approximate position %d" % (line))
                        continue
                    else:
                        # self.logger.debug("No counterpart found @ %d, continuing" % (line))
                        pass
                else:
                    # this is the first line, so we for sure don't know about it yet
                    pass

                added_new_line = True

                _left = numpy.max([0, int(numpy.floor(line - window_size))])
                _right = numpy.min([int(numpy.ceil(line + window_size)), spec.shape[0] - 1])

                try:
                    _left2 = numpy.max([_left, numpy.max(x[(x < line) & (diffslopes_left < 0)]) + 1])
                except ValueError:
                    _left2 = _left

                try:
                    _right2 = numpy.min([_right, numpy.min(x[(x > line) & (diffslopes_right > 0)]) - 1])
                except ValueError:
                    _right2 = _right
                # print(_left, _right, "-->", _left2, _right2)

                # generate a flux-weighted mean position as starting guess for the following gauss fit
                w_x = x[_left2:_right2 + 1]
                w_spec = spec[_left2:_right2 + 1]
                weighted = numpy.sum(w_x * w_spec) / numpy.sum(w_spec)
                # print(line, weighted)
                peak_flux = contsub[line]

                full_fit_results = scipy.optimize.leastsq(
                    func=_fit_gauss,
                    x0=[weighted, 1, peak_flux],
                    args=(w_x, w_spec)
                )
                gaussfit = full_fit_results[0]
                # print(line, gaussfit)

                _fine_x = fine_x[_left * supersample:_right * supersample + 1]
                modelgauss = gauss(w_x, gaussfit[0], gaussfit[1], gaussfit[2])
                next_cs[_left2:_right2 + 1] -= modelgauss
                gf[_left2:_right2 + 1] += modelgauss
                # centers.append(gaussfit[0])

                # ax.scatter(x[_left2:_right2+1], cs[_left2:_right2+1], marker=markers[i%4], alpha=0.5)

                # fine_gauss = gauss(_fine_x, gaussfit[0], gaussfit[1], gaussfit[2])
                # ax.plot(_fine_x, fine_gauss, lw=1.2, ls=":", c='black')

                idx = len(line_inventory.index) + 1
                line_inventory.loc[idx, 'position'] = line
                line_inventory.loc[idx, 'peak'] = contsub[line]
                line_inventory.loc[idx, 'gauss_center'] = gaussfit[0]
                line_inventory.loc[idx, 'gauss_width'] = gaussfit[1]
                line_inventory.loc[idx, 'gauss_amp'] = gaussfit[2]
                line_inventory.loc[idx, 'center_weight'] = weighted
                line_inventory.loc[idx, 'iteration'] = iteration
                line_inventory.loc[idx, 'threshold'] = threshold
                line_inventory.loc[idx, 'fake_x'] = 500

                # linemask = x > gaussfit[0]-

            # print(len(peaks), 'liens founds')
            # ax.plot(x, cs, lw=0.3)
            # # ax.plot(x, next_cs, lw=0.5)
            # # ax.plot(x, gf, lw=0.5, ls=":")
            # ax.set_title("iteration: %d" % (iteration))
            # # ax.axhline(y=threshold)
            # ax.axhline(y=0)
            # ax.scatter(centers, numpy.ones_like(centers) * 100, marker='|')
            # ax.scatter(peaks, numpy.ones_like(peaks) * 150, marker='|')
            # ax.set_ylim((-0.5 * threshold, 5 * threshold))
            # # ax.set_xlim((100,700))
            # # ax.set_xlim((350,410))
            contsub = next_cs
            # cs_iterations.append(cs.copy())

            if (not added_new_line):
                self.logger.debug("No new lines found, aborting search")
                break

        if (filter):
            # apply some filtering:

            # require the gauss-fits to have similar peaks to the actual data
            amp_ratio = line_inventory['peak'] / line_inventory['gauss_amp']
            good = (amp_ratio > 0.5) & (amp_ratio < 1.5)

            # only select lines with "typical" line widths
            gw = line_inventory['gauss_width'].to_numpy()
            for i in range(3):
                _stats = numpy.nanpercentile(gw[good], [16, 50, 84])
                _med = _stats[1]
                _sig = 0.5 * (_stats[2] - _stats[0])
                good = good & (gw > (_med - 3 * _sig)) & (gw < (_med + 3 * _sig))

            line_inventory = line_inventory[good]

        # line_inventory.to_csv("line_inventory.csv", index=False)
        if (return_contsub):
            contsub = spec - cont
            return line_inventory, contsub
        return line_inventory

    def find_reflines_from_spec(self, ref_spec_fn=None, sci_sigma=None, wl_min=None, wl_max=None):

        if (ref_spec_fn is None):
            ref_spec_fn = "scidoc2212.fits"
        self.logger.debug("Reading wavelength reference spectrum from %s" % (ref_spec_fn))
        hdu = pyfits.open(ref_spec_fn)
        s = hdu[0].data
        # hdu.info()
        # hdu[0].header
        fig, ax = plt.subplots(figsize=(13, 4))
        _x = numpy.arange(s.shape[0], dtype=float) + 1.
        _l = (_x - hdu[0].header['CRPIX1']) * hdu[0].header['CD1_1'] + hdu[0].header['CRVAL1']

        ref_inventory, contsub = self.get_refined_lines_from_spectrum(spec=s, return_contsub=True)

        # threshold = 5000
        # contsub, reflines = self.find_lines(s, threshold=threshold, distance=10)
        # # print(reflines)
        # # refpeaks, props = scipy.signal.find_peaks(s, height=5000, distance=30)
        # reflines = ref_inventory['gauss_center'].to_numpy()
        # refpeaks_wl = (reflines - hdu[0].header['CRPIX1']) * hdu[0].header['CD1_1'] + hdu[0].header['CRVAL1']
        ref_inventory['gauss_wl'] = (ref_inventory['gauss_center'] - hdu[0].header['CRPIX1']) * hdu[0].header['CD1_1'] + hdu[0].header['CRVAL1']

        # trim down lines to approximately the observed range
        keepers = numpy.isfinite(ref_inventory['gauss_wl'])
        if (wl_min is not None):
            keepers[ref_inventory['gauss_wl'] < wl_min] = False
        if (wl_max is not None):
            keepers[ref_inventory['gauss_wl'] > wl_max] = False
        ref_inventory = ref_inventory[keepers]

        self.refspec_raw = hdu[0].data
        self.refspec_continnum_subtracted = contsub
        self.refspec_wavelength = _l

        # widths, _, _, _ = scipy.signal.peak_widths(contsub, reflines)
        widths = ref_inventory['gauss_width'].to_numpy()
        reflines = ref_inventory['gauss_center'].to_numpy()
        # print(linewidths)
        good_width = numpy.isfinite(widths) # & (reflines > 6300) & (reflines < 6900)
        for _iter in range(3):
            stats = numpy.nanpercentile(widths[good_width], [16, 50, 84])
            _med = stats[1]
            _sigma = 0.5 * (stats[2] - stats[0])
            good_width = good_width & (widths < (_med + 3 * _sigma)) & (widths > (_med - 3 * _sigma))
        med_ref_width = numpy.nanmedian(widths[good_width])
        self.logger.debug("reference spectrum line width: %.4f pixels ==> %.4f AA" % (
            med_ref_width, (med_ref_width * hdu[0].header['CD1_1'])))
        # self.logger.debug("reference spectrum line width: %.4f AA" % )

        sci_dispersion = 0.31
        med_width = 3.3
        if (sci_sigma is None):
            sci_sigma = med_width * sci_dispersion / 2.634
        ref_dispersion = hdu[0].header['CD1_1']
        ref_sigma = med_ref_width * ref_dispersion / 2.634
        self.logger.debug("Comparing instrumental resolutions: data:%fAA reference:%fAA" % (sci_sigma, ref_sigma))

        if (sci_sigma <= ref_sigma):
            smooth_sigma = None
            smooth_px_width = None
            self.logger.debug("Data is higher resolution than reference, no reference smoothing needed")
            smoothed = contsub.copy()
        else:
            smooth_sigma = numpy.sqrt(sci_sigma ** 2 - ref_sigma ** 2)
            smooth_px_sigma = smooth_sigma / ref_dispersion
            self.logger.debug("smoothing needed: sigma=%fAA ==> %.2fpx" % (smooth_sigma, smooth_px_sigma))
            smoothed = scipy.ndimage.gaussian_filter1d(contsub, sigma=smooth_px_sigma)
        numpy.savetxt("refspec_smoothed", smoothed)
        numpy.savetxt("refspec_contsub", contsub)
        # print(type(refpeaks_wl))
        # print(refpeaks_wl)

        self.refspec_smoothed = smoothed

        self.logger.debug("Creating diagnostic plot")
        ax.plot(_l, contsub, lw=0.2, label='contsub')
        ax.plot(_l, smoothed, lw=0.5, c='red', label='smoothed')
        # ax.scatter(_l, contsub, s=0.2)
        ax.set_xlim((6350, 6680))
        ax.set_ylim((0, 2e4))

        ax.set_xlim((6500, 6600))
        ax.set_xlim((6000, 7000))
        ax.set_ylim((0, 5e4))

        # sel_wl = refpeaks_wl[(refpeaks_wl > 6350) & (refpeaks_wl < 6480) ]
        # TODO: fix
        threshold = 5000
        sel_wl = ref_inventory['gauss_wl']
        ax.scatter(sel_wl, ref_inventory['gauss_amp'], marker="|", label="lines") # (thr=%g)" % (threshold))
        #ax.axhline(y=threshold)
        ax.legend()
        #plot_fn =
        fig.savefig("refspec_caliblines.png", dpi=300)
        fig.savefig("refspec_caliblines.pdf")
        self.logger.debug("Saving plot to refspec_caliblines.(pdf/png)")

        # self.logger.debug("Refining positions by centroiding")
        # fine_lines = self.fine_line_centroiding(spec=smoothed, line_pos=reflines)
        #
        # fine_lines['cal_peak_pos'] = (fine_lines['peak_pos'] - hdu[0].header['CRPIX1']) * hdu[0].header['CD1_1'] + hdu[0].header['CRVAL1']
        # fine_lines['cal_center'] = (fine_lines['center'] - hdu[0].header['CRPIX1']) * hdu[0].header['CD1_1'] + hdu[0].header['CRVAL1']
        # fine_lines.to_csv("fine_lines.csv", index=False)
        # self.linelist = fine_lines
        # self.logger.debug("Extracted a total of %d reference lines in (range %.2f .. %.2f) AA" % (
        #     len(fine_lines.index), numpy.min(fine_lines['cal_center']), numpy.max(fine_lines['center'])))
        # return fine_lines

        # Repeat line extraction etc, now that we have a resolution-matched reference spectrum
        ref_inventory = self.get_refined_lines_from_spectrum(spec=smoothed) #, return_contsub=True)
        ref_inventory['gauss_wl'] = (ref_inventory['gauss_center'] - hdu[0].header['CRPIX1']) * hdu[0].header['CD1_1'] + hdu[0].header['CRVAL1']

        # trim down lines to approximately the observed range
        keepers = numpy.isfinite(ref_inventory['gauss_wl'])
        if (wl_min is not None):
            keepers[ref_inventory['gauss_wl'] < wl_min] = False
        if (wl_max is not None):
            keepers[ref_inventory['gauss_wl'] > wl_max] = False
        ref_inventory = ref_inventory[keepers]
        self.ref_inventory = ref_inventory
        return ref_inventory

        # print(sel_wl)
        # ax.axvline(sel_wl) #, ymin=0.4, ymax=1.)

    def find_wavelength_solution(self, spec, lambda_central=None, dispersion=None, min_lines=3, make_plots=True):

        self.logger.info("Finding wavelength solution")

        if (lambda_central is None):
            lambda_central = self.get_config('setup', 'central_wavelength')
        if (dispersion is None):
            dispersion = self.get_config('setup', 'dispersion')

        # print("spec shape:", spec.shape)
        self.logger.info("User solution: central wavelength: %.3f; dispersion: %.4f" % (lambda_central, dispersion))

        self.logger.info("Getting approximate wavelength solution from grating setup")
        self.grating_solution = grating_from_header(self.comp_header)
        self.logger.info("GRATING: central wavelength: %f" % (self.grating_solution.central_wavelength))
        self.logger.info("GRATING: solution: %s" % (self.grating_solution.wl_polyfit))

        # comp_inventory = self.get_refined_lines_from_spectrum(spec)
        # # get linewidth
        # line_width_px = numpy.nanmedian(comp_inventory['gauss_width'])
        # self.find_reflines_from_spec(comp_inventory)
        # self.linelist.to_csv("reference_linelist_dummy.csv", index=False)

                #self.grating_solution


        lambda_central = self.grating_solution.central_wavelength
        dispersion = self.grating_solution.wl_polyfit[-2]

        # find peaks in the specified spectrum
        # contsub, peaks = self.find_lines(spec, threshold=500, distance=5)
        # peaks_fine = self.fine_line_centroiding(spec=contsub, line_pos=peaks)
        self.comp_line_inventory, contsub = self.get_refined_lines_from_spectrum(spec, return_contsub=True)
        self.comp_line_inventory.to_csv("inventory_comp.csv", index=False)
        peaks = self.comp_line_inventory['gauss_center'].to_numpy()
        self.comp_spectrum_raw = spec
        self.comp_spectrum_continuumsub = contsub
        self.comp_spectrum_lines = peaks

        # find typical linewidth
        line_width_px = numpy.nanmedian(self.comp_line_inventory['gauss_width'])
        line_width_AA = line_width_px * dispersion

        # now extract reference lines, after matching resolution to that of the
        # data we are about to calibrate
        self.find_reflines_from_spec(
            sci_sigma=line_width_AA,
            wl_min=self.grating_solution.wl_blueedge,
            wl_max=self.grating_solution.wl_rededge
        )
        self.ref_inventory.to_csv("inventory_refspec.csv", index=False)

        numpy.savetxt("spec_spec", spec)
        numpy.savetxt("spec_peaks", peaks)
        # numpy.savetxt("spec_peaks_fine", peaks_fine)
        numpy.savetxt("spec_contsub", contsub)
        self.logger.info("Found %d peaks in arc spectrum" % (peaks.shape[0]))
        full_y = numpy.arange(spec.shape[0])
        full_y0 = full_y - spec.shape[0]/2
        full_wl = numpy.polyval(self.grating_solution.wl_polyfit, full_y0)
        peaks0 = peaks - spec.shape[0]/2
        peaks_wl = numpy.polyval(self.grating_solution.wl_polyfit, peaks0)
        self.comp_spectrum_full_y = full_y
        self.comp_spectrum_full_y0 = full_y0
        self.comp_spectrum_center_y = spec.shape[0]/2

        if (make_plots):
            self.logger.debug("Generating initial reference spectrum comparison")
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
            self.logger.debug("Saved plot to reference_spectrum.png")

        # generate a tree for the reference lines
        in_window = (self.ref_inventory['gauss_wl'] >= self.grating_solution.wl_blueedge) & \
                    (self.ref_inventory['gauss_wl'] <= self.grating_solution.wl_rededge)
        selected_list = self.ref_inventory[in_window]
        # selected_list.info()
        reflines = selected_list['gauss_wl'].reset_index(drop=True).to_numpy()
        self.logger.info("Found %s calibrated reference lines" % (str(reflines.shape)))

        if (True):
            fig, ax = plt.subplots(figsize=(13, 5))
            ax.plot(full_wl, contsub, lw=0.5)
            # ax.plot(wl, contsub, lw=0.5)
            ylabelpos = 20000
            for p in peaks_wl:
                ax.axvline(x=p, ymin=0.0, ymax=0.8, lw=0.2, color='red', alpha=0.5)
                ax.text(p, ylabelpos, "%d" % (p), rotation='vertical', ha='center')
            ax.scatter(reflines, numpy.ones_like(reflines)*2500, marker="|")
            # ax.set_yscale('log')
            ax.set_ylim(0, 25000)
            fig.savefig("reference_spectrum_wlcal.png", dpi=300)

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

        results_df = pandas.DataFrame()
        for i_iter, (cw, disp) in enumerate(itertools.product(scan_wl, scan_disp)):

            # for simple linear fit -- maybe not as good
            # wl = cw + (peaks2d - central_y) * disp

            # use the modified grating solution
            wl_polyfit_mod = self.grating_solution.wl_polyfit.copy()
            wl_polyfit_mod[-1] = cw
            wl_polyfit_mod[-2] = disp
            peaks0 = peaks2d - central_y
            wl = numpy.polyval(wl_polyfit_mod, peaks0)

            # print(wl.shape)
            # break

            #
            # find matches based on simple solution
            #
            d, i = ref_tree.query(wl, k=1, p=1, distance_upper_bound=2)
            n_good_line_matches = numpy.sum(i < ref_tree.n)

            #
            # assume we correctly matched all matched lines, let's do a proper fit and recount matches
            #
            testfit = numpy.array([0, 0, 0])
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

            results.append([cw, disp, n_good_line_matches, n_good_line_matches2, var_wl, testfit[0], testfit[1], testfit[2]])
            results_df.loc[i_iter, 'central_wavelength'] = cw
            results_df.loc[i_iter, 'dispersion'] = disp
            results_df.loc[i_iter, 'n_matches'] = n_good_line_matches
            results_df.loc[i_iter, 'n_matches_refined'] = n_good_line_matches2
            results_df.loc[i_iter, 'variance'] = var_wl
            for _i in range(testfit.shape[0]):
                results_df.loc[i_iter, 'polyfit_%d' % _i] = testfit[_i]

            # break

        self.logger.debug("Done exploring all combinations of central wavelength & dispersion")
        results = numpy.array(results)
        numpy.savetxt("results.dump", results)
        results_df.to_csv("wl_solution.results", index=False)

        i_most_matches = numpy.argmax(results[:, 3])
        #print(i_most_matches)
        #print("most matches", results[i_most_matches])

        all_best_matches = results[(results[:, 3] >= results[i_most_matches, 3])]

        i_smallest_scatter = numpy.argmin(all_best_matches[:, 4])
        best_solution = all_best_matches[i_smallest_scatter]
        self.logger.info("found best solution: %s" % (str(best_solution)))

        # Now that we have the best solution, do a final match and make some plots
        best_fit = best_solution[-3:]

        fit_order = 5

        wl_postfit = numpy.polyval(best_fit, peaks2d - central_y)
        full_wl = numpy.polyval(best_fit, full_y-central_y)
        d2, i2 = ref_tree.query(wl_postfit, k=1, p=1, distance_upper_bound=2)
        matched = (i2 < ref_tree.n)
        n_good_line_matches2 = numpy.sum(matched)
        ref_wl_refined = reflines[i2[matched]]

        self.matched_line_inventory = pandas.DataFrame()
        print("peaks:", peaks.shape)
        print("matched:", matched.shape)
        print("ref_wl_refined:", ref_wl_refined.shape)

        self.matched_line_inventory = pandas.DataFrame.from_dict({
            'comp_spectrum_pixel': peaks[matched],
            'reference_wl': ref_wl_refined
        },
            orient='columns',
        )

        comp_peaks_px = peaks[matched] - self.comp_spectrum_center_y
        use_in_final_fit = numpy.isfinite(ref_wl_refined)
        print("USE IN FIT", use_in_final_fit.shape)
        plot_fn = "wavelength_solution_details_initial.png"
        self.make_wavelength_calibration_overview_plot(spec, best_fit, plot_fn=plot_fn)
        for iteration in range(5):
            # now we have line positions in pixels and wavelength in A, let's fit
            polyfit = numpy.polyfit(x=comp_peaks_px[use_in_final_fit],
                                    y=ref_wl_refined[use_in_final_fit],
                                    deg=fit_order)
            fit_wl = numpy.polyval(polyfit, comp_peaks_px)

            delta_wl = ref_wl_refined - fit_wl
            stats = numpy.nanpercentile(delta_wl[use_in_final_fit], [16,50,84])
            _median = stats[1]
            _sigma = 0.5*(stats[2]-stats[0])
            outlier = (delta_wl < (_median-3*_sigma)) | (delta_wl > (_median+3*_sigma))
            use_in_final_fit[outlier] = False

            _pm = peaks[matched]
            _rem = ref_wl_refined
            print(_pm.shape, _rem.shape)
            plot_fn = "wavelength_solution_details_iteration%0d.png" % (iteration+1)
            self.make_wavelength_calibration_overview_plot(spec, polyfit, plot_fn=plot_fn, used_in_fit=use_in_final_fit)

        _pm = peaks[matched]
        _rem = ref_wl_refined
        print(_pm.shape, _rem.shape)

        self.logger.debug("Generating final reference plot")
        # fig, axs = plt.subplots(nrows=2, figsize=(12, 8))
        # #ax.scatter(wl_postfit[matched], ref_wl_refined)
        # axs[0].scatter(peaks[matched], ref_wl_refined)
        # axs[0].plot(full_y, full_wl, alpha=0.5)
        #
        # print(wl_postfit.shape, ref_wl_refined.shape)
        # axs[1].scatter(wl_postfit[matched], ref_wl_refined-wl_postfit[matched])
        # axs[1].axhline(y=0, ls=":")
        # fig.savefig("matched__wavelength_vs_pixel.png")
        # self.logger.debug("Plot saved to matched__wavelength_vs_pixel.png")
        plot_fn = "wavelength_solution_details_final.png"
        self.make_wavelength_calibration_overview_plot(spec, polyfit, plot_fn=plot_fn, used_in_fit=use_in_final_fit)

        #print(best_fit)
        #self.make_wavelength_calibration_overview_plot(spec, best_fit)#, used_in_fit=use_in_final_fit)
        return best_solution  # results[i_most_matches]

    def spec_scale(self, spec):
        scaled = numpy.sqrt(spec)
        scaled[spec <= 0] = 0
        return scaled

    def make_wavelength_calibration_overview_plot(self, comp_spectrum, wavelength_solution, plot_fn=None, used_in_fit=None):

        fig, axs = plt.subplots(figsize=(25, 10), nrows=3, tight_layout=True)

        wl_min = self.grating_solution.wl_blueedge
        wl_max = self.grating_solution.wl_rededge
        wl_diff = wl_max - wl_min
        wl_min -= 0.05*wl_diff
        wl_max += 0.05*wl_diff

        #
        # Compare/overlay reference and comp spectra
        #
        ref_line_amps = numpy.nanpercentile(self.ref_inventory['gauss_amp'], [16,50,84])
        typical_ref_line_amp = ref_line_amps[2]
        axs[0].plot(self.refspec_wavelength, self.spec_scale(self.refspec_smoothed / typical_ref_line_amp), lw=0.4, c='blue', label='ref')

        # disp = -0.30
        # cwl = 6568
        # spec_wl = spec_x0 * disp + cwl
        # wlpf = numpy.polyval(pf, spec_x0)


        comp_wl = numpy.polyval(wavelength_solution, self.comp_spectrum_full_y0)
        print("womp-wl:\n", comp_wl)
        comp_line_amps = numpy.nanpercentile(self.comp_line_inventory['gauss_amp'], [16,50,84])
        typical_comp_line_amp = comp_line_amps[2]
        self.logger.info("Spec scaling: ref:%f  comp:%f" % (typical_ref_line_amp, typical_comp_line_amp))
        axs[0].plot(comp_wl, self.spec_scale(self.comp_spectrum_continuumsub / typical_comp_line_amp), lw=0.4, c='orange', label='data')
        #
        peaks0 = self.comp_spectrum_lines - self.comp_spectrum_center_y
        peaks_wl = numpy.polyval(wavelength_solution, peaks0)
        # print(peaks_wl)
        axs[0].scatter(peaks_wl, numpy.ones_like(peaks0) * 0.5, marker="|", c='orange')
        #
        axs[0].scatter(self.ref_inventory['gauss_wl'], numpy.ones_like(self.ref_inventory['gauss_center']) * 0.6, c='blue', marker="|")
        # print(finelines['center'])
        #
        axs[0].set_xlim((wl_min, wl_max))
        axs[0].set_ylim((0, 1))
        axs[0].legend()
        axs[0].set_xlabel("wavelength [A]")
        axs[0].set_ylabel("normalized flux")

        #
        # Plot wavelength vs pixel coordinate
        #
        matched_comp_px_raw = self.matched_line_inventory['comp_spectrum_pixel']
        matched_comp_px = matched_comp_px_raw - self.comp_spectrum_center_y
        matched_ref_wl = self.matched_line_inventory['reference_wl']
        matched_comp_wl = numpy.polyval(wavelength_solution, matched_comp_px)
        if (used_in_fit is None):
            used_in_fit = numpy.isfinite(matched_comp_wl)
            print("Assuming all points were used_in_fit")
        comp_wl_full = numpy.polyval(wavelength_solution, self.comp_spectrum_full_y0)
        # peaks_wl = numpy.polyval(wavelength_solution, peaks0)

        #self.matched_line_inventory.info()

        axs[1].plot(comp_wl, self.comp_spectrum_full_y)
        axs[1].scatter(matched_ref_wl[used_in_fit], matched_comp_px_raw[used_in_fit], c='green')
        axs[1].scatter(matched_ref_wl[~used_in_fit], matched_comp_px_raw[~used_in_fit], c='red')
        axs[1].set_ylabel("pixel position [pixel]")
        axs[1].set_xlabel("wavelength [A]")
        axs[1].set_xlim((wl_min, wl_max))

        #
        # Deviations/residuals from perfect fit
        #
        #comp_pixel_0 = self.matched_line_inventory['comp_spectrum_pixel'] - self.comp_spectrum_center_y
        #comp_wl = numpy.polyval(wavelength_solution, comp_pixel_0)
        #ref_wl = self.matched_line_inventory['reference_wl']

        # print(final_spec_wl.shape)
        # print(ref_wl_refined.shape)
        #print("REF-WL", len(ref_wl.index))
        #print("COMP WL ", comp_wl.shape)
        #print("USED4FIT", used_in_fit.shape)
        axs[2].scatter(matched_ref_wl[used_in_fit], (matched_ref_wl - matched_comp_wl)[used_in_fit], c='green')
        fake_zero = numpy.zeros_like(matched_ref_wl)
        axs[2].scatter(matched_ref_wl[~used_in_fit], fake_zero[~used_in_fit], c='red', facecolors='none')
        # axs[2].scatter(ref_wl, ref_wl - comp_wl)
        axs[2].axhline(y=0)
        axs[2].set_xlabel("Wavelength [A]")
        axs[2].set_ylabel("Difference Reference - Calibrated [A]")
        axs[2].set_xlim((wl_min, wl_max))

        if (plot_fn is None):
            plot_fn = "wavelength_solution_details.png"
        fig.savefig(plot_fn)
        self.logger.debug("Plot saved to %s" % (plot_fn))


    def reidentify_lines(self, comp_spectra, ref_fiberid=41, make_plots=False):
        ##################################
        #
        #   RE-IDENTIFY
        #
        ##################################
        self.logger.info("starting to re-identify lines across frame")

        # cross-correlate in pixelspace to match curvature
        findlines_opt = dict(threshold=1000, distance=5)
        max_shift = 3
        full_y = numpy.arange(comp_spectra.shape[0])
        print("full_y shape", full_y.shape)
        center_y = full_y[full_y.shape[0] // 2]
        polydeg = 2
        centered_y = full_y - center_y
        # debug = False

        # ref_fiberid = 41
        ref_contsub, ref_peaks = self.find_lines(comp_spectra[ref_fiberid], **findlines_opt)
        ref_peaks -= center_y
        ref_tree = scipy.spatial.KDTree(data=numpy.array([ref_peaks, ref_peaks]).T)

        poly_transforms = [None] * 82
        poly_transforms[ref_fiberid] = [0., 1., 0.]

        for ranges in [numpy.arange(ref_fiberid + 1, self.n_fibers, 1),
                       numpy.arange(0, ref_fiberid)[::-1]]:
            poly_transform = poly_transforms[ref_fiberid]
            for fiberid in ranges:  # numpy.arange(ref_fiberid, 81, 1):
                # find lines in the new fiber
                contsub, new_peaks = self.find_lines(
                    comp_spectra[fiberid], **findlines_opt)
                new_peaks -= center_y

                # apply correction from the previous fiber trace
                matched2prev_peaks = numpy.polyval(poly_transform, new_peaks)

                # convert to 2d to make trees work
                np2 = numpy.array([matched2prev_peaks, matched2prev_peaks]).T

                # now match the rough-aligned peaks to the reference peaks
                d, i = ref_tree.query(np2, k=1, p=1, distance_upper_bound=max_shift)
                good_match = i < ref_tree.n
                print("fiber %d: ref=%d, this=%d, matched=%d" % (
                    fiberid, ref_tree.n, new_peaks.shape[0], numpy.sum(good_match)))

                new_pos = new_peaks[good_match]
                ref_pos = ref_peaks[i[good_match]]
                # print(new_pos.shape, ref_pos.shape)
                new_poly = numpy.polyfit(new_pos, ref_pos, deg=polydeg)
                print(new_poly)

                if (make_plots):
                    fig, ax = plt.subplots(nrows=2, figsize=(12, 5))
                    fig.suptitle("fiber %d (ref: %d)" % (fiberid, ref_fiberid))
                    ax[0].plot(centered_y, ref_contsub, lw=0.5, label="#%d / REF" % (ref_fiberid))
                    ax[0].plot(centered_y, contsub, lw=0.5, label="#%d" % (fiberid))
                    ax[0].scatter(centered_y, contsub, s=1, label="#%d" % (fiberid))
                    ax[0].legend()
                    ax[0].set_ylim((0, 1e4))
                    # ax[0].set_xlim((-120,20))

                    corrected_cy = numpy.polyval(new_poly, centered_y)
                    ax[1].plot(centered_y, ref_contsub, lw=0.5, label="#%d / REF" % (ref_fiberid))
                    ax[1].plot(corrected_cy, contsub, lw=0.5, label="#%d" % (fiberid))
                    ax[1].scatter(corrected_cy, contsub, s=1, label="#%d" % (fiberid))
                    ax[1].legend()
                    ax[1].set_ylim((0, 1e4))
                    # ax[1].set_xlim((-120,20))
                    ax[1].annotate("shift: %+.2f" % (new_poly[-1]), (0.05, 0.9), xycoords='axes fraction', ha='left',
                                   va='top')
                    fig.savefig("reidentify_fiber_%d.png" % (fiberid), dpi=300)
                    plt.close(fig)
                    print()

                poly_transform = new_poly  # numpy.polyfit(ref_pos, new_pos, deg=polydeg)
                poly_transforms[fiberid] = new_poly

            self.transform2d = numpy.zeros((82, 3))
            for i, t in enumerate(poly_transforms):
                if (t is not None):
                    self.transform2d[i, :] = t

            # poly_transforms = numpy.array(poly_transforms)

        if (make_plots):
            i = numpy.arange(self.transform2d.shape[0])
            fig, axs = plt.subplots(nrows=3, figsize=(7,5), sharex=True)

            axs[0].scatter(i, self.transform2d[:, -1], s=0.5)
            axs[0].set_ylabel("Offset dY")

            axs[1].scatter(i, self.transform2d[:, -2], s=0.5)
            axs[1].set_ylabel("relative dispersion")

            axs[2].scatter(i, self.transform2d[:, -3], s=0.5)
            axs[2].set_ylabel("quadratic term")
            axs[2].set_xlabel("Fiber ID")

            fig.savefig("reidentify_poly_transforms.png", dpi=300)

        return poly_transforms



    def map_2d_wavelength_solution(self, comp_image, traces):
        #
        # Now derive the full 2-d wavelength solution for the comp frames
        #
        fullmap_y = numpy.zeros_like(comp_image, dtype=float)
        # fullmap_y.shape
        full_y = numpy.arange(comp_image.shape[0])
        center_y = full_y[full_y.shape[0] // 2]
        centered_y = full_y - center_y

        center_x = self.master_comp.shape[1] // 2
        full_correction_per_fiber = numpy.array([
            numpy.polyval(self.transform2d[i, :], centered_y) for i in range(self.n_fibers)])
        print(full_correction_per_fiber.shape)

        iy,ix = numpy.indices(comp_image.shape, dtype=float)

        # print(ix.shape)
        centered_ix = ix - center_x
        print(traces.fullres_centers.shape)
        for y in full_y:  # [600:610]:
            centers = traces.fullres_centers[y, :] - center_x
            corrections = full_correction_per_fiber[:, y]
            pfy = numpy.polyfit(centers, corrections, deg=2)
            # print(pfy)
            fullmap_y[y, :] = numpy.polyval(pfy, centered_ix[y, :])

        # calculate actual wavelength for each point
        self.wavelength_mapping_2d = numpy.polyval(self.wavelength_solution, fullmap_y)

        # fig, ax = plt.subplots()
        # ax.imshow(fullmap_y)
        pyfits.PrimaryHDU(data=fullmap_y).writeto("full_ymapping.fits", overwrite=True)
        pyfits.PrimaryHDU(data=self.wavelength_mapping_2d).writeto("full_wlmapping.fits", overwrite=True)

        return self.wavelength_mapping_2d

    def rectify(self, image, poly_transforms, min_wl=None, max_wl=None, out_dispersion=0.2):

        inspec = image.copy()

        # prepare the final output wavelength grid
        if (min_wl is None):
            min_wl = numpy.nanmin(self.wavelength_mapping_2d)
        if (max_wl is None):
            max_wl = numpy.nanmax(self.wavelength_mapping_2d)

        # out_dispersion = 0.2
        # buffer = 5
        out_min_wl = out_dispersion * (numpy.floor(min_wl / out_dispersion))
        out_max_wl = out_dispersion * (numpy.ceil(max_wl / out_dispersion))
        self.logger.info("output min/max wavelength: %f // %f", out_min_wl, out_max_wl)

        n_wl_points = int(((max_wl - min_wl) / out_dispersion)) + 1
        # out_max_wl = out_dispersion * (numpy.ceil(max_wl/dispersion))
        out_wl_points = numpy.arange(n_wl_points, dtype=float) * out_dispersion + out_min_wl
        out_spectral_axis = out_wl_points * u.AA
        # print(out_spectral_axis)

        if (self.output_wl_min is None):
            self.output_wl_min = min_wl
        if (self.output_wl_max is None):
            self.output_wl_max = max_wl
        if (self.output_dispersion is None):
            self.output_dispersion = out_dispersion

        self.logger.info("Rectifying frame, output wavelength range is %.3f ... %.3f AA, dispersion %.3f A/px" % (
            min_wl, max_wl, out_dispersion))


        fluxcon = FluxConservingResampler(extrapolation_treatment='nan_fill')

        rectified_2d = numpy.zeros((n_wl_points, inspec.shape[1]))
        for x in range(self.wavelength_mapping_2d.shape[1]):
            # make sure the input is sorted
            wl_raw = self.wavelength_mapping_2d[:, x]
            wl_sort = numpy.argsort(wl_raw)
            wl_AA = wl_raw[wl_sort] * u.AA
            # print(wl_AA)
            flux = inspec[:, x][wl_sort] * u.DN
            spec = Spectrum1D(spectral_axis=wl_AA, flux=flux)
            rect_spec = fluxcon(spec, out_spectral_axis)
            rectified_2d[:, x] = rect_spec.flux.to(u.DN).value

        return rectified_2d

    def get_config(self, *args, fallback=None):
        config = self.config
        for opt in args:
            if opt not in config:
                return fallback
            config = config[opt]
        return config

    def calibrate(self, save=False):

        _master_bias_fn = "master_bias.fits" if save else None
        self.make_master_bias(save=_master_bias_fn)

        _master_flat_fn = "master_flat.fits" if save else None
        self.make_master_flat(save=_master_flat_fn)

        # return

        _master_comp_fn = "master_comp.fits" if save else None
        self.make_master_comp(save=_master_comp_fn)

        self.logger.info("Tracing fibers")
        # self.trace_fibers_raw(flat=self.master_flat)

        self.logger.info("Extracting fiber spectra from master flat")
        self.raw_traces = fibertraces.SparsepakFiberSpecs()
        self.raw_traces.find_trace_fibers(self.master_flat)
        # comp_spectra = raw_traces.extract_fiber_spectra(
        #     imgdata=self.master_comp,
        #     weights=self.master_flat,
        # )

        # self.flat_spectra = self.extract_spectra_raw(imgdata=self.master_flat, weights=self.master_flat)
        self.flat_spectra = self.raw_traces.extract_fiber_spectra(
            imgdata=self.master_flat, weights=self.master_flat)
        # print("flat_spectra.shape", self.flat_spectra.shape)
        # numpy.savetxt("flat_spectra2.dat", self.flat_spectra)

        self.logger.info("Extracting fiber spectra from master comp")
        # self.comp_spectra = self.extract_spectra_raw(imgdata=self.master_comp, weights=self.master_flat)
        self.comp_spectra = self.raw_traces.extract_fiber_spectra(
            imgdata=self.master_comp, weights=self.master_flat)
        # print(self.comp_spectra)
        numpy.savetxt("comp_spectra2.dat", self.comp_spectra)


        # self.read_reference_linelist()

        # extract lines for reference spectrum
        self.ref_fiberid = 41

        # find wavelength solution for one "reference" fiber
        _wavelength_solution = self.find_wavelength_solution(
            self.comp_spectra[self.ref_fiberid],
            make_plots=True
        )
        self.wavelength_solution = _wavelength_solution[-3:]
        # print("wavelength solution:", self.wavelength_solution)

        # Now re-identify lines across all other fiber traces
        self.poly_transforms = self.reidentify_lines(
            comp_spectra=self.comp_spectra,
            ref_fiberid=self.ref_fiberid,
            make_plots=False, #True
        )

        self.map_2d_wavelength_solution(comp_image=self.master_comp, traces=self.raw_traces)


        self.comp_rectified_2d = self.rectify(
            self.master_comp, self.poly_transforms,
            min_wl=self.get_config('output', 'min_wl', fallback=None),
            max_wl=self.get_config('output', 'max_wl', fallback=None),
            out_dispersion=self.get_config('output', 'dispersion', fallback=None)
        )

        phdu = pyfits.PrimaryHDU(data=self.comp_rectified_2d)
        # dispersion solution
        phdu.header['CRVAL2'] = self.output_wl_min * 1.e-10
        phdu.header['CRPIX2'] = 1.
        phdu.header['CD2_2'] = self.output_dispersion * 1.e-10
        phdu.header['CTYPE2'] = "WAVE-W2A"
        # fiber-id (approximate)
        phdu.header['CRVAL1'] = 1
        phdu.header['CRPIX1'] = self.raw_traces.get_mean_fiber_position(fiber_id=0)
        phdu.header['CD1_1'] = 1./self.raw_traces.get_fiber_spacing()
        phdu.header['CTYPE1'] = "FIBER-ID"
        self.logger.info("Writing rectified COMP spectrum")
        phdu.writeto("comp_rectified.fits", overwrite=True)

        # TODO HERE: create traces for the rectified spectra
        self.logger.info("Rectifing master flatfield for final extraction")
        self.flat_rectified_2d = self.rectify(
            self.master_flat, self.poly_transforms,
            min_wl=self.get_config('output', 'min_wl', fallback=None),
            max_wl=self.get_config('output', 'max_wl', fallback=None),
            out_dispersion=self.get_config('output', 'dispersion', fallback=None)
        )
        phdu = pyfits.PrimaryHDU(data=self.flat_rectified_2d)
        # dispersion solution
        phdu.header['CRVAL2'] = self.output_wl_min * 1.e-10
        phdu.header['CRPIX2'] = 1.
        phdu.header['CD2_2'] = self.output_dispersion * 1.e-10
        phdu.header['CTYPE2'] = "WAVE-W2A"
        # fiber-id (approximate)
        phdu.header['CRVAL1'] = 1
        phdu.header['CRPIX1'] = self.raw_traces.get_mean_fiber_position(fiber_id=0)
        phdu.header['CD1_1'] = 1./self.raw_traces.get_fiber_spacing()
        phdu.header['CTYPE1'] = "FIBER-ID"
        self.logger.info("Writing rectified COMP spectrum")
        phdu.writeto("flat_rectified.fits", overwrite=True)

        # Now prepare line-traces using the rectified master flatfield
        self.rect_traces = fibertraces.SparsepakFiberSpecs()
        self.rect_traces.find_trace_fibers(self.flat_rectified_2d)

        self.get_fiber_flatfields()

        sys.exit(0)


    def get_fiber_flatfields(self, filter_width=50):
        self.flat_fibers = self.rect_traces.extract_fiber_spectra(
            imgdata=self.flat_rectified_2d,
            weights=self.flat_rectified_2d,
        )

        wl = numpy.arange(self.flat_fibers.shape[1], dtype=float)
        pad_width = filter_width - (wl.shape[0] % filter_width)
        wl_padded = numpy.pad(wl, (0, pad_width),
                              mode='constant', constant_values=0)
        wl_padded[-pad_width:] = numpy.NaN
        rebinned_wl = numpy.nanmedian(wl_padded.reshape((-1, filter_width)), axis=1)

        fiber_flatfields = [None] * self.n_fibers
        self.fiber_flat_splines = [None] * self.n_fibers

        for fiber_id in range(self.n_fibers):
            # pick a fiber to work on
            fiberspec = self.flat_fibers[fiber_id]

            # make sure we can parcel out the full-res spectra into
            # chunks of a given width
            fiber_padded = numpy.pad(fiberspec, (0, pad_width),
                                     mode='constant', constant_values=0)
            fiber_padded[-pad_width:] = numpy.NaN
            n_good = numpy.isfinite(wl_padded) & numpy.isfinite(fiber_padded)

            # calculate the median flux in each little parcel of fluxes
            rebinned_spec = numpy.nanmedian(fiber_padded.reshape((-1, filter_width)), axis=1)
            rebinned_samples = numpy.nansum(n_good.astype(int).reshape((-1, filter_width)), axis=1)
            #     print(rebinned_samples)
            #     print(rebinned_wl)
            good = rebinned_samples > 0.8 * filter_width

            spline = scipy.interpolate.CubicSpline(
                x=rebinned_wl[good],
                y=rebinned_spec[good],
                bc_type='natural'
            )
            full_spline = spline(wl)

            self.fiber_flat_splines[fiber_id] = spline
            fiber_flatfields[fiber_id] = full_spline

        self.fiber_flatfields = numpy.array(fiber_flatfields)

    def apply_fiber_flatfields(self, fiberspecs):
        return fiberspecs / self.fiber_flatfields

    def write_rectified_spectrum(self, spec, output_filename):
        phdu = pyfits.PrimaryHDU(data=spec)
        # dispersion solution
        phdu.header['CRVAL2'] = self.output_wl_min * 1.e-10
        phdu.header['CRPIX2'] = 1.
        phdu.header['CD2_2'] = self.output_dispersion * 1.e-10
        phdu.header['CTYPE2'] = "WAVE-W2A"
        # fiber-id (approximate)
        phdu.header['CRVAL1'] = 1
        phdu.header['CRPIX1'] = self.raw_traces.get_mean_fiber_position(fiber_id=0)
        phdu.header['CD1_1'] = 1. / self.raw_traces.get_fiber_spacing()
        phdu.header['CTYPE1'] = "FIBER-ID"
        #out_fn = "%s_rect.fits" % (target_name)
        self.logger.info("Writing rectified spectrum to %s", output_filename)
        phdu.writeto(output_filename, overwrite=True)
        #

    def write_extracted_spectra(self, spec, output_filename):
        phdu = pyfits.PrimaryHDU(data=spec)
        # dispersion solution
        phdu.header['CRVAL1'] = self.output_wl_min * 1.e-10
        phdu.header['CRPIX1'] = 1.
        phdu.header['CD1_1'] = self.output_dispersion * 1.e-10
        phdu.header['CTYPE1'] = "WAVE-W2A"
        # fiber-id (approximate)
        phdu.header['CRVAL2'] = 1
        phdu.header['CRPIX2'] = 1.
        phdu.header['CD2_2'] = 1.
        phdu.header['CTYPE2'] = "FIBER-ID"
        #out_fn = "%s_rect.fits" % (target_name)
        self.logger.info("Writing rectified spectrum to %s", output_filename)
        phdu.writeto(output_filename, overwrite=True)
        #

    def reduce(self):

        for target_name in self.get_config('science'):
            self.logger.info("Starting reduction for target %s",  target_name)
            filelist = self.get_config(target_name)
            # make sure we always deal with lists, even if they only have one element
            if (not isinstance(filelist, list)):
                filelist = [filelist]
            # print(filelist)

            target_raw = self.basic_reduction(
                filelist=filelist,
                bias=self.master_bias,
                flat=None,
                op=numpy.nanmedian
            )
            target_rect = self.rectify(
                target_raw, self.poly_transforms,
                min_wl=self.get_config('output', 'min_wl', fallback=None),
                max_wl=self.get_config('output', 'max_wl', fallback=None),
                out_dispersion=self.get_config('output', 'dispersion', fallback=None)
            )
            self.write_rectified_spectrum(
                spec=target_rect,
                output_filename="%s__rectified.fits" % (target_name)
            )

            # extract all spectra for all fibers
            target_fiberspecs = self.rect_traces.extract_fiber_spectra(
                imgdata=target_rect,
                weights=self.flat_rectified_2d,
            )

            # apply flatfielding
            target_flatfielded = self.apply_fiber_flatfields(target_fiberspecs)
            self.write_extracted_spectra(
                spec=target_flatfielded,
                output_filename="%s__flatfielded.fits" % (target_name)
            )

            # prepare sky spectrum
            sky_fibers = numpy.array([22, 16, 2, 38, 54, 80, 70])
            sky_fiberids = self.n_fibers - sky_fibers
            sky = target_flatfielded[sky_fiberids]
            print(sky.shape)
            mean_sky = numpy.nanmedian(sky, axis=0)

            # subtract sky
            target_skysub = target_flatfielded - mean_sky
            self.write_extracted_spectra(
                spec=target_skysub,
                output_filename="%s__skysub.fits" % (target_name)
            )
            self.logger.info("done with target %s", target_name)

if __name__ == '__main__':

    #    logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(filename)-15s [ %(funcName)-30s ] :: %(message)s')

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

    benchspec.calibrate(save=True)
    # benchspec.reduce()

