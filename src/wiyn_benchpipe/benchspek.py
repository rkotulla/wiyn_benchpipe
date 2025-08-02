#!/usr/bin/env python

import argparse
import os

import numpy
import scipy.ndimage

import scipy
import scipy.signal
import scipy.ndimage
import scipy.interpolate
import scipy.optimize
import itertools
import multiparlog as mplog

from specutils import Spectrum1D
import astropy.units as u
from specutils.manipulation import FluxConservingResampler
import ccdproc

# from src.wiyn_benchpipe import fibertraces
from .fibertraces import *
#from .grating import Grating #, grating_from_header
from .spec_and_lines import SpecAndLines
from .config import Config
from .instruments import *
from .data import get_file

import warnings
#with warnings.catch_warnings():
#print("Disabling warnings")
warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
warnings.filterwarnings('ignore', r'divide by zero encountered in divide')
warnings.filterwarnings('ignore', r'invalid value encountered in divide')
warnings.simplefilter("ignore", RuntimeWarning)

def gauss(x, center, sigma, amplitude):
    return amplitude * numpy.exp(-(x - center) ** 2 / (2*sigma ** 2))
def normalized_gaussian(x, mu, sig):
    return 1.0 / (numpy.sqrt(2.0 * numpy.pi) * sig) * numpy.exp(-numpy.power((x - mu) / sig, 2.0) / 2)

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
    wl_polyfit_order = 5   # TODO: make this a user-tunable parameter

    fiber_inventories = None
    fiber_wavelength_solutions = None
    refspec_smoothed = None
    comp_spectrum_full_y = None
    comp_spectrum_full_y0 = None
    comp_spectrum_center_y = None
    ref_inventory = None
    grating_solution = None
    comp_line_inventory = None
    matched_line_inventory = None
    raw_traces = None
    fiber_inventories = None
    fiber_wavelength_solutions = None
    fiber_wavelength_solutions_inverse = None
    wavelength_solution = None

    def __init__(self, json_file, raw_dir=None, debug=False):
        self.logger = logging.getLogger('BenchSpek')
        self.debug = debug

        self.json_file = json_file
        self.read_config()
        if (raw_dir is not None and os.path.isdir(raw_dir)):
            self.raw_dir = raw_dir
        self.make_plots = str(self.config.get('plots')).lower() == "yes"
        if (self.make_plots):
            self.logger.info("Activating plot generation")

    def read_config(self):
        self.logger.info(self.json_file)
        # with open(self.json_file, "r") as f:
        #     self.config = json.load(f)
        # if (self.raw_dir is None):
        #     try:
        #         self.raw_dir = self.config['raw_directory']
        #     except:
        #         self.raw_dir = "."
        self.config = Config(self.json_file)

        # create all output directories
        self.config.create_output_directories()

        #
        #
        # # ensure output directories exist
        # if (not os.path.isdir(self.config['cals_directory'])):
        #     self.logger.info("Creating CALS directory: %s" % (self.config['cals_directory']))
        #     os.makedirs(self.config['cals_directory'])
        # else:
        #     self.logger.info("CALS directory (%s) exists" % (self.config['cals_directory']))
        # if (not os.path.isdir(self.config['out_directory'])):
        #     self.logger.info("Creating OUTPUT directory: %s" % (self.config['out_directory']))
        #     os.makedirs(self.config['output_directory'])
        # else:
        #     self.logger.info("OUTPUT directory (%s) already exists" % (self.config['out_directory']))

    def write_FITS(self, hdulist, filename, overwrite=True):
        self.logger.info("Writing FITS file (%s)" % (filename))
        hdulist.writeto(filename, overwrite=overwrite)

    def write_cals_FITS(self, hdulist, filename, **kwargs):
        full_fn = os.path.join(self.config.get('cals_directory'), filename)
        self.write_FITS(hdulist, full_fn, **kwargs)

    def write_results_FITS(self, hdulist, filename, **kwargs):
        full_fn = os.path.join(self.config.get('out_directory'), filename)
        self.write_FITS(hdulist, full_fn, **kwargs)

    def get_cosmic_ray_rejection_options(self, target_name):
        cosmics = self.config.get(target_name, "cosmics", "clean", fallback=None)
        if (cosmics is not None):
            if (type(cosmics) == str and cosmics.lower() == 'no'):
                return None
            cosmics = {}
            for opt,default in [('sigfrac', 0.4),
                        ('sigclip', 10),
                        ('niter', 4),
                        ]:
                value = self.config.get(target_name, 'cosmics', opt, fallback=default)
                cosmics[opt] = value
        return cosmics

    def basic_reduction(self, filelist, bias=None, flat=None,
                        op=numpy.mean, return_stack=False, cosmics=None, gain=True):
        _list = []
        header = None
        for fn in filelist:
            _fn = os.path.join(self.raw_dir, fn)
            if (not os.path.isfile(_fn)):
                self.logger.warning("Specified input file (%s) not found" % (_fn))
                continue
            try:
                data, _header = self.instrument.load_raw_file(_fn, logger=self.logger)
            except OSError:
                self.logger.critical("File %s can not be read" % (_fn))
                continue
            if (header is None):
                header = _header

            if (bias is not None):
                data -= bias
            if (cosmics is not None):
                # apply cosmic ray rejection
                self.logger.info("Cosmic Ray rejection (fn=%s, opts: %s)" % (fn, str(cosmics)))
                data, crmask = ccdproc.cosmicray_lacosmic(data, satlevel=2**18, verbose=False, **cosmics)
            if (flat is not None):
                data /= flat
            _list.append(data)

        if (_list):
            stack = numpy.array(_list)
            combined = op(stack, axis=0)
        else:
            stack = None
            combined = None
        if (return_stack):
            return combined, header, stack
        return combined, header

    def make_master_bias(self, save=None, *opts, **kwopts):
        self.logger.info("Creating master bias")
        self.master_bias, _, bias_stack = self.basic_reduction(
            filelist=self.config.get('bias'),
            bias=None, flat=None, op=numpy.median, return_stack=True, *opts, **kwopts)
        # print(self.master_bias.shape)
        # pyfits.PrimaryHDU(data=bias_stack).writeto("bias_stack.fits", overwrite=True)
        if (self.master_bias is None):
            self.logger.warning("No BIAS created, skipping bias correction")
        elif (save is not None):
            self.logger.info("Writing master bias to %s", save)
            self.write_cals_FITS(pyfits.PrimaryHDU(data=self.master_bias), filename=save)
            # pyfits.PrimaryHDU(data=self.master_bias).writeto(save, overwrite=True)

        rdnoise = self.config.get("readnoise", fallback='auto')
        if (rdnoise != 'auto'):
            self.readnoise_adu = rdnoise
            self.readnoise_adu_std = 0
        elif (bias_stack.shape[0] < 2):
            self.logger.warning("Need at least 2 bias frames to compute readnoise")
            self.readnoise_adu = 10
        else:
            self.logger.debug("Finding readnoise")
            readnoise2d = numpy.std(bias_stack, axis=0)
            good = numpy.isfinite(readnoise2d)
            for i in range(3):
                _stats = numpy.nanpercentile(readnoise2d[good], [16,50,84])
                _med = _stats[1]
                _sigma = 0.5*(_stats[2]-_stats[0])
                good = good & (readnoise2d > (_med-3*_sigma)) & (readnoise2d < (_med+3*_sigma))
            self.readnoise_adu = _med
            self.readnoise_adu_std = _sigma
            self.logger.info("Found Readnoise of %.3f +/- %.3f counts" % (
                self.readnoise_adu, self.readnoise_adu_std))

            if (self.make_plots):
                self.logger.debug("Generating readnoise plot")
                fig, ax = plt.subplots(figsize=(8,6), tight_layout=True)
                ax.set_xlim((numpy.max([0,self.readnoise_adu-4*self.readnoise_adu_std]),
                             self.readnoise_adu+8*self.readnoise_adu_std))
                ax.set_xlabel("Readnoise [ADU]")
                ax.set_ylabel("# pixels")
                bins = numpy.linspace(0,40,800)
                bincenters = bins + 0.5*(bins[2]-bins[1])
                ax.hist(readnoise2d.ravel(), bins=bins, alpha=0.7)
                ax.axvline(x=self.readnoise_adu,c='red', label="Median: %.3f ADU" % (self.readnoise_adu))
                n_pixels = readnoise2d.size
                ax.set_yscale('log')
                ax.set_ylim((0.5, n_pixels * (bins[1]-bins[0])))
                ax.plot(bincenters,
                        n_pixels * (bins[1]-bins[0]) * normalized_gaussian(bincenters, self.readnoise_adu, self.readnoise_adu_std),
                        label=r"$\sigma$ = %.3f" % (self.readnoise_adu_std))
                ax.legend(loc='upper right')
                ax.set_title("Readnoise (#frames: %d)" % (bias_stack.shape[0]))
                __fn = "readnoise_distribution.png"
                self.logger.debug("Saving readnoise plot to %s" % (__fn))
                fig.savefig(__fn, dpi=300)

    def make_master_flat(self, save=None, *opts, **kwopts):
        self.logger.info("Creating master flat")
        # _list = []
        self.master_flat, _, flat_stack = self.basic_reduction(
            filelist=self.config.get('flat'),
            bias=self.master_bias, flat=None, op=numpy.median,
            return_stack=True, *opts, **kwopts)
        pyfits.PrimaryHDU(data=self.master_flat).writeto("masterflat_raw.fits", overwrite=True)

        #
        # for fn in self.config.get('flat'):
        #     _fn = os.path.join(self.raw_dir, fn)
        #     hdulist = pyfits.open(_fn)
        #     data = hdulist[0].data.astype(float)
        #     if (self.master_bias is not None):
        #         data -= self.master_bias
        #     _list.append(data)
        # stack = numpy.array(_list)
        # self.master_flat = numpy.median(stack, axis=0)
        # print(self.master_flat.shape)
        # if (save is not None):
        #     self.logger.info("Writing master flat to %s", save)
        #     pyfits.PrimaryHDU(data=self.master_flat).writeto(save, overwrite=True)

        # normalize flat-field
        avg_flat = numpy.nanmedian(self.master_flat, axis=1).reshape((-1,1))
        cleaned_flat = scipy.ndimage.median_filter(avg_flat, size=11, mode='nearest')

        # print(avg_flat.shape)
        self.master_flat /= cleaned_flat
        if (self.debug): numpy.savetxt("masterflat_norm.txt", avg_flat)
        if (save is not None):
            self.logger.info("Writing master flat to %s", save)
            self.write_cals_FITS(pyfits.PrimaryHDU(data=self.master_flat), filename=save)

        gain = self.config.get("gain", fallback='auto')
        if (gain != 'auto'):
            self.gain = gain
        elif (flat_stack.shape[0] < 3):
            self.logger.warning(
                "Unable to compute gain from only %d flats, need at least 3 (more is better)" % (flat_stack.shape[0]))
            self.gain = 0.5
            # need at least three flatfields to generate noise across 2 pairs; more is better!
        else:
            # need at least three flatfields to generate noise across 2 pairs; more is better!
            self.logger.info("Computing gain from flatfields")
            # generate mean flatfield levels in pairs of flatfields
            mean_flat = numpy.mean((flat_stack[1:, :, :] + flat_stack[:-1, :, :]) / 2., axis=0)
            # generate pairwise differences in flatfields (to take out any pixel variations), then compute the noise
            # in these difference flats
            dflat = numpy.diff(flat_stack, axis=0)
            std_dflat = numpy.std(dflat, axis=0)

            # define the fitting functions for two scenarios, with readnoise fixed or variable
            def _noise_var_readnoise(p, f):
                gain = p[0]
                rne = p[1]
                return numpy.sqrt( f*gain + 2*(rne*gain)**2 ) / gain
            def _error_var_readnoise(p,f,n):
                pred = _noise_var_readnoise(p,f)
                delta = pred - n
                return delta
            def _noise_fixed_readnoise(p, f, rdnoise):
                gain = p[0]
                # need readnoise x 2; we are dealing with differences between flats (so 2 reads)
                return numpy.sqrt( f*gain + 2*(rdnoise*gain)**2 ) / gain
            def _error_fixed_readnoise(p,f,n,rdnoise):
                pred = _noise_fixed_readnoise(p, f, rdnoise)
                delta = pred - n
                return delta

            # select good data
            good = (mean_flat < 100000)
            good_flux = mean_flat[good]
            good_noise = std_dflat[good]

            # One fit, using readnoise as a free parameter
            _fitresults_var_readnoise = scipy.optimize.leastsq(
                func=_error_var_readnoise, x0=[0.5, 10], args=(good_flux, good_noise),
                full_output=True)
            best_fit_var_readnoise = _fitresults_var_readnoise[0]
            uncert_var = numpy.diag(numpy.sqrt(_fitresults_var_readnoise[1]))
            # print(uncert_var)

            # second fit, using a fixed readnoise taken from the biases
            _fitresults_fixed_rdnoise = scipy.optimize.leastsq(
                func=_error_fixed_readnoise, x0=[0.5], args=(good_flux, good_noise, self.readnoise_adu),
                full_output=True)
            # print("\nFit w fixed readnoise")
            best_fit_fixed_readnoise = _fitresults_fixed_rdnoise[0]
            uncert = numpy.diag(numpy.sqrt(_fitresults_fixed_rdnoise[1]))

            self.gain = best_fit_var_readnoise[0]

            if (self.make_plots):
                i = numpy.logspace(0, 5.3, 100)

                fig, ax = plt.subplots(figsize=(12,7), tight_layout=True)
                ax.set_xlabel("flux [ADU]")
                ax.set_ylabel("noise (combined photon-noise + readnoise) [ADU]")

                # show a subset of data
                ev=10
                ax.scatter(mean_flat[::ev], std_dflat[::ev], s=1, alpha=0.05)

                # overplot the fit with readnoise as parameter
                ax.plot(i, _noise_var_readnoise(best_fit_var_readnoise, i), lw=3, alpha=0.5, c='red',
                        label=r"Gain: %.3f e$^-$/ADU" "\n" r"Readnoise [fit]: %.3f ADU = %.3f e$^-$" % (
                           best_fit_var_readnoise[0], best_fit_var_readnoise[1], best_fit_var_readnoise[1]*best_fit_var_readnoise[0]
                       ))

                # 2nd overplot, fit using fixed readnoise
                ax.plot(i, _noise_fixed_readnoise(best_fit_fixed_readnoise, i, self.readnoise_adu),
                        lw=3, alpha=0.5, c='green',
                        label=r"Gain: %.3f e$^-$/ADU" "\n" r"Readnoise [fixed]: %.3f ADU = %.3f e$^-$" % (
                           best_fit_fixed_readnoise[0], self.readnoise_adu, self.readnoise_adu * best_fit_fixed_readnoise[0]
                       ))

                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.grid(which='major', c='k', alpha=0.4)
                ax.grid(which='minor', c='grey', alpha=0.1)
                ax.legend(loc='upper left')

                self.logger.info("Results var.  readnoise: Readnoise: %.3f ADU = %.3f e- // Gain: %.3f e-/ADU" % (
                    best_fit_var_readnoise[1], best_fit_var_readnoise[1]*best_fit_var_readnoise[0],  best_fit_var_readnoise[0]))
                self.logger.info("Results fixed readnoise: Readnoise: %.3f ADU = %.3f e- // Gain: %.3f e-/ADU" % (
                    self.readnoise_adu, self.readnoise_adu * best_fit_fixed_readnoise[0], best_fit_fixed_readnoise[0]))

                _plot_fn = "gain_derivation.png"
                fig.savefig(_plot_fn)

        self.readnoise_electrons = self.readnoise_adu * self.gain

    def make_master_comp(self, save=None, *opts, **kwopts):
        self.logger.info("Creating master comp")
        self.master_comp, self.comp_header = self.basic_reduction(
            filelist=self.config.get('comp'),
            bias=self.master_bias, flat=self.master_flat,
            op=numpy.median,
            *opts, **kwopts
        )
        self.logger.debug("MasterComp dimensions: %s" % (str(self.master_comp.shape)))
        # print(self.master_comp.shape)
        # print(self.comp_header)
        if (save is not None):
            self.logger.info("Writing master comp to %s", save)
            pyfits.PrimaryHDU(data=self.master_comp, header=self.comp_header).writeto(save, overwrite=True)


    def read_reference_linelist(self):
        opt = self.config.get('linelistx')
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


    def get_refined_lines_from_spectrum(self, spec, distance=5, window_size=8,
                                        filter=True, return_contsub=False, min_threshold=None):
        # find a background approximation for the spectrum
        spec = spec.copy()
        spec[(spec < 0) | ~numpy.isfinite(spec)] = 0 # Clean up data
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
        _thresholds = numpy.array([1.5]) #[10,5,3,2,1])
        if (min_threshold is not None and False): # TODO: Fix // disabling this for now
            _thresholds = _thresholds[_thresholds >= min_threshold]
        thresholds = _thresholds * _sigma
        # thresholds = numpy.array([20,5,2]) * _sigma

        min_pixels = 3 # required minimum # of samples above threshold

        thresh = _sigma * 1.5

        pad1 = numpy.pad(contsub, (1,1))
        diff_left = pad1[1:-1] - pad1[:-2] #numpy.diff(pad1)[:-1]
        diff_right = pad1[2:] - pad1[1:-1] #((numpy.diff(pad1[::-1])[:-1])[::-1]
        # print(pad1.shape, contsub.shape, diff_left.shape, diff_right.shape)

        #        end_of_line_left  = not_significant | (diff_left < 0)  #| (diff_left > 0)
        #        end_of_line_right = not_significant | (diff_right > 0) #| (diff_right > 0)

        for iteration, threshold in enumerate(thresholds):  # range(5):
            self.logger.debug("Finding lines: Iteration %d, threshold: %.2f * sigma = %f" % (
                    iteration+1, _thresholds[iteration], threshold))

            significant = contsub >= threshold
            not_significant = contsub < threshold
            end_of_line_left  = not_significant | (diff_left < 0)  #| (diff_left > 0)
            end_of_line_right = not_significant | (diff_right > 0) #| (diff_right > 0)

            # self.logger.debug("\n"+"*" * 25 + "\n\n   ITERATION %d\n\n" % (iteration + 1) + "*" * 25)
            added_new_line = False

            gf = numpy.zeros_like(contsub, dtype=float)
            diffslopes_left = numpy.pad(numpy.diff(contsub), (0, 1), mode='constant', constant_values=0)
            diffslopes_right = numpy.pad(numpy.diff(contsub), (1, 0), mode='constant', constant_values=0)

            peaks, peak_props = scipy.signal.find_peaks(contsub, height=threshold, distance=distance)
            self.logger.debug("Found %d lines with peak > %.2f" % (len(peaks), threshold))

            # fig, ax = plt.subplots(figsize=(25, 5))

            next_cs = contsub.copy()
            for i, line in enumerate(peaks):

                # check if we already have a line at this position; only do this for valid lines
                if (len(line_inventory.index) > 0):
                    good_fit = numpy.isfinite(line_inventory['gauss_center']) & numpy.isfinite(line_inventory['gauss_width'])
                    good_lines = line_inventory[good_fit]
                    if (len(good_lines.index) > 0):
                        gc = good_lines['gauss_center'].to_numpy()
                        gw = good_lines['gauss_width'].to_numpy()
                        gw[gw > distance] = distance
                        match = (line > (gc - width_buffer * gw)) & (line < (gc + width_buffer * gw))
                        # print(match)
                        if (numpy.sum(match) > 0):
                            #self.logger.debug("already found a line at this approximate position %d" % (line))
                            continue
                    else:
                        #self.logger.debug("No counterpart found @ %d, continuing" % (line))
                        pass
                else:
                    # this is the first line, so we for sure don't know about it yet
                    pass

                added_new_line = True

                p = line
                try:
                    _leftedge = numpy.max(x[:p][end_of_line_left[:p]]+1)
                except:
                    _leftedge = p-window_size
                try:
                    _rightedge = numpy.min(x[p:][end_of_line_right[p:]])
                except:
                    _rightedge = p+window_size

                _left = numpy.nanmax([
                    0,                                        # left edge of data
                    p - window_size,                          # left edge of window size
                    _leftedge               # either up-turn or no longer significant
                ])
                _right = numpy.nanmin([
                    spec.shape[0] - 1,
                    p + window_size,
                    _rightedge,
                ])
                # int(numpy.ceil(line + window_size)), ])
                #
                #                    int(numpy.floor(line - window_size))])
                # _right = numpy.min([int(numpy.ceil(line + window_size)), spec.shape[0] - 1])
                #
                # try:
                #     _left2 = numpy.max([_left, numpy.max(x[(x < line) & (diffslopes_left < 0)]) + 1])
                # except ValueError:
                #     _left2 = _left
                #
                # try:
                #     _right2 = numpy.min([_right, numpy.min(x[(x > line) & (diffslopes_right > 0)]) - 1])
                # except ValueError:
                #     _right2 = _right
                # print(_left, _right, "-->", _left2, _right2)

                # generate a flux-weighted mean position as starting guess for the following gauss fit
                raw_width = _right - _left

                w_x = x[_left:_right + 1]
                w_spec = spec[_left:_right + 1]
                weighted = numpy.sum(w_x * w_spec) / numpy.sum(w_spec)
                # print(line, weighted)
                peak_flux = contsub[line]

                idx = len(line_inventory.index) + 1
                line_inventory.loc[idx, 'position'] = line
                line_inventory.loc[idx, 'peak'] = peak_flux #contsub[line]
                line_inventory.loc[idx, 'center_weight'] = weighted
                line_inventory.loc[idx, 'iteration'] = iteration
                line_inventory.loc[idx, 'threshold'] = threshold
                line_inventory.loc[idx, 'left'] = _left
                line_inventory.loc[idx, 'right'] = _right
                line_inventory.loc[idx, 'raw_width'] = raw_width
                line_inventory.loc[idx, 'fake_x'] = 500
                line_inventory.loc[idx, 'gauss_center'] = numpy.nan
                line_inventory.loc[idx, 'gauss_width'] = numpy.nan
                line_inventory.loc[idx, 'gauss_amp'] = numpy.nan


                if (raw_width < min_pixels):
                    # No need to spend more time on a line we'll exclude throw out later anyway,
                    # plus, fitting a gaussian to a single/few pixels doesn't make much sense
                    continue



                full_fit_results = scipy.optimize.leastsq(
                    func=_fit_gauss,
                    x0=[weighted, 3, peak_flux], ## position,width,amplitude
                    args=(w_x, w_spec)
                )
                gaussfit = full_fit_results[0]
                # print(line, gaussfit)

                _fine_x = fine_x[_left * supersample:_right * supersample + 1]
                modelgauss = gauss(w_x, gaussfit[0], gaussfit[1], gaussfit[2])
                next_cs[_left:_right + 1] -= modelgauss
                gf[_left:_right + 1] += modelgauss
                # centers.append(gaussfit[0])

                line_inventory.loc[idx, 'gauss_center'] = gaussfit[0]
                line_inventory.loc[idx, 'gauss_width'] = numpy.fabs(gaussfit[1])
                line_inventory.loc[idx, 'gauss_amp'] = gaussfit[2]

                # ax.scatter(x[_left2:_right2+1], cs[_left2:_right2+1], marker=markers[i%4], alpha=0.5)

                # fine_gauss = gauss(_fine_x, gaussfit[0], gaussfit[1], gaussfit[2])
                # ax.plot(_fine_x, fine_gauss, lw=1.2, ls=":", c='black')


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

        if (self.debug): line_inventory.to_csv("line_inventory_before_filtering.csv", index=False)
        if (filter):
            # apply some filtering:
            self.logger.debug("Applying filtering [start: %d]" % (len(line_inventory.index)))

            # require the gauss-fits to have similar peaks to the actual data
            amp_ratio = line_inventory['peak'] / line_inventory['gauss_amp']
            good_amps = (amp_ratio > 0.5) &  (amp_ratio < 1.5)                    # gauss is a reasonable fit
            good_width = (line_inventory['raw_width'] > min_pixels)         # minimum number of pixels detected
            good_fit = (numpy.isfinite(line_inventory['gauss_center']))   # has a valid gauss fit
            self.logger.debug("Good... amps: %d // width: %d // fit: %d" % (
                numpy.sum(good_amps), numpy.sum(good_width), numpy.sum(good_fit)))
            good = good_width & good_fit
            self.logger.debug("Good @ start: %d" % (numpy.sum(good)))

            # only select lines with "typical" line widths
            gw = line_inventory['gauss_width'].to_numpy()
            for i in range(3):
                _stats = numpy.nanpercentile(gw[good], [16, 50, 84])
                _med = _stats[1]
                _sig = 0.5 * (_stats[2] - _stats[0])
                self.logger.debug("Stats: med: %f, sigma: %f" % (_med, _sig))
                if (_sig <= 0):
                    break
                good = good & (gw >= (_med - 3 * _sig)) & (gw <= (_med + 3 * _sig))
                self.logger.debug("Good after iteration %d: %d" % (i+1, numpy.sum(good)))
            line_inventory = line_inventory[good]

        # line_inventory.to_csv("line_inventory.csv", index=False)
        if (return_contsub):
            contsub = spec - cont
            return line_inventory, contsub
        return line_inventory

    def find_reflines_from_csv(self, ref_spec_fn, wl_min, wl_max):
        self.logger.info("Loading reference lines from CSV file (%s)" % (ref_spec_fn))

        data = pandas.read_csv(ref_spec_fn, sep=',', comment='#')
        self.logger.info("Raw column names: %s" % (", ".join(list(data.columns))))

        # Rename the first two columns, and add other required columns as needed
        output_columns = ['gauss_wl', 'gauss_amp'] + list(data.columns[2:])
        data.columns = output_columns

        required_columns = ['gauss_width']
        for rc in required_columns:
            if (rc not in data.columns):
                data[rc] = numpy.nan

        self.logger.info("Corrected column names: %s" % (", ".join(list(data.columns))))
        self.logger.info("Read %d lines between wavelength %.2f ... %.2f" % (
            len(data), data['gauss_wl'].min(), data['gauss_wl'].max()))

        keepers = numpy.isfinite(data['gauss_wl'])
        if (wl_min is not None):
            keepers[data['gauss_wl'] < wl_min] = False
        if (wl_max is not None):
            keepers[data['gauss_wl'] > wl_max] = False
        self.ref_inventory = data[keepers]
        self.logger.info("Pre-selected line list has %d lines (%.1f ... %.1f)" % (
            len(self.ref_inventory), self.ref_inventory['gauss_wl'].min(), self.ref_inventory['gauss_wl'].max()))

    def find_reflines(self, ref_spec_fn=None, sci_sigma=None, wl_min=None, wl_max=None):

        if (ref_spec_fn is None):
            ref_spec_fn = self.config.get('linelist') # "scidoc2212.fits"
        if (not os.path.isfile(ref_spec_fn)):
            self.logger.info("Linelist file (%s) not found in current directory, trying data repository instead" % (ref_spec_fn))
            _fn = get_file(ref_spec_fn)
            if (not os.path.isfile(_fn)):
                self.logger.error("Unable to locate linelist file (checked %s and %s)" % (
                    ref_spec_fn, _fn
                ))
                return None
            else:
                ref_spec_fn = _fn

        self.logger.info("Processing linelists from %s" % (ref_spec_fn))
        _, ext = os.path.splitext(ref_spec_fn)
        if (ext in ['.csv', '.dat', '.txt']):
            # load using pandas
            self.logger.info("Loading linelist catalog")
            return self.find_reflines_from_csv(ref_spec_fn, wl_min, wl_max)
        elif (ref_spec_fn.endswith(".fits")):
            # assume this is a spectrum file
            return self.find_reflines_from_spec(ref_spec_fn, sci_sigma, wl_min, wl_max)
        else:
            self.logger.error("Unable to identify how to process linelist in %s" % (ref_spec_fn))
            return None




    def find_reflines_from_spec(self, ref_spec_fn=None, sci_sigma=None, wl_min=None, wl_max=None):

        if (ref_spec_fn is None):
            ref_spec_fn = self.config.get('linelist') # "scidoc2212.fits"
        self.logger.debug("Reading wavelength reference spectrum from %s" % (ref_spec_fn))
        hdu = pyfits.open(ref_spec_fn)
        s = hdu[0].data
        # hdu.info()
        # hdu[0].header
        fig, ax = plt.subplots(figsize=(13, 4))
        _x = numpy.arange(s.shape[0], dtype=float) + 1.
        _l = (_x - hdu[0].header['CRPIX1']) * hdu[0].header['CD1_1'] + hdu[0].header['CRVAL1']

        test_inventory, contsub = self.get_refined_lines_from_spectrum(
            spec=s, return_contsub=True, min_threshold=10)

        # threshold = 5000
        # contsub, reflines = self.find_lines(s, threshold=threshold, distance=10)
        # # print(reflines)
        # # refpeaks, props = scipy.signal.find_peaks(s, height=5000, distance=30)
        # reflines = ref_inventory['gauss_center'].to_numpy()
        # refpeaks_wl = (reflines - hdu[0].header['CRPIX1']) * hdu[0].header['CD1_1'] + hdu[0].header['CRVAL1']
        test_inventory['gauss_wl'] = (test_inventory['gauss_center'] - hdu[0].header['CRPIX1']) * hdu[0].header['CD1_1'] + hdu[0].header['CRVAL1']
        if (self.debug): test_inventory.to_csv("reflines_inventory.csv", index=False)

        # trim down lines to approximately the observed range
        keepers = numpy.isfinite(test_inventory['gauss_wl'])
        if (wl_min is not None):
            keepers[test_inventory['gauss_wl'] < wl_min] = False
        if (wl_max is not None):
            keepers[test_inventory['gauss_wl'] > wl_max] = False
        test_inventory = test_inventory[keepers]

        self.refspec_raw = hdu[0].data
        self.refspec_continnum_subtracted = contsub
        self.refspec_wavelength = _l

        # widths, _, _, _ = scipy.signal.peak_widths(contsub, reflines)
        widths = test_inventory['gauss_width'].to_numpy()
        reflines = test_inventory['gauss_center'].to_numpy()
        # print(linewidths)
        good_width = numpy.isfinite(widths) # & (reflines > 6300) & (reflines < 6900)
        for _iter in range(3):
            stats = numpy.nanpercentile(widths[good_width], [16, 50, 84])
            _med = stats[1]
            _sigma = 0.5 * (stats[2] - stats[0])
            good_width = good_width & (widths < (_med + 3 * _sigma)) & (widths > (_med - 3 * _sigma))
        med_ref_width = numpy.nanmedian(widths[good_width])
        ref_dispersion = hdu[0].header['CD1_1']
        ref_sigma = med_ref_width * ref_dispersion #/ 2.634
        self.logger.debug("reference spectrum line width: %.4f pixels ==> %.4f AA" % (
            med_ref_width, ref_sigma))
        # self.logger.debug("reference spectrum line width: %.4f AA" % )

        sci_dispersion = 0.31
        med_width = 3.3
        if (sci_sigma is None):
            sci_sigma = (med_width * sci_dispersion / 2.634)
        self.logger.debug("Comparing instrumental resolutions: data:%fAA reference:%fAA" % (sci_sigma, ref_sigma))

        if (sci_sigma <= ref_sigma):
            smooth_sigma = None
            smooth_px_width = None
            self.logger.debug("Data is higher resolution than reference, no reference smoothing needed")
            smoothed = contsub.copy()
        else:
            smooth_sigma = numpy.sqrt(sci_sigma ** 2 - ref_sigma ** 2)
            smooth_px_sigma = smooth_sigma / ref_dispersion / 2. # TODO: take out /2. fudge factor
            self.logger.debug("smoothing needed: sigma=%fAA ==> %.2fpx" % (smooth_sigma, smooth_px_sigma))
            smoothed = scipy.ndimage.gaussian_filter1d(contsub, sigma=smooth_px_sigma)
        if (self.debug):
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
        ax.set_xlim((6400, 6900))
        # TODO: automatically adjust range to match actual covered range, plus some margins

        ax.set_ylim((0, 5e4))

        # sel_wl = refpeaks_wl[(refpeaks_wl > 6350) & (refpeaks_wl < 6480) ]
        # TODO: fix
        threshold = 5000
        sel_wl = test_inventory['gauss_wl']
        ax.scatter(sel_wl, test_inventory['gauss_amp'], marker="|", label="lines") # (thr=%g)" % (threshold))
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

        #
        # Repeat line extraction etc, now that we have a resolution-matched reference spectrum
        #
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

    def find_wavelength_solution(self, spec, lambda_central=None, dispersion=None, min_lines=3, make_plots=True, n_brightest=0):

        self.logger.info("Finding wavelength solution")

        # if (lambda_central is None):
        #     lambda_central = self.config.get('setup', 'central_wavelength')
        # if (dispersion is None):
        #     dispersion = self.config.get('setup', 'dispersion')

        # print("spec shape:", spec.shape)
        # self.logger.info("User solution: central wavelength: %.3f; dispersion: %.4f" % (lambda_central, dispersion))

        # self.logger.info("Getting approximate wavelength solution from grating setup")
        self.grating_solution = self.raw_traces.grating_from_header(self.comp_header)
        self.grating_solution.report()
        # self.logger.info("GRATING: solution: %s" % (self.grating_solution.wl_polyfit))

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
        self.comp_line_inventory, contsub = self.get_refined_lines_from_spectrum(
            spec, return_contsub=True, min_threshold=5, distance=20, window_size=30)

        if (n_brightest > 0):
            self.logger.info("Restricting comp line list to brightest %d lines" % (n_brightest))
            amp_sorted = numpy.argsort(self.comp_line_inventory['gauss_amp'])[::-1]
            self.comp_line_inventory = self.comp_line_inventory.iloc[amp_sorted[:n_brightest], :]
        if (self.debug or True):
            numpy.savetxt("ref_comp_spec", spec)
            numpy.savetxt("ref_comp_spec_contsub", contsub)
            self.comp_line_inventory.to_csv("inventory_comp.csv", index=False)
        peaks = self.comp_line_inventory['gauss_center'].to_numpy()
        self.comp_spectrum_raw = spec
        self.comp_spectrum_continuumsub = contsub
        self.comp_spectrum_lines = peaks

        # find typical linewidth
        line_width_px = numpy.nanmedian(self.comp_line_inventory['gauss_width'])
        line_width_AA = numpy.fabs(line_width_px * dispersion)
        self.logger.debug("Typical line width in comp spectrum: %.3fpx -> %.4fAA" % (
            line_width_px, line_width_AA
        ))

        # now extract reference lines, after matching resolution to that of the
        # data we are about to calibrate
        wl_padding = 0.05*(self.grating_solution.wl_rededge - self.grating_solution.wl_blueedge)
        self.find_reflines(
            sci_sigma=line_width_AA,
            wl_min=(self.grating_solution.wl_blueedge - wl_padding),
            wl_max=(self.grating_solution.wl_rededge + wl_padding)
        )
        if (self.debug):
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

        # generate a tree for the reference lines
        in_window = (self.ref_inventory['gauss_wl'] >= self.grating_solution.wl_blueedge) & \
                    (self.ref_inventory['gauss_wl'] <= self.grating_solution.wl_rededge)
        selected_list = self.ref_inventory[in_window].reset_index(drop=True)
        # selected_list.info()
        reflines = selected_list['gauss_wl'].to_numpy()
        self.logger.info("Found %s calibrated reference lines" % (str(reflines.shape)))

        if (make_plots):
            self.logger.debug("Generating initial reference spectrum comparison")
            fig, ax = plt.subplots(figsize=(13, 5), tight_layout=True)
            ax.plot(full_y, contsub, lw=0.5)
            # ax.plot(wl, contsub, lw=0.5)
            peak85 = numpy.nanpercentile(self.comp_line_inventory['gauss_amp'].to_numpy(), 85)
            ymax = 1.1 * peak85
            ymin = -0.02 * peak85
            y1 = 0.9 * ymax
            y2 = 0.8 * ymax
            for p in peaks:
                ax.axvline(x=p, ymin=0.0, ymax=0.9, lw=0.2, color='red', alpha=0.5)
                ax.text(p, y1, "%d" % (p), rotation='vertical', ha='center', fontsize='small')
            #     if (reflines is not None):
            #         ax.scatter(reflines, numpy.ones_like(reflines)*2500, marker="|")
            # ax.set_yscale('log')
            ax.set_ylim(ymin, ymax)
            ax.set_xlabel("comp spectrum position [pixel]")
            ax.set_ylabel("comp spectrum amplitude")
            fig.savefig("reference_spectrum.png", dpi=300)
            self.logger.debug("Saved plot to reference_spectrum.png")


            self.logger.debug("Generating initial reference spectrum comparison [wavelength]")
            fig, ax = plt.subplots(figsize=(13, 5), tight_layout=True)
            ax.plot(full_wl, contsub, lw=0.5)
            # ylabelpos = numpy.nanpercentile(self.comp_line_inventory['gauss_center'].to_numpy(), 85)
            for p in peaks_wl:
                ax.axvline(x=p, ymin=0.0, ymax=0.9, lw=0.2, color='red', alpha=0.5)
                ax.text(p, y1, "%d" % (p), rotation='vertical', ha='center', fontsize='xx-small', c='red')
            for rl in reflines:
                ax.axvline(x=rl, ymin=0.0, ymax=0.8, lw=0.2, color='green', alpha=0.5)
                ax.text(rl, y2, "%d" % (rl), rotation='vertical', ha='center', fontsize='xx-small', c='green')
            ax.set_ylim(ymin, ymax)
            ax.set_xlabel("comp spectrum position [pixel]")
            ax.set_ylabel("comp spectrum amplitude")
            fig.savefig("reference_spectrum_wlcal.png", dpi=300)
            self.logger.debug("Saved plot to reference_spectrum_wlcal.png")



        # if (True):
        #     fig, ax = plt.subplots(figsize=(13, 5))
        #     ax.plot(full_wl, contsub, lw=0.5)
        #     # ax.plot(wl, contsub, lw=0.5)
        #     ylabelpos = 20000
        #     for p in peaks_wl:
        #         ax.axvline(x=p, ymin=0.0, ymax=0.8, lw=0.2, color='red', alpha=0.5)
        #         ax.text(p, ylabelpos, "%d" % (p), rotation='vertical', ha='center')
        #     for rp in
        #     ax.scatter(reflines, numpy.ones_like(reflines)*2500, marker="|")
        #     # ax.set_yscale('log')
        #     ax.set_ylim(0, 25000)
        #     fig.savefig("reference_spectrum_wlcal.png", dpi=300)

        ref2d = numpy.array([reflines, reflines]).T
        # print(ref2d.shape)
        ref_tree = scipy.spatial.KDTree(
            data=ref2d
        )

        #lambda_central = 6573.56
        #dispersion = -0.31386

        # scan the range
        var_wl, n_wl = 0.01,100 #0.002, 100
        var_disp, n_disp = 0.05, 100 #0.05, 100
        # var_wl, n_wl = 0.002, 11
        # var_disp, n_disp = 0.02, 11
        scan_wl = numpy.linspace(lambda_central * (1. - var_wl), lambda_central * (1 + var_wl), n_wl)
        scan_disp = numpy.linspace(dispersion * (1. - var_disp), dispersion * (1. + var_disp), n_disp)
        self.logger.debug("Scanned central wavelength range: %.3f ... %.3f" % (scan_wl[0], scan_wl[-1]))
        self.logger.debug("Scanned dispersion range: %.3f ... %.3f" % (scan_disp[0], scan_disp[-1]))

        central_y = full_y[full_y.shape[0] // 2]
        peaks2d = numpy.array([peaks, peaks]).T
        results = []
        # print(ref2d.shape)

        results_df = pandas.DataFrame()
        match_radius = numpy.fabs(20 * dispersion)
        self.logger.info("Matching radius for line matching: %.2f AA" % (match_radius))
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
            d, i = ref_tree.query(wl, k=1, p=1, distance_upper_bound=match_radius)
            # TODO: CHANGE 2 to depend on line width and dispersion
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

                #
                # Perform a proper dispersion solution fit using the matched line pairs
                #
                testfit = numpy.polyfit(lines_y, ref_wl, deg=2)
                wl_postfit = numpy.polyval(testfit, peaks2d - central_y)
                d2, i2 = ref_tree.query(wl_postfit, k=1, p=1, distance_upper_bound=match_radius)
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
        if (self.debug):
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

        wl_postfit = numpy.polyval(best_fit, peaks2d - central_y)
        full_wl = numpy.polyval(best_fit, full_y-central_y)
        d2, i2 = ref_tree.query(wl_postfit, k=1, p=1, distance_upper_bound=2)
        matched = (i2 < ref_tree.n)
        n_good_line_matches2 = numpy.sum(matched)
        ref_wl_refined = reflines[i2[matched]]
        self.logger.debug("MATCHING CATALOGS")
        # print("indices:", selected_list.index)
        # print("i2_matched:", i2[matched])
        ref_matched = selected_list.iloc[i2[matched]]
        # print('ref_matched:', len(ref_matched.index))
        # print('comp XXX', len(self.comp_line_inventory['gauss_center'][matched].index))

        self.matched_line_inventory = pandas.DataFrame()
        # print("peaks:", peaks.shape)
        # print("matched:", matched.shape)
        # print("ref_wl_refined:", ref_wl_refined.shape)

        _comps = self.comp_line_inventory[matched].reset_index(drop=True)
        _ref = ref_matched.reset_index(drop=True)
        self.matched_line_inventory = pandas.DataFrame.from_dict({
            'comp_spectrum_pixel': peaks[matched],
            'reference_wl': ref_wl_refined,
            'comp_gauss_center': _comps['gauss_center'],
            'comp_gauss_width': _comps['gauss_width'],
            'comp_gauss_amp': _comps['gauss_amp'],
            'ref_gauss_center': _ref['gauss_wl'],
            'ref_gauss_width': _ref['gauss_width'],
            'ref_gauss_amp': _ref['gauss_amp'],
            },
            orient='columns',
        )
        if (self.debug): self.matched_line_inventory.to_csv("matched_line_inventory.csv", index=False)

        comp_peaks_px = peaks[matched] - self.comp_spectrum_center_y
        use_in_final_fit = numpy.isfinite(ref_wl_refined)
        # print("USE IN FIT", use_in_final_fit.shape)
        plot_fn = "wavelength_solution_details_initial.png"
        try:
            self.make_wavelength_calibration_overview_plot(spec, best_fit, plot_fn=plot_fn)
        except:
            self.logger.warning("Unable to create wave-length calib plot (%s)" % (plot_fn))
            mplog.log_exception()
        for iteration in range(5):
            # TODO: change back to 5, or even better, keep going until
            #  no more changes are detected

            # now we have line positions in pixels and wavelength in A, let's fit
            polyfit = numpy.polyfit(x=comp_peaks_px[use_in_final_fit],
                                    y=ref_wl_refined[use_in_final_fit],
                                    deg=self.wl_polyfit_order)
            fit_wl = numpy.polyval(polyfit, comp_peaks_px)

            delta_wl = ref_wl_refined - fit_wl
            stats = numpy.nanpercentile(delta_wl[use_in_final_fit], [16,50,84])
            _median = stats[1]
            _sigma = 0.5*(stats[2]-stats[0])
            outlier = (delta_wl < (_median-3*_sigma)) | (delta_wl > (_median+3*_sigma))
            use_in_final_fit[outlier] = False

            _pm = peaks[matched]
            _rem = ref_wl_refined
            # print(_pm.shape, _rem.shape)
            try:
                plot_fn = "wavelength_solution_details_iteration%0d.png" % (iteration+1)
                self.make_wavelength_calibration_overview_plot(spec, polyfit, plot_fn=plot_fn, used_in_fit=use_in_final_fit)
            except:
                self.logger.warning("Unable to create wave-length calib plot (%s)" % (plot_fn))

        _pm = peaks[matched]
        _rem = ref_wl_refined
        # print(_pm.shape, _rem.shape)

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
        try:
            plot_fn = "wavelength_solution_details_final.png"
            self.make_wavelength_calibration_overview_plot(spec, polyfit, plot_fn=plot_fn, used_in_fit=use_in_final_fit)
        except:
            self.logger.warning("Unable to create wave-length calib plot (%s)" % (plot_fn))

        #print(best_fit)
        #self.make_wavelength_calibration_overview_plot(spec, best_fit)#, used_in_fit=use_in_final_fit)
        return polyfit  # results[i_most_matches]

    def spec_scale(self, spec):
        scaled = spec.copy() #numpy.sqrt(spec)
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
        ref_line_amp_stats = numpy.nanpercentile(self.ref_inventory['gauss_amp'], [16,50,84])
        # print("ref line amp stats: %s" % (str(ref_line_amp_stats)))
        try:
            typical_ref_line_amp = ref_line_amp_stats[2]
        except:
            typical_ref_line_amp = 100
        if (self.refspec_smoothed is not None):
            axs[0].plot(self.refspec_wavelength, self.spec_scale(self.refspec_smoothed / typical_ref_line_amp),
                        lw=0.4, c='blue', label='ref')

        # disp = -0.30
        # cwl = 6568
        # spec_wl = spec_x0 * disp + cwl
        # wlpf = numpy.polyval(pf, spec_x0)


        comp_wl = numpy.polyval(wavelength_solution, self.comp_spectrum_full_y0)
        # print("womp-wl:\n", comp_wl)
        comp_line_amp_stats = numpy.nanpercentile(self.comp_line_inventory['gauss_amp'], [16,50,84])
        typical_comp_line_amp = comp_line_amp_stats[2]
        self.logger.info("Spec scaling: ref:%f  comp:%f" % (typical_ref_line_amp, typical_comp_line_amp))
        axs[0].plot(comp_wl, self.spec_scale(self.comp_spectrum_continuumsub / typical_comp_line_amp),
                    lw=0.7, c='orange', label='data')
        #
        peaks0 = self.comp_spectrum_lines - self.comp_spectrum_center_y
        peaks_wl = numpy.polyval(wavelength_solution, peaks0)
        # print(peaks_wl)
        axs[0].scatter(peaks_wl, numpy.ones_like(peaks0) * 0.5, marker="|", c='orange')
        #
        if ('gauss_center' in self.ref_inventory.columns):
            axs[0].scatter(self.ref_inventory['gauss_wl'],
                           numpy.ones_like(self.ref_inventory['gauss_center']) * 0.6,
                           c='blue', marker="|")
        for line in self.ref_inventory['gauss_wl']:
            axs[0].axvline(x=line, ymax=0.9, lw=0.4, color='blue', alpha=0.4)
            axs[0].text(line, 0.9, " %d" % (line), rotation='vertical',
                        ha='center', fontsize='xx-small', c='blue')
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
            # print("Assuming all points were used_in_fit")
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

        # First, find average X-position for each fiber
        # fiber_positions = numpy.array([self.raw_traces.get_mean_fiber_position(fiber_id=i) \
        #                    for i in range(self.raw_traces.n_fibers)])
        fiber_positions = self.raw_traces.get_mean_fiber_position(fiber_id='all')
        # print(fiber_positions)

        # self.grating_solution = grating_from_header(self.comp_header, midline_x=332)
        wls = self.grating_solution.wavelength_from_xy(x=fiber_positions,y=None)
        # from fiber-position, derive offsets in central wavelength
        # print(wls)
        central_wl_offset = wls - wls[ref_fiberid]
        # print(central_wl_offset)
        if (self.debug): numpy.savetxt("wl_xy.txt", numpy.array([fiber_positions, wls, central_wl_offset]).T)

        best_fit_dispersion = self.wavelength_solution[-2]
        self.logger.debug("best fit dispersion: %.4f" % (best_fit_dispersion))

        pixel_offsets = central_wl_offset / best_fit_dispersion
        # print(pixel_offsets)

        # ref_fiber_positions = self.comp_line_inventory['gauss_center'].to_numpy()
        ref_fiber_positions = self.matched_line_inventory['comp_gauss_center'].to_numpy()
        ref_fiber_tree = scipy.spatial.KDTree(data=numpy.array([ref_fiber_positions, ref_fiber_positions]).T)

        self.fiber_inventories = {}
        self.fiber_wavelength_solutions = {}
        self.fiber_wavelength_solutions_inverse = {}

        rect_poly = numpy.array([1., 0]) #pixel_offsets[fiber_id]])


        for fiber_id in numpy.hstack([
            numpy.arange(ref_fiberid, self.raw_traces.n_fibers, 1),
            numpy.arange(0, ref_fiberid+1)[::-1]
        ]):
            if (fiber_id == ref_fiberid):
                rect_poly = numpy.array([1., 0])
                prev_fiber_id = ref_fiberid

            # extract spectrum for this fiber
            self.logger.debug("Working on re-identifying lines in fiber %d" % (fiber_id))
            fiber_inventory = self.get_refined_lines_from_spectrum(comp_spectra[fiber_id])

            # apply the approximate position shift to account for curvature
            fiber_inventory['rough_rect_center'] = fiber_inventory['gauss_center'] + pixel_offsets[fiber_id]

            if (self.debug):
                fiber_inventory.to_csv("comp_spectrum_inventory_%d.csv" % (fiber_id), index=False)
                numpy.savetxt("comp_spectrum_%d.spec" % (fiber_id), comp_spectra[fiber_id])

            incremental_curvature = pixel_offsets[fiber_id] - pixel_offsets[prev_fiber_id]
            rect_poly[-1] += incremental_curvature

            # cross-match line positions (in pixel space) from this fiber with the reference fiber
            # this allows us to assign wavelengths (from the calibrated reference fiber) to each matched line
            fiber_pos = fiber_inventory['gauss_center'].to_numpy()
            for iter in range(3):
                rect_position = numpy.polyval(rect_poly, fiber_pos)
                self.logger.debug("Iteration %d --  rect-poly=%s" % (iter+1, str(rect_poly)))
                np2 = numpy.array([rect_position, rect_position]).T

                # now match the rough-aligned peaks to the reference peaks
                max_shift = 5
                d, i = ref_fiber_tree.query(np2, k=1, p=1, distance_upper_bound=max_shift)
                good_match = i < ref_fiber_tree.n
                # print("fiber %d: ref=%d, this=%d, matched=%d" % (
                #     fiberid, ref_tree.n, new_peaks.shape[0], numpy.sum(good_match)))

                # generate a pair of match line positions, in this fiber and the reference fiber
                good_ref_fiber_pos = ref_fiber_positions[i[good_match]]
                good_fiber_pos = fiber_pos[good_match]

                # from this match, derive a simple translation fit
                rect_poly = numpy.polyfit(x=good_fiber_pos, y=good_ref_fiber_pos, deg=1)
                self.logger.debug("FIBER %d, iteration %d: %s (%d / %d|%d)" % (
                    fiber_id, iter, str(rect_poly), numpy.sum(good_match), fiber_pos.shape[0], ref_fiber_positions.shape[0]))

            # calculate the "rectified position"
            fiber_inventory['rect_center'] = numpy.polyval(rect_poly, fiber_inventory['gauss_center'])

            # combine information from the matched comp/ref spectrum
            matched_inventory = fiber_inventory[good_match]
            crossmatched_comp_ref = self.matched_line_inventory.iloc[i[good_match]]
            # overwrite the indices to make merging easier
            crossmatched_comp_ref.index = matched_inventory.index
            self.logger.debug("FIBER %d: matched: %d // crossmatched ref/comp: %d" % (
                fiber_id, len(matched_inventory.index), len(crossmatched_comp_ref.index)
            ))
            fiber_inventory_combined = fiber_inventory.join(
                other=crossmatched_comp_ref, how='outer')

            if (self.debug): fiber_inventory_combined.to_csv("fiber_inventory_%d.csv" % (fiber_id), index=False)
            self.fiber_inventories[fiber_id] = fiber_inventory_combined

            # Now we have a matched catalog, so we can derive a new wavelength calibration
            # for this fiber
            line_pos = fiber_inventory_combined['gauss_center'] - self.raw_traces.midpoint_y
            line_wl = fiber_inventory_combined['reference_wl']
            valid = numpy.isfinite(line_pos) & numpy.isfinite(line_wl)
            fiber_wl_polyfit = numpy.polyfit(
                x=line_pos[valid], y=line_wl[valid],
                deg=self.wl_polyfit_order
            )
            self.fiber_wavelength_solutions[fiber_id] = fiber_wl_polyfit

            fiber_wl_polyfit_inverse = numpy.polyfit(
                y=line_pos[valid], x=line_wl[valid],
                deg=self.wl_polyfit_order
            )
            self.fiber_wavelength_solutions_inverse[fiber_id] = fiber_wl_polyfit_inverse

            # for testing and verification, write out the spectrum including
            # wavelength calibration
            spec_y = numpy.arange(comp_spectra[fiber_id].shape[0]) - self.raw_traces.midpoint_y
            spec_wl = numpy.polyval(fiber_wl_polyfit, spec_y)
            spec_combined = numpy.array([spec_y, spec_wl, comp_spectra[fiber_id]]).T
            if (self.debug): numpy.savetxt("comp_spectrum_%d.txt" % (fiber_id), spec_combined)

            # keep track of what we just worked on
            prev_fiber_id = fiber_id

        return

        #

        # return

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

        for ranges in [numpy.arange(ref_fiberid + 1, self.raw_traces.n_fibers, 1),
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
        center_y = self.raw_traces.midpoint_y #   full_y[full_y.shape[0] // 2]
        centered_y = full_y - center_y

        # along each fiber trace, derive the wavelength
        center_x = self.master_comp.shape[1] // 2
        full_correction_per_fiber = numpy.array([
            numpy.polyval(self.fiber_wavelength_solutions[i], centered_y) for i in range(self.raw_traces.n_fibers)])
        # numpy.polyval(self.transform2d[i, :], centered_y) for i in range(self.n_fibers)])
        # print(full_correction_per_fiber.shape)

        iy,ix = numpy.indices(comp_image.shape, dtype=float)

        # print(ix.shape)
        # for each row, consider the wavelength points in each fiber trace, and fit a horizontal
        # polynomial to this data to interpolate wavelength across fibers
        centered_ix = ix - center_x
        # print(traces.fullres_centers.shape)
        for y in full_y:  # [600:610]:
            centers = traces.fullres_centers[y, :] - center_x
            corrections = full_correction_per_fiber[:, y]
            pfy = numpy.polyfit(centers, corrections, deg=4)
            # print(pfy)
            fullmap_y[y, :] = numpy.polyval(pfy, centered_ix[y, :])

        # calculate actual wavelength for each point
        self.wavelength_mapping_2d = fullmap_y #numpy.polyval(self.wavelength_solution, fullmap_y)

        # fig, ax = plt.subplots()
        # ax.imshow(fullmap_y)
        # pyfits.PrimaryHDU(data=fullmap_y).writeto("full_ymapping.fits", overwrite=True)
        pyfits.PrimaryHDU(data=self.wavelength_mapping_2d).writeto("full_wlmapping.fits", overwrite=True)

        return self.wavelength_mapping_2d

    def get_wavelength_axis(
            self, wavelength_solution=None,
            output_min_wl=None, output_max_wl=None,
            output_dispersion=0.2):
        """
        calculate the user-specified wavelength grid of all
        wavelength-corrected and/or rectified spectra.
        """

        if (wavelength_solution is None):
            wavelength_solution = self.wavelength_solution

        _,y = numpy.indices((self.raw_traces.n_fibers, self.comp_spectra[0].shape[0]), dtype=float)
        # print(y.shape, x.shape)
        y0 = y - self.raw_traces.midpoint_y
        wl = numpy.array([numpy.polyval(self.fiber_wavelength_solutions[id], y0[id])
              for id in range(self.raw_traces.n_fibers)])
        if (self.debug):
            numpy.savetxt("wl_range_info", wl)
            pyfits.PrimaryHDU(data=wl).writeto("reident_wl.fits", overwrite=True)
            pyfits.PrimaryHDU(data=y).writeto("reident_y.fits", overwrite=True)

        _wl_min = numpy.nanmin(wl)
        _wl_max = numpy.nanmax(wl)
        self.data_wl_min = _wl_min
        self.data_wl_max = _wl_max

        self.logger.info("Calibrated wavelength range: %.3f ... %3f" % (_wl_min, _wl_max))
        # y0 = numpy.arange(spec.shape[0]) - self.raw_traces.midpoint_y
        # wl = numpy.polyval(wavelength_solution, y0)

        # prepare the final output wavelength grid
        if (output_min_wl is None):
            output_min_wl = _wl_min
        if (output_max_wl is None):
            output_max_wl = _wl_max

        n_wl_points = int(((output_max_wl - output_min_wl) / output_dispersion)) + 1
        out_wl_points = numpy.arange(n_wl_points, dtype=float) * output_dispersion + output_min_wl
        self.logger.debug("Setting output wavelength grid: %d data points, final range %.2f -- %.2f" % (
            n_wl_points, output_min_wl, output_max_wl
        ))

        return out_wl_points



    def wavelength_calibrate_from_raw_trace(
            self,
            spec, wavelength_solution,
            output_min_wl=None, output_max_wl=None,
            output_dispersion=0.2):

        # # prepare the final output wavelength grid
        # if (output_min_wl is None):
        #     output_min_wl = numpy.nanmin(wl)
        # if (output_max_wl is None):
        #     output_max_wl = numpy.nanmax(wl)
        #
        # n_wl_points = int(((output_max_wl - output_min_wl) / output_dispersion)) + 1
        # out_wl_points = numpy.arange(n_wl_points, dtype=float) * output_dispersion + output_min_wl

        # use the output wavelength grid, convert it to AA to make astropy happy
        out_spectral_axis = self.output_wavelength_axis * u.AA

        # setup spectrum interpolator
        fluxcon = FluxConservingResampler(extrapolation_treatment='nan_fill')

        # Calculate the calibrated wavelength for each uncalibrated pixel
        y0 = numpy.arange(spec.shape[0]) - self.raw_traces.midpoint_y
        wl = numpy.polyval(wavelength_solution, y0)

        # sort by wavelength to make sure wavelength is increasing
        wl_sort = numpy.argsort(wl)
        wl_AA = wl[wl_sort] * u.AA
        flux = spec[wl_sort] * u.DN

        # apply
        spec1d = Spectrum1D(spectral_axis=wl_AA, flux=flux)
        cal_spec = fluxcon(spec1d, out_spectral_axis)

        return cal_spec.flux.to(u.DN).value


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

    # def get_config(self, *args, fallback=None):
    #     config = self.config
    #     for opt in args:
    #         if opt not in config:
    #             return fallback
    #         config = config[opt]
    #     return config

    def calibrate(self, save=False):

        cosmic_options = self.get_cosmic_ray_rejection_options(None)

        # find first file
        flatlist = self.config.get('flat')
        flatlist = [os.path.join(self.raw_dir, fn) for fn in flatlist]
        self.instrument = select_instrument(flatlist[0])
        # print("INSTRUMENT:", self.instrument.name)
        # print(type(self.instrument))

        _master_bias_fn = "master_bias.fits" if save else None
        self.make_master_bias(save=_master_bias_fn, cosmics=cosmic_options)

        _master_flat_fn = "master_flat.fits" if save else None
        self.make_master_flat(save=_master_flat_fn, cosmics=cosmic_options)

        _master_comp_fn = "master_comp.fits" if save else None
        self.make_master_comp(save=_master_comp_fn, cosmics=cosmic_options)

        self.logger.info("Tracing fibers")
        # self.trace_fibers_raw(flat=self.master_flat)

        self.logger.info("Extracting fiber spectra from master flat")
        self.raw_traces = select_instrument(self.comp_header)
        self.raw_traces.find_trace_fibers(self.master_flat)
        self.logger.info("Extracting line profiles for each fiber")
        self.raw_traces.extract_lineprofiles()
        self.logger.info("Saving line profiles")
        self.raw_traces.save_lineprofiles(filename="lineprofiles.fits")

        # comp_spectra = raw_traces.extract_fiber_spectra(
        #     imgdata=self.master_comp,
        #     weights=self.master_flat,
        # )

        # self.reidentify_lines(None, ref_fiberid=45)
        # return

        # self.flat_spectra = self.extract_spectra_raw(imgdata=self.master_flat, weights=self.master_flat)
        self.logger.info("Extracting trace spectra from flatfield")
        self.flat_spectra = self.raw_traces.extract_fiber_spectra(
            imgdata=self.master_flat, weights=self.master_flat)
        # print("flat_spectra.shape", self.flat_spectra.shape)
        pyfits.PrimaryHDU(data=self.flat_spectra).writeto("flat_spectra.fits", overwrite=True)
        # numpy.savetxt("flat_spectra2.dat", self.flat_spectra)

        self.logger.info("Extracting fiber spectra from master comp")
        # self.comp_spectra = self.extract_spectra_raw(imgdata=self.master_comp, weights=self.master_flat)
        self.comp_spectra = self.raw_traces.extract_fiber_spectra(
            imgdata=self.master_comp, weights=self.master_flat)
        # print(self.comp_spectra)
        pyfits.PrimaryHDU(data=self.comp_spectra).writeto("comp_spectra.fits", overwrite=True)
        if (self.debug): numpy.savetxt("comp_spectra2.dat", self.comp_spectra)


        # self.read_reference_linelist()

        # extract lines for reference spectrum
        ref_fibers = numpy.array(self.raw_traces.ref_fiber_id)
        print("#@#@#@#@#@", self.raw_traces.ref_fiber_id)
        self.ref_fiberid = numpy.median(numpy.array(self.raw_traces.ref_fiber_id)).astype(int)
        print("@#@#@#@#@", ref_fibers, ref_fibers.size, self.ref_fiberid)
        if (ref_fibers.size > 1):
            # we have more than one reference fiber -- need to median-combine them
            self.logger.info("Instrument specifies multiple reference fibers -- adding median combine")
            ref_fibers = []
            for rf in self.raw_traces.ref_fiber_id:
                ref_fibers.append(self.comp_spectra[rf])
            ref_fibers = numpy.array(ref_fibers)
            master_ref_fiber = numpy.nanmedian(ref_fibers, axis=0)
        else:
            master_ref_fiber = self.comp_spectra[self.ref_fiberid]
        # find wavelength solution for one "reference" fiber
        self.wavelength_solution = self.find_wavelength_solution(
            master_ref_fiber, make_plots=True, n_brightest=15,
        )
        # print("wavelength solution:", self.wavelength_solution)

        # Now re-identify lines across all other fiber traces
        self.poly_transforms = self.reidentify_lines(
            comp_spectra=self.comp_spectra,
            ref_fiberid=self.ref_fiberid,
            make_plots=False, #True
        )

        self.logger.info("Generating output wavelength grid")
        wl_axis = self.get_wavelength_axis(
            output_min_wl=self.config.get('output','min_wl'),
            output_max_wl=self.config.get('output', 'max_wl'),
            output_dispersion=self.config.get('output', 'dispersion')
        )
        # print(wl_axis)
        self.output_wavelength_axis = wl_axis

        # for human verification, extract and rectify all comp spectra
        self.logger.info("Extracting and calibrating all COMP spectra")
        rect_comp = []
        for fiber_id in range(self.raw_traces.n_fibers):
            rf = self.wavelength_calibrate_from_raw_trace(
                spec=self.comp_spectra[fiber_id],
                wavelength_solution=self.fiber_wavelength_solutions[fiber_id],
                output_min_wl=self.config.get('output', 'min_wl'),
                output_max_wl=self.config.get('output', 'max_wl'),
                output_dispersion=self.config.get('output', 'dispersion'),
            )
            rect_comp.append(rf)
        rect_comp = numpy.array(rect_comp)
        pyfits.PrimaryHDU(data=rect_comp).writeto("rect_comp.fits", overwrite=True)


        self.prepare_fiber_flatfields()
        # pyfits.PrimaryHDU(data=self.fiber_flatfields).writeto("fiber_flatfields.fits", overwrite=True)

        # also extract and wavelength-calibrate all flat spectra
        self.logger.info("Extracting and calibrating all FLAT spectra")
        flat_spectra = self.raw_traces.extract_fiber_spectra(
            imgdata=self.master_flat,
            weights=self.master_flat,
        )
        pyfits.PrimaryHDU(data=flat_spectra).writeto("flat_spectra.fits", overwrite=True)

        rect_flat = []
        for fiber_id in range(self.raw_traces.n_fibers):
            rf = self.wavelength_calibrate_from_raw_trace(
                spec=flat_spectra[fiber_id],
                wavelength_solution=self.fiber_wavelength_solutions[fiber_id],
                output_min_wl=self.config.get('output', 'min_wl'),
                output_max_wl=self.config.get('output', 'max_wl'),
                output_dispersion=self.config.get('output', 'dispersion'),
            )
            rect_flat.append(rf)
        rect_flat = numpy.array(rect_flat)
        pyfits.PrimaryHDU(data=rect_flat).writeto("rect_flat.fits", overwrite=True)

        # fiberflats = self.get_fiber_flatfields()

        rectify = False
        if (rectify):
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

    #         return

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
            self.rect_traces = SparsepakFiberSpecs()
            self.rect_traces.find_trace_fibers(self.flat_rectified_2d)

            self.get_fiber_flatfields()

        #sys.exit(0)


    def prepare_fiber_flatfields(self, filter_width=50):
        # self.flat_fibers = self.raw_traces.extract_fiber_spectra(
        #     imgdata=self.master_flat,
        #     weights=self.master_flat,
        # )
        #

        #
        # In the first iteration, extract all flatfield fiber spectra in
        # raw pixel coordinates (without any interpolation)
        #
        self.flat_spectra = self.raw_traces.extract_fiber_spectra(
            imgdata=self.master_flat, weights=self.master_flat)

        pixel = numpy.arange(self.flat_spectra.shape[1], dtype=float)
        pad_width = filter_width - (pixel.shape[0] % filter_width)
        wl_padded = numpy.pad(pixel, (0, pad_width),
                              mode='constant', constant_values=0)
        wl_padded[-pad_width:] = numpy.nan
        rebinned_wl = numpy.nanmedian(wl_padded.reshape((-1, filter_width)), axis=1)

        data_wl_range = self.data_wl_max - self.data_wl_min
        center_left = self.data_wl_min + 0.4 * data_wl_range
        center_right = self.data_wl_min + 0.6 * data_wl_range
        center_wl = numpy.array([center_left, center_right])
        self.logger.info("Normalizing each flat in the range %.1f ... %.1f" % (
            center_left, center_right))

        fiber_flatfields = [None] * self.raw_traces.n_fibers
        self.fiber_flat_splines = [None] * self.raw_traces.n_fibers
        self.fiber_mean_flux_center = [None] * self.raw_traces.n_fibers

        normalized_master_flat = self.master_flat.copy()

        for fiber_id in range(self.raw_traces.n_fibers):
            # pick a fiber to work on
            fiberspec = self.flat_spectra[fiber_id]

            #
            # prepare a smoothing spline for each fiber to take out large-scale
            # spectral variations
            #

            # make sure we can parcel out the full-res spectra into
            # chunks of a given width
            fiber_padded = numpy.pad(fiberspec, (0, pad_width),
                                     mode='constant', constant_values=0)
            fiber_padded[-pad_width:] = numpy.nan
            n_good = numpy.isfinite(wl_padded) & numpy.isfinite(fiber_padded)

            # calculate the median flux in each little parcel of fluxes
            rebinned_spec = numpy.nanmedian(fiber_padded.reshape((-1, filter_width)), axis=1)
            rebinned_samples = numpy.nansum(n_good.astype(int).reshape((-1, filter_width)), axis=1)
            #     print(rebinned_samples)
            #     print(rebinned_wl)
            good = rebinned_samples > 0.2 * filter_width

            spline = scipy.interpolate.CubicSpline(
                x=rebinned_wl[good],
                y=rebinned_spec[good],
                bc_type='natural'
            )
            full_spline = spline(pixel)

            self.fiber_flat_splines[fiber_id] = spline
            fiber_flatfields[fiber_id] = full_spline

            #
            # Now let's apply a wavelength calibration so we can normalize
            # each flatfield to the average intensity in the same wavelength range
            #
            center_pixels = numpy.polyval(
                self.fiber_wavelength_solutions_inverse[fiber_id],
                center_wl,
            ) + self.raw_traces.midpoint_y
            center_pixel_left = numpy.min(center_pixels).astype(int)
            center_pixel_right = numpy.max(center_pixels).astype(int)
            # print(fiber_id, center_wl, center_pixels, center_pixel_left, center_pixel_right)
            mean_flux = numpy.mean(fiberspec[center_pixel_left:center_pixel_right])
            self.fiber_mean_flux_center[fiber_id] = mean_flux

            # numpy.savetxt(
            #     "fiberflat_%d.txt" % (fiber_id),
            #     numpy.array([wl_padded, fiber_padded, spline(wl_padded)]).T
            # )

            # self.master_flat
            #
            # normalize the master-flat to account for large-scale variations
            #
            mf = self.master_flat / full_spline.reshape((-1,1)) / mean_flux
            fibermask = self.raw_traces.get_fiber_mask(self.master_flat, fiber_id)
            normalized_master_flat[fibermask] = mf[fibermask]

        self.master_flat_normalized = normalized_master_flat

        # not sure if we actually need these here
        # TODO: check
        pyfits.PrimaryHDU(data=self.master_flat).writeto("masterflat_before.fits", overwrite=True)
        pyfits.PrimaryHDU(data=normalized_master_flat).writeto("masterflat_after.fits", overwrite=True)
        pyfits.PrimaryHDU(data=self.master_flat/normalized_master_flat).writeto("masterflat_xxx.fits", overwrite=True)

        self.fiber_flatfields_smoothed_spline = numpy.array(fiber_flatfields)
        self.fiber_mean_flux_center = numpy.array(self.fiber_mean_flux_center)
        if (self.debug): numpy.savetxt("fiber_mean_flux_center", self.fiber_mean_flux_center)

        self.logger.debug("removing overall variations from flatfields to isolate pixel-to-pixel variations")
        self.fiber_flatfield_pixel2pixel = self.flat_spectra / self.fiber_flatfields_smoothed_spline

        pyfits.PrimaryHDU(data=self.flat_spectra).writeto(
            "fiber_flatfield_raw.fits", overwrite=True)
        pyfits.PrimaryHDU(data=self.fiber_flatfields_smoothed_spline).writeto(
            "fiber_flatfield_smoothedspline.fits", overwrite=True)
        pyfits.PrimaryHDU(data=self.fiber_flatfield_pixel2pixel).writeto(
            "fiber_flatfield_pixel2pixel.fits", overwrite=True)

        #
        # Generate the average intensity around the middle of the spectral range
        #
        # _wl_range = pixel[-1] - pixel[0]
        # _left = int(pixel[0] + 0.4 * _wl_range)

    def get_rectified_fiber_flatfields(self, filter_width=50):
        self.flat_fibers = self.rect_traces.extract_fiber_spectra(
            imgdata=self.flat_rectified_2d,
            weights=self.flat_rectified_2d,
        )

        wl = numpy.arange(self.flat_fibers.shape[1], dtype=float)
        pad_width = filter_width - (wl.shape[0] % filter_width)
        wl_padded = numpy.pad(wl, (0, pad_width),
                              mode='constant', constant_values=0)
        wl_padded[-pad_width:] = numpy.nan
        rebinned_wl = numpy.nanmedian(wl_padded.reshape((-1, filter_width)), axis=1)

        fiber_flatfields = [None] * self.raw_traces.n_fibers
        self.fiber_flat_splines = [None] * self.raw_traces.n_fibers

        for fiber_id in range(self.raw_traces.n_fibers):
            # pick a fiber to work on
            fiberspec = self.flat_fibers[fiber_id]

            # make sure we can parcel out the full-res spectra into
            # chunks of a given width
            fiber_padded = numpy.pad(fiberspec, (0, pad_width),
                                     mode='constant', constant_values=0)
            fiber_padded[-pad_width:] = numpy.nan
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

    def plot_sky_spectrum(self, skyspec, plot_fn):
        self.logger.info("Plotting sky spectrum")
        # Read the wavelengths of all skylines
        all = []
        skylines_fn = get_file("skylines.dat")
        with open(skylines_fn,"r") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("#"):
                    continue
                try:
                    wl = float(line.split(" ")[0])
                    all.append(wl)
                except:
                    continue
        #skylines_p1 = pandas.read_csv(skylines_fn, comment="#", names=['wl'], header=0, sep='\s+')
        ohlines_fn = get_file("ohlines.dat")
        with open(ohlines_fn,"r") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("#"):
                    continue
                try:
                    wl = float(line.split(" ")[0])
                    all.append(wl)
                except:
                    continue
        #ohlines = pandas.read_csv(ohlines_fn, comment="#", names=['wl'], header=0, sep='\s+')
        #skyline_cat = pandas.concat([skylines_p1,ohlines], ignore_index=True, axis='index')
        #print(skyline_cat)
        #skyline_cat.info()
        skylines = numpy.array(all) #skyline_cat['wl'].to_numpy()

        wl = self.output_wavelength_axis
        good_data = numpy.isfinite(skyspec)
        min_wl = numpy.min(wl[good_data])
        max_wl = numpy.max(wl[good_data])
        print(min_wl, max_wl)

        sky_amp_stats = numpy.nanpercentile(skyspec, [1, 98])
        sky_amp_range = sky_amp_stats[1] - numpy.max([sky_amp_stats[0],0])
        min_sky_amp = sky_amp_stats[0] - 0.03 * sky_amp_range
        max_sky_amp = sky_amp_stats[1] + 0.03 * sky_amp_range
        y1,y2 = min_sky_amp, max_sky_amp #-0.03*peak_sky_amp,peak_sky_amp
        ytext = 0.87*(y2-y1)+y1

        fig, ax = plt.subplots(figsize=(14,4), tight_layout=True)
        ax.plot(wl, skyspec, lw=0.5, label='sky template')
        skyy = numpy.ones_like(skylines)

        ax.scatter(skylines, skyy*4, marker="|", c='red', label='OH Lines')
        ax.set_xlim((min_wl,max_wl))
        ax.set_ylim((y1,y2))
        #ax.legend()
        ax.set_xlabel("Wavelength [A]")
        ax.set_ylabel("flux [counts/second]")

        selected_lines = (skylines > min_wl) & (skylines < max_wl)
        for line in skylines[selected_lines]:
            # print(line)
            ax.axvline(x=line, ls=":", alpha=0.5, ymax=0.85, c='grey')
            ax.text(line, ytext, "%.1f" % line, rotation=90, fontsize='xx-small',ha='center', va='bottom')
        fig.savefig(plot_fn, dpi=200)
        self.logger.info("sky specturm plot written to %s" % (plot_fn))
        return

    def create_sky_spectrum(self, flattened_spec, sky_fiber_ids):
        self.logger.info("Creating SKY spectrum")


        # first, generate a mean sky
        self.logger.debug("First, generate mean sky spectrum (%s)" % (str(sky_fiber_ids)))
        skies = numpy.array([
            flattened_spec[_id].spec for _id in sky_fiber_ids
        ])
        pyfits.PrimaryHDU(data=skies).writeto("rough_skies.fits", overwrite=True)
        rough_sky = numpy.nanmedian(skies, axis=0)
        if (self.debug):
            numpy.savetxt("rough_skies.txt", skies)
            numpy.savetxt("rough_sky.txt", rough_sky)
        rough_snl = SpecAndLines(rough_sky)

        # next, match all individual spectra to the mean sky to achieve some common scaling
        self.logger.debug("Deriving common scaling for all sky fibers")
        scalings = []
        sky_snl = []
        for specid in sky_fiber_ids:  # [:1]:
            #spec = rect_flattened[specid]
            # sky_n_l = SpecAndLines(rect_flattened[specid])
            scale, mfs, mfo, fig = rough_snl.match_amplitude(flattened_spec[specid], plot=True) # TODO: CHECK
            self.logger.debug("scaling single sky to mean sky: %f" % (scale))
            # print(scale)
            if (not numpy.isfinite(scale) ):
                self.logger.warning("Excluding fiber %d from mean sky spectrum" % (specid+1))
                flattened_spec[specid].dump("problem_sky_spec_%03d.txt" % (specid+1))
                continue
            scalings.append(scale)
            sky_snl.append(SpecAndLines(flattened_spec[specid].spec / scale))

        # create a master-sky, combined a mean continuum with a median line
        self.logger.debug("Computing mean sky continuum spectrum")
        conts = numpy.array([s.continuum for s in sky_snl])
        master_cont = numpy.mean(conts, axis=0)
        # print(master_cont.shape)
        #
        # fig, ax = plt.subplots(figsize=(20, 4))
        # for s in sky_snl:
        #     ax.plot(wl, s.continuum, lw=1, alpha=0.5)
        # ax.plot(wl, master_cont, lw=3, alpha=0.5)

        # Now focus on sky lines
        self.logger.debug("Refining sky line spectrum by clipping outliers")
        matched_lines = numpy.array([s.contsub for s in sky_snl])
        iter_lines = matched_lines.copy()
        # print(iter_lines.shape)
        sigmas = []
        for it in range(3):
            _stats = numpy.nanpercentile(iter_lines, [16, 50, 84], axis=0)
            # print(_stats.shape)
            _med = _stats[1]
            _sigma = 0.5 * (_stats[2] - _stats[0])
            bad = (iter_lines > (_med + 3 * _sigma)) | (iter_lines < (_med - 3 * _sigma))
            iter_lines[bad] = numpy.nan
            sigmas.append(_sigma)
        iter_lines_mean = numpy.nanmean(iter_lines, axis=0)

        # master_lines = numpy.median(lines, axis=0)
        # fig, ax = plt.subplots(figsize=(20, 4))
        # for s in sky_snl:
        #     ax.plot(wl, s.contsub, lw=1, alpha=0.5)
        # ax.plot(wl, master_lines, lw=3, alpha=0.5)
        # ax.set_ylim((-20, 400))
        # ax.set_xlim((800, 1150))

        # combine sky continuum and lines
        master_sky_combined = master_cont + iter_lines_mean #  master_lines
        master_sky_snl = SpecAndLines(master_sky_combined)


        #
        #
        # # rescale all sky-spectra to yield a better median line spectrum
        # match_sky_snl = []
        # for specid, scale in zip(all_skyfibers_id, sky_scalings):
        #     snl = SpecAndLines(rect_flattened[specid] * scale)
        #     match_sky_snl.append(snl)
        #
        # matched_lines = numpy.array([s.contsub for s in match_sky_snl])
        # master_lines_matched = numpy.median(matched_lines, axis=0)
        #
        # iter_lines = matched_lines.copy()
        # print(iter_lines.shape)
        # # valid = numpy.isfinite(matched_lines)
        # sigmas = []
        # for it in range(3):
        #     _stats = numpy.nanpercentile(iter_lines, [16, 50, 84], axis=0)
        #     #    if _stats: # is empty
        #     #        break
        #     print(_stats.shape)
        #     _med = _stats[1]
        #     _sigma = 0.5 * (_stats[2] - _stats[0])
        #     bad = (iter_lines > (_med + 3 * _sigma)) | (iter_lines < (_med - 3 * _sigma))
        #     iter_lines[bad] = numpy.nan
        #     sigmas.append(_sigma)
        # iter_lines_mean = numpy.nanmean(iter_lines, axis=0)
        #
        # fig, ax = plt.subplots(figsize=(20, 4))
        # for s in match_sky_snl:
        #     ax.plot(wl, s.contsub, lw=1, alpha=0.5)
        # ax.plot(wl, master_lines_matched, lw=3, alpha=0.5)
        # ax.plot(wl, iter_lines_mean, lw=3, alpha=0.5, c='red')
        # ax.plot(wl, _sigma - 20, lw=1, alpha=0.5, c='red')
        # ax.set_ylim((-50, 400))
        # ax.set_xlim((800, 1150))
        #
        # # create a master-sky, combined a mean continuum with a median line
        # matched_conts = numpy.array([s.continuum for s in match_sky_snl])
        # master_cont_matched = numpy.mean(matched_conts, axis=0)
        # print(master_cont_matched.shape)
        #
        # final_master_sky = master_cont_matched + master_lines_matched
        # final_master_sky_snl = SpecAndLines(final_master_sky)
        # fig, ax = plt.subplots(figsize=(20, 4))
        # ax.plot(wl, final_master_sky)
        # for s in all_skyfibers_id:
        #     ax.plot(wl, rect_flattened[s], lw=1, alpha=0.3)
        # ax.set_ylim((100, 1200))
        #
        # fig, ax = plt.subplots(figsize=(20, 4))
        # ax.plot(wl, master_lines_matched)
        # ax.plot(wl, iter_lines_mean)
        # ax.set_ylim((-20, 200))
        # ax.set_xlim((850, 1500))
        #
        # fig, ax = plt.subplots(figsize=(20, 4))
        # for s in sigmas:
        #     ax.plot(wl, s, lw=0.5)
        # ax.set_ylim((0, 60))
        #

        return master_sky_snl
        # lastly, generate the actual master sky template


    def select_lowflux_fibers(self, sci_rect, wl, wl_range, n_fibers):
        self.logger.info("Finding %d lowest-flux fibers"  % (n_fibers))

        # identify which rectified points are within the selected range
        in_wl_range = (wl >= wl_range[0]) & (wl <= wl_range[1])
        idx_in_range = numpy.arange(wl.shape[0])[in_wl_range]
        wl_idx_left = numpy.min(idx_in_range)
        wl_idx_right = numpy.min([numpy.max(idx_in_range) + 1, wl.shape[0]])

        # compute average flux in this range
        fluxes = numpy.nanmean(sci_rect[:, wl_idx_left:wl_idx_right], axis=1)
        # print("fluxes-shape:", fluxes.shape)

        # sort fluxes
        flux_sort = numpy.argsort(fluxes)

        # determine IDs of N lowest-flux fibers
        fiber_id = numpy.arange(sci_rect.shape[1])
        fibers = fiber_id[flux_sort[:n_fibers]]
        return fibers

    def write_wavelength_calibrated_image(self, data, wl, filename, header=None):

        phdu = pyfits.PrimaryHDU(data=data, header=header)
        # TODO: initialize header with header keywords from input file

        # define the actual wavelength axis
        hdr = phdu.header
        hdr['CTYPE1'] = 'WAVE'
        hdr['CRPIX1'] = 1
        hdr['CRVAL1'] = wl[0] #*1e-10
        hdr['CD1_1'] = (wl[1]-wl[0]) #*1e-10
        hdr['CDELT1'] = (wl[1]-wl[0]) #*1e-10

        # also define a linear scale denoting the fiber number
        # Todo: Add this to the instrument definition for more flexibility
        hdr['CTYPE2'] = "LINEAR"
        hdr['CRVAL2'] = 1
        hdr['CRPIX2'] = 1
        hdr['CDELT2'] = 1
        hdr['CD2_2'] = 1

        phdu.writeto(filename, overwrite=True)
        self.logger.debug("done writing wavelength-calibrated image to %s" % (filename))

    def gather_pointing_data(self, target_name, raw_traces, header):
        pointing_mode = self.config.get(target_name, 'pointing', 'mode')
        if (pointing_mode is None):
            self.logger.warning("No pointing data information found for target %s" % (target_name))
            return None

        pointing_reference = self.config.get(target_name, 'pointing', 'reference')
        if (pointing_reference is None):
            self.logger.warning("Missing pointing reference for target %s" % (target_name))
            return None

        ra = self.config.get(target_name, 'pointing', 'ra')
        dec = self.config.get(target_name, 'pointing', 'dec')
        rotation = self.config.get(target_name, 'pointing', 'rotation', fallback=0)
        if (ra is None or dec is None):
            self.logger.warning("Missing reference coordinates (RA:%s, Dec:%s)" % (str(ra), str(dec)))
            return None

        self.logger.info("Pointing info for target %s: mode:%s ref:%s -- RA:%.5f DEC:%.5f rot=%.1f" % (
            target_name, pointing_mode, pointing_reference, ra, dec, rotation
        ))
        fiber_coords = raw_traces.sky_positions(
            pointing_mode, pointing_reference, ra, dec, rotation)
        return fiber_coords

    def pointing_data_to_header(self, fiber_coords, header):
        for fiberid in range(self.raw_traces.n_fibers):
            coord = fiber_coords[fiberid]
            header['F%03d_RA' % (fiberid+1)] = coord.ra.to(u.degree).value
            header['F%03d_DEC'% (fiberid+1)] = coord.dec.to(u.degree).value


    def reduce(self):

        for target_name in self.config.get('science'):
            self.logger.info("Starting reduction for target %s",  target_name)
            filelist = self.config.get(target_name, "files")
            target_outdir = self.config.get(target_name, "output_directory")

            # make sure we always deal with lists, even if they only have one element
            if (not isinstance(filelist, list)):
                filelist = [filelist]
            # print(filelist)

            target_combined, target_header = self.basic_reduction(
                filelist=filelist,
                bias=self.master_bias,
                flat=None,
                op=numpy.nanmedian
            )

            pointing_data = self.gather_pointing_data(target_name, self.raw_traces, target_header)
            # print(pointing_data)
            # print(target_header)
            if (pointing_data is None):
                self.logger.warning("Unable to add pointing information to output headers")
            else:
                self.logger.info("Preparing pointing information for output headers")
                self.pointing_data_to_header(pointing_data, target_header)

            # positioning = self.raw_traces.

            __fn = os.path.join(target_outdir, "%s__combined.fits" % (target_name))
            self.logger.info("Writing results for target '%s' to %s ..." % (target_name, __fn))
            pyfits.PrimaryHDU(data=target_combined, header=target_header).writeto(__fn, overwrite=True)
            # print(target_combined)

            # __fn = "%s__flat.fits" % (target_name)
            # self.logger.info("Writing results for target '%s' to %s ..." % (target_name, __fn))
            # pyfits.PrimaryHDU(data=self.master_flat_normalized, header=target_header).writeto(__fn, overwrite=True)
            #
            # # divide by normalized master flat
            # target_combined = target_combined / self.master_flat_normalized
            # __fn = "%s__normflatcorr.fits" % (target_name)
            # self.logger.info("Writing results for target '%s' to %s ..." % (target_name, __fn))
            # pyfits.PrimaryHDU(data=target_combined, header=target_header).writeto(__fn, overwrite=True)

            self.logger.info("Extracting trace spectra [target: %s]" % (target_name))
            sci_spectra = self.raw_traces.extract_fiber_spectra(
                imgdata=target_combined, weights=self.master_flat)

            # for human verification, extract and rectify all comp spectra
            self.logger.info("Appying wavelength calibrating to extracted spectra")
            rect_sci_target = []
            target_wl = self.get_wavelength_axis(
                wavelength_solution=self.fiber_wavelength_solutions,
                output_min_wl=self.config.get(target_name, 'output', 'min_wl'),
                output_max_wl=self.config.get(target_name, 'output', 'max_wl'),
                output_dispersion=self.config.get(target_name, 'output', 'dispersion')

            )

            for fiber_id in range(self.raw_traces.n_fibers):
                rf = self.wavelength_calibrate_from_raw_trace(
                    spec=sci_spectra[fiber_id],
                    wavelength_solution=self.fiber_wavelength_solutions[fiber_id],
                    output_min_wl=self.config.get('output', 'min_wl'),
                    output_max_wl=self.config.get('output', 'max_wl'),
                    output_dispersion=self.config.get('output', 'dispersion')
                )
                rect_sci_target.append(rf)
            rect_sci_target = numpy.array(rect_sci_target)

            __fn = os.path.join(target_outdir, "%s_rectified_check.fits" % (target_name))
            self.logger.info("Writing extracted & calibrated spectra to %s" % (__fn))
            self.write_wavelength_calibrated_image(
                self.raw_traces.reorder_fibers(rect_sci_target),
                target_wl, __fn, target_header)
            # pyfits.PrimaryHDU(data=rect_sci_target).writeto(__fn, overwrite=True)

            # TODO: apply flatfielding
            sky_fiber_ids = []

            # TODO: Combine all sky-fibers into a master sky spectrum
            fiber_snls = [None] * self.raw_traces.n_fibers
            for fiber_id in range(self.raw_traces.n_fibers):
                fiber_snls[fiber_id] = SpecAndLines(rect_sci_target[fiber_id])

            sky_mode = self.config.get(target_name, "sky", "mode")
            self.logger.debug("raw sky mode: %s" % (sky_mode))
            if (sky_mode == 'default'):
                self.logger.info("Using DEFAULT sky mode for target %s" % (target_name))
                sky_fiber_ids = self.raw_traces.get_sky_fiber_ids()
            elif (sky_mode == 'custom'):
                _fibers = self.config.get(target_name, "sky", "fibers")
                sky_fiber_ids = numpy.array([int(f) - 1 for f in _fibers])
                self.logger.info("Using CUSTOM sky fibers for target %s (%s)" % (
                    target_name, ", ".join(["%d" % (f+1) for f in sky_fiber_ids])))
            elif (sky_mode == "minflux"):
                n_fibers = self.config.get(target_name, "sky", "fibers")
                wl_range = self.config.get(target_name, "sky", "window")
                sky_fiber_ids = self.select_lowflux_fibers(rect_sci_target, target_wl, wl_range, n_fibers)

            self.logger.info("Creating sky template for %s from fibers %s" % (
                target_name, ", ".join(["%d" % (f+1) for f in sky_fiber_ids])
            ))
            final_master_sky_snl = self.create_sky_spectrum(fiber_snls, sky_fiber_ids=sky_fiber_ids)
            __fn = os.path.join(target_outdir, "%s_skyspec.fits" % (target_name))
            self.write_wavelength_calibrated_image(
                final_master_sky_snl.spec,
                target_wl, __fn, target_header)
            # pyfits.PrimaryHDU(data=final_master_sky_snl.spec).writeto(__fn, overwrite=True)
            if (self.make_plots):
                __fn = os.path.join(target_outdir, "%s_skyspec.png" % (target_name))
                self.plot_sky_spectrum(final_master_sky_snl.spec, __fn)

            # TODO: Subtract sky from each fiber: Option1: simple subtract; Option2: Fit optimal shift & amplitude
            self.logger.info("Fitting sky amplitude and performing sky subtraction for each spectrum")
            sky_subtraction_mode = self.config.get(target_name, "sky", "components")
            sky_matching_mode = self.config.get(target_name, "sky", "match")
            skysub_all = numpy.zeros_like(rect_sci_target)
            rect_flattened = rect_sci_target
            sky_scalings = []
            plot=self.debug
            for fiberid in range(self.raw_traces.n_fibers):
                spec = rect_flattened[fiberid]
                spec_snl = SpecAndLines(spec)
                if (plot):
                    skyscale, _, _, fig = final_master_sky_snl.match_amplitude(spec_snl, plot=True)
                    fig.suptitle("fiber %d // scale=%.5f" % (fiberid + 1, skyscale))
                    fig.savefig(os.path.join(target_outdir, '%s__skymatch_%02d.png' % (target_name, fiberid + 1)), dpi=200)
                    plt.close(fig)
                else:
                    skyscale, _, _ = final_master_sky_snl.match_amplitude(spec_snl, plot=False)

                sky_scalings.append(skyscale)
                self.logger.debug("Sky scaling fiber %d -- %s / %s -- scaling: %f " % (
                    fiberid, sky_subtraction_mode, sky_matching_mode, skyscale))
                if (sky_subtraction_mode == "lines"):
                    ## only subtract lines
                    this_skyspec = final_master_sky_snl.contsub
                elif (sky_subtraction_mode == "continuum"):
                    this_skyspec = final_master_sky_snl.continuum
                else:
                    this_skyspec = final_master_sky_snl.spec

                if (sky_matching_mode == "sky2spec"):
                    skysub_all[fiberid] = spec - this_skyspec / skyscale
                else:
                    skysub_all[fiberid] = spec * skyscale - this_skyspec

            self.logger.debug("Sky scalings:\n%s" % (" ".join(["%7.3f" % s for s in sky_scalings])))

            __fn = os.path.join(target_outdir, "%s_skysub.fits" % (target_name))
            self.logger.info("Writing sky-subtracted spectra to %s" % (__fn))
            self.write_wavelength_calibrated_image(
                self.raw_traces.reorder_fibers(skysub_all),
                target_wl, __fn, target_header)
            # pyfits.PrimaryHDU(data=rect_sci_target).writeto(__fn, overwrite=True)
            # pyfits.PrimaryHDU(data=skysub_all).writeto(__fn, overwrite=True)

            continue

            # target_rect = self.rectify(
            #     target_, self.poly_transforms,
            #     min_wl=self.get_config('output', 'min_wl', fallback=None),
            #     max_wl=self.get_config('output', 'max_wl', fallback=None),
            #     out_dispersion=self.get_config('output', 'dispersion', fallback=None)
            # )
            # self.write_rectified_spectrum(
            #     spec=target_rect,
            #     output_filename="%s__rectified.fits" % (target_name)
            # )

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
            sky_fibers = numpy.array([22, 16, 2, 38, 54, 80, 70]) # TODO: Use from instrument definition
            sky_fiberids = self.raw_traces.n_fibers - sky_fibers # TODO: Fix this too
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
    benchspec.reduce()

