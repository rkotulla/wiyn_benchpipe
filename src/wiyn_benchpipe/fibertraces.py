

import numpy
import logging
import scipy.ndimage
import scipy.signal
import pandas
import astropy.io.fits as pyfits
import astropy.stats
import matplotlib.pyplot as plt

from .grating import Grating

DS9_HEADER = """\
# Region file format: DS9 version 4.1
global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
physical\
"""

class GenericFiberSpecs(object):

    n_fibers = -1
    ref_fiber_id = 0
    sky_fiber_ids = None
    fiber_profiles = None
    header = None

    trace_minx = 0
    trace_maxx = 1e9

    name = "Generic Instrument"

    input_transpose = False
    input_flipx = False
    input_flipy = False
    input_ext_data = 0
    input_ext_header = 0

    def __init__(self, logger=None, debug=False, trace_minx=None, trace_maxx=None, header=None):
        if (self.n_fibers < 0):
            raise ValueError("Invalid number of fibers (%d) -- don't use the base class!" % (self.n_fibers))
        self.debug = debug

        if (logger is None):
            logger = logging.getLogger('FiberSpecs')
        self.logger = logger

        if (trace_minx is not None):
            self.trace_minx = trace_minx
        if (trace_maxx is not None):
            self.trace_maxx = trace_maxx

        if (header is not None):
            self.header = header

        self.logger.info("Loading definitions for %s" % (self.name))
        return

    def find_trace_fibers(self, trace_image):

        if (trace_image is None):
            raise ValueError("Need to provide a trace_image!")
        self.trace_image = trace_image

        trace_minx = self.trace_minx if self.trace_minx is not None else 0
        trace_maxx = self.trace_maxx if self.trace_maxx is not None else trace_image.shape[1]

        # store some information about image dimensions
        self.size_x = trace_image.shape[1]
        self.size_y = trace_image.shape[0]
        self.midpoint_y = self.size_y / 2.

        self.full_y = numpy.arange(self.size_y)


        # self.n_fibers = 82

        #
        # do a background subtraction first to increase contrast
        #
        # first step, reject outliers by median-filtering ALONG fibers
        # (9,1) works well for sparsepak (9 px wide across fibers, 1px long along fibers)
        self.logger.debug("Preparing frame for fiber tracing")
        median_filter_1d = scipy.ndimage.median_filter(
            input=trace_image, size=(9, 1),
        )
        min_filter = scipy.ndimage.minimum_filter(
            input=median_filter_1d,  # masterflat,
            size=(5, 30)
        )
        # Now fit a linear slope to the background
        w = 10
        left_edge = w #80  ## adjust this for binning, assuming 4x3
        right_edge = self.size_x - w - 1 #570

        left = numpy.mean(min_filter[:, left_edge - w:left_edge + w], axis=1).reshape((-1, 1))
        right = numpy.mean(min_filter[:, right_edge - w:right_edge + w], axis=1).reshape((-1, 1))
        slope = (right - left) / (right_edge - left_edge)
        iy, ix = numpy.indices(trace_image.shape, dtype=float)
        gradient_2d = (ix - left_edge) * slope + left

        # subtract the modeled background
        bgsub = trace_image - gradient_2d
        bgsub[bgsub < 0] = 0
        self.bgsub = bgsub  # TODO: fix this with proper variable name

        # Now trace the fibers
        self.logger.debug("Tracing ridge-lines of each fiber")
        dy = 5  ## adjust for binning
        traces = pandas.DataFrame()

        smooth_dx = int(numpy.ceil(bgsub.shape[1] / self.n_fibers * 10))

        #
        # Go through the entire spectrum in chunks of dy and identify peaks (for spectra tracing)
        # and valleys between peaks (as boundaries for spectra traces)
        #
        raw_all_peaks = []
        raw_all_y = []

        all_peaks = []
        all_traces_y = []
        all_valleys = []
        for y in range(dy, bgsub.shape[0], 2 * dy):
            prof = numpy.nanmedian(bgsub[y - dy:y + dy, :], axis=0)

            # apply some min and max scaling to normalize the area to peaks have values close to 1,
            # and ideally valleys have values close to 0
            mins = scipy.ndimage.minimum_filter1d(prof, size=smooth_dx)
            maxs = scipy.ndimage.maximum_filter1d(prof, size=smooth_dx)
            minsub = prof - mins
            maxsub = scipy.ndimage.maximum_filter1d(minsub, size=smooth_dx)
            norm_prof = (prof - mins) / maxsub

            # Find all peaks
            peaks, peak_props = scipy.signal.find_peaks(norm_prof, height=0.25, distance=3)
            valid_peaks = (peaks > trace_minx) & (peaks <= trace_maxx)
            peaks = peaks[valid_peaks]

            raw_all_y.append(y)
            raw_all_peaks.append(peaks)

            if (self.debug):
                numpy.savetxt("prof_y=%d" % y, prof)
                numpy.savetxt("profpeaks_y=%d" % y, numpy.array([peaks, prof[peaks]]).T)

            if (peaks.shape[0] != self.n_fibers):
                # We didn't find the right number of peaks, so we can't use this one
                print(y, "off, #=%d" % (peaks.shape[0]))
                continue

            # reverse the flat, and find all valleys
            inv_norm = 1. - norm_prof
            valleys, _ = scipy.signal.find_peaks(inv_norm, height=0.2, distance=3)
            valid_valley = (valleys >= trace_minx) & (valleys <= trace_maxx)
            valleys = valleys[valid_valley]

            _left = peaks[0]
            _right = peaks[-1]
            good = (valleys > _left) & (valleys < _right)
            good_valleys = valleys[good]
            if (len(good_valleys) != self.n_fibers-1):
                fixed_valleys = []
                for l,r in zip(peaks[:-1], peaks[1:]):
                    in_between = (good_valleys > l) & (good_valleys < r)

                    if (numpy.sum(in_between) == 0):
                        valley_pos = ((l+r)/2)
                    else:
                        valley_pos = numpy.nanmedian(good_valleys[in_between])
                    # if (y == 3105):
                    #     print(l,r,valley_pos, )
                    fixed_valleys.append(valley_pos)
                good_valleys = fixed_valleys

            all_peaks.append(peaks)
            all_valleys.append(good_valleys)
            all_traces_y.append(y)

        # make a quick & dirty plot showing all line traces; also export in ds9 format for visualization
        self.logger.info("Exporting all trace data as plot and ds9 region file")
        ds9_reg = open("linetraces.reg", "w")
        print(DS9_HEADER, file=ds9_reg)
        fig, ax = plt.subplots(figsize=(12,12))
        for y,peaks in zip(raw_all_y, raw_all_peaks):
            peaks = numpy.array(peaks)
            _y = numpy.ones_like(peaks) * y
            for x in peaks:
                print("point(%f,%f) # point=cross" % (x+1,y+1), file=ds9_reg)
            ax.scatter(peaks, _y, marker='+', s=2)
        fig.tight_layout()
        fig.savefig("linetraces.png", dpi=200)

        self.logger.info("Found all traces across %d samples (dy=%d)" % (len(all_peaks), dy))
        centers = numpy.array(all_peaks)
        valleys = numpy.array(all_valleys)
        all_traces_y = numpy.array(all_traces_y)
        if (self.debug):
            numpy.savetxt("centers", numpy.hstack([all_traces_y.reshape((-1,1)),centers]))
            numpy.savetxt("valleys", numpy.hstack([all_traces_y.reshape((-1,1)),valleys]))

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

#         # invert masterflat to search for the minima between the cells
#         self.logger.debug("Tracing valley lines that limit fibers")
#         inverted = -1. * bgsub
#         all_troughs = []
#         print("CENTERS shape", centers.shape)
# #        with open("all_troughs_before", "w") as at:
# #            for y, t in zip(all_traces_y, all_troughs):
# #                print("%f %d %s" % (y, len(t), " ".join(["%.2f" % f for f in t])), file=at)
#         for i, y in enumerate(all_traces_y):
#             prof = numpy.nanmedian(inverted[y - dy:y + dy, :], axis=0)
#             if (self.debug): numpy.savetxt("prof_inv_raw_y=%d" % y, prof)
#             prof = scipy.ndimage.gaussian_filter(prof, sigma=1)
#             if (self.debug): numpy.savetxt("prof_inv_filt_y=%d" % y, prof)
#             peak_intensity = numpy.min(prof)
#             # print(y, peak_intensity)
#
#             troughs, troughs_props = scipy.signal.find_peaks(prof, height=0.5 * peak_intensity, distance=3)
#             if (self.debug): numpy.savetxt("proftrough_y=%d" % y, numpy.array([troughs, prof[troughs]]).T)
#
#             _left = leftmost_peak[i]
#             _right = rightmost_peak[i]
#             good = (troughs > _left) & (troughs < _right)
#             good_troughs = troughs[good]
#             if (len(good_troughs) != self.n_fibers-1):
#                 fixed_troughs = []
#                 for l,r in zip(centers[i,:-1], centers[i, 1:]):
#                     in_between = (good_troughs > l) & (good_troughs < r)
#                     if (len(in_between) == 0):
#                         fixed_troughs.append((l+r)/2)
#                     else:
#                         fixed_troughs.append(numpy.median(good_troughs[in_between]))
#                 good_troughs = fixed_troughs
#             # print(y, peak_intensity, peaks.shape, good_peaks.shape)
#             all_troughs.append(good_troughs)
#
#         with open("all_troughs", "w") as at:
#             for y, t in zip(all_traces_y, all_troughs):
#                 print("%f %d %s" % (y, len(t), " ".join(["%.2f" % f for f in t])), file=at)
#         all_troughs = numpy.array(all_troughs)

        # figure out the outer edge of the left & rightmost fibers
        self.logger.info("Finding outer edges")
        far_left = centers[:, 0].reshape((-1, 1)) - 0.5 * avg_peak2peak_vertical
        far_right = centers[:, -1].reshape((-1, 1)) + 0.5 * avg_peak2peak_vertical
        all_lefts = numpy.hstack([far_left, valleys])
        all_rights = numpy.hstack([valleys, far_right])
        if (self.debug):
            numpy.savetxt("all_lefts", numpy.hstack([all_traces_y.reshape((-1, 1)), all_lefts]))
            numpy.savetxt("all_rights", numpy.hstack([all_traces_y.reshape((-1, 1)), all_rights]))

        # Refine center positions -- instead of using the peak position use the weighted mean position
        # for computation, only use pixels between the left and right boundaries we just derived
        self.logger.info("Refining trace centroiding")
        centers_refined = numpy.zeros_like(centers, dtype=float)
        for fiberid in range(self.n_fibers):
            self.logger.debug("Refining centroiding for fiber %d" % (fiberid))
            # all_weighted = []
            for y_block in range(all_traces_y.shape[0]):
                y = all_traces_y[y_block]
                y1,y2 = y-dy,y+dy

                _l = all_lefts[y_block, fiberid]
                _r = all_rights[y_block, fiberid]
                if (numpy.isfinite(_l) and numpy.isfinite(_r)):
                    l = int(_l)
                    r = int(_r)
                    sel_flux = bgsub[y1:y2+1, l:r+1]
                    sel_x = ix[y1:y2+1, l:r+1]
                    good = numpy.isfinite(sel_flux)
                    weighted = numpy.sum((sel_flux * sel_x)[good]) / numpy.sum(sel_flux[good])
                    # print(y,l,r)
                    # all_weighted.append(weighted)
                else:
                    weighted = centers[y_block, fiberid]
                centers_refined[y_block, fiberid] = weighted
        if (self.debug):
            numpy.savetxt("centers_refined", numpy.hstack([all_traces_y.reshape((-1,1)),centers_refined]))

        hdr = pyfits.Header()
        hdr['DY'] = dy
        pyfits.HDUList([
            pyfits.PrimaryHDU(),
            pyfits.ImageHDU(data=all_traces_y, name='Y', header=hdr),
            pyfits.ImageHDU(data=centers_refined, name='CENTER'),
            pyfits.ImageHDU(data=all_lefts, name='LEFT'),
            pyfits.ImageHDU(data=all_rights, name='RIGHT'),
            pyfits.ImageHDU(data=centers, name='PEAKS'),
        ]).writeto("fiber_tracers_orig.fits", overwrite=True)

            # optional, for testing & development, create a plot showing the different modes
            # fig, ax = plt.subplots()
            # ax.set_title("fiber: %d" % (fiberid))
            # ax.scatter(all_traces_y, centers[:, fiber], s=1)
            # ax.scatter(all_traces_y, all_weighted, s=1)
            # ax.scatter(all_traces_y, all_lefts[:, fiber], s=1)
            # ax.scatter(all_traces_y, all_rights[:, fiber], s=1)
            # ax.plot(self.full_y, self.fullres_centers[:, fiber], c='grey', lw=2, alpha=0.5)


        # Now we have the coarsely sampled position along the fiber, upscale this to
        # full frame
        self.logger.debug("Upsampling fiber traces to full resolution")
        # TODO: Check if we need to do some rounding here to make sure we capture the left/right edge correctly
        y_dim = self.full_y.shape[0]
        self.fullres_left = numpy.full((y_dim, self.n_fibers), fill_value=numpy.nan)
        self.fullres_right = numpy.full((y_dim, self.n_fibers), fill_value=numpy.nan)
        self.fullres_centers = numpy.full((y_dim, self.n_fibers), fill_value=numpy.nan)
        self.fullres_peaks = numpy.full((y_dim, self.n_fibers), fill_value=numpy.nan)
        for fiber_id in range(self.n_fibers):
            for meas, full in zip([all_lefts, all_rights, centers_refined, centers],
                                  [self.fullres_left, self.fullres_right, self.fullres_centers, self.fullres_peaks]):
                polyfit = numpy.polyfit(all_traces_y, meas[:, fiber_id], deg=2)
                full[:, fiber_id] = numpy.polyval(polyfit, self.full_y)
        pyfits.HDUList([
            pyfits.PrimaryHDU(),
            pyfits.ImageHDU(data=self.fullres_centers, name='CENTER'),
            pyfits.ImageHDU(data=self.fullres_left, name='LEFT'),
            pyfits.ImageHDU(data=self.fullres_right, name='RIGHT'),
            pyfits.ImageHDU(data=self.fullres_peaks, name='PEAKS'),
        ]).writeto("fiber_tracers_fullres.fits", overwrite=True)

        self.logger.info("All done tracing fibers")

    def extract_lineprofiles(self, fiber_ids=None, supersample=10, sample_step=None):
        # by default generate all profiles
        if (fiber_ids is None):
            fiber_ids = numpy.arange(self.n_fibers)

        left = self.fullres_left.T
        center = self.fullres_centers.T
        right = self.fullres_right.T
        # print(left.shape, center.shape, right.shape)

        max_left = numpy.min(left - center)
        max_right = numpy.max(right - center)
        left95 = numpy.nanpercentile(left-center, 2)
        right95 = numpy.nanpercentile(right-center, 98)

        iy,ix = numpy.indices(self.bgsub.shape)

        self.fiber_profiles = {}

        for fiberid in fiber_ids:
            self.logger.debug("Extracting line profile for fiber %d" % (fiberid+1))

            cutout_left = numpy.floor(numpy.min(left[fiberid])).astype(int)
            cutout_right = numpy.floor(numpy.max(right[fiberid])).astype(int)
            # print("#FIBER",fiberid," cutouts:", cutout_left, cutout_right)
            self.logger.debug("Maximum extent of fiber: %d -- %d" % (cutout_left, cutout_right))

            # extract relevant part of flat
            strip_x = ix[:, cutout_left:cutout_right + 1]  # + cutout_left
            strip_y = iy[:, cutout_left:cutout_right + 1]  # + cutout_left
            strip_left = left[fiberid].reshape((-1, 1))
            strip_center = center[fiberid].reshape(-1, 1)
            strip_right = right[fiberid].reshape((-1, 1))
            strip_dx = strip_x - strip_center
            # print("#FIBER", fiberid, strip_center.shape, strip_left.shape, strip_right.shape, strip_dx.shape)


            # mask out pixels in strip that are not part of this line
            strip_mask = (strip_x > strip_left) & (strip_x <= (strip_right))
            strip_flat = self.bgsub[:, cutout_left:cutout_right + 1].copy()
            strip_flat[~strip_mask] = numpy.nan

            # determine final grid
            step = 1. / supersample
            min_dx = numpy.nanmin(strip_dx)
            max_dx = numpy.nanmax(strip_dx)
            # print(min_dx, max_dx)
            min_dx_int = int(numpy.floor(min_dx))
            max_dx_int = int(numpy.ceil(max_dx))
            # print(min_dx_int, max_dx_int)
            self.logger.debug("profile range for fiber %d: %.3f -- %.3f" % (fiberid+1, min_dx_int, max_dx_int))

            if (sample_step is None):
                sample_step = step  # 0.5

            # generate output dx-grid
            ss_x = numpy.arange(min_dx_int, max_dx_int, step, dtype=float) + 0.5 * step
            ss_flat = numpy.zeros_like(ss_x)

            for i, _x in enumerate(ss_x):
                # select pixels relevant for this step along the profile
                sel = (strip_dx > (_x - sample_step)) & (strip_dx <= (_x + sample_step))
                sel_flat = strip_flat[sel]

                # generate average profile amplitude for this position, rejecting outliers as good as possible
                good = numpy.isfinite(sel_flat)
                try:
                    for iter in range(3):
                        stats = numpy.nanpercentile(sel_flat[good], [16, 50, 84])
                        _sigma = 0.5 * (stats[2] - stats[0])
                        _med = stats[1]
                        good = good & (sel_flat > (_med - 3 * _sigma)) & (sel_flat < (_med + 3 * _sigma))
                        if (numpy.sum(good) <= 0):
                            break
                    # print(i, _x, stats, _med, _sigma)
                except IndexError:
                    # Ignore IndexErrors -- most likely this means there are no pixels for this dx position
                    pass

                med = numpy.nanmedian(sel_flat[good])
                ss_flat[i] = med

            self.fiber_profiles[fiberid] = (ss_x, ss_flat)

        # end:: for fiberid in ....

    def save_lineprofiles(self, lineprofiles=None, filename=None):
        self.logger.debug("Saving line profiles")
        if (lineprofiles is None):
            # by default use the internal line profiles
            lineprofiles = self.fiber_profiles

        if (lineprofiles is None):
            self.logger.critical("No line profiles found for saving")
            return

        # generate multi-extension FITS for output, one extension for each line profile
        hdulist = [pyfits.PrimaryHDU()]
        for fiber, (x,prof) in lineprofiles.items():
            hduext = pyfits.ImageHDU(data=prof, name="FIBER_%03d" % (fiber+1))
            hdr = hduext.header
            hdr['CRPIX1'] = 1
            hdr['CRVAL1'] = x[0]
            hdr['CD1_1'] = x[1] - x[0]
            hdr['CDELT1'] = x[1] - x[0]
            hdr['CTYPE1'] = 'LINEAR'
            hdr['OBJECT'] = "average line profile -- fiber: %03d" % (fiber+1)
            hdulist.append(hduext)
        hdulist = pyfits.HDUList(hdulist)
        if (filename is not None):
            hdulist.writeto(filename, overwrite=True)
        return hdulist

    def get_fiber_mask(self, imgdata, fiber_id):
        iy, ix = numpy.indices(imgdata.shape)
        in_this_fiber = (ix > self.fullres_left[:, fiber_id].reshape((-1, 1))) & (
                ix <= self.fullres_right[:, fiber_id].reshape((-1, 1)))
        return in_this_fiber

    def extract_fiber_spectra(self, imgdata, weights=None, vmin=0, vmax=75000, fibers=None,
                              plot=False, optimize=True, extraction_mode='optimal'):
        # extract all fibers
        self.logger.info("Using extraction mode: %s" % (extraction_mode))
        iy, ix = numpy.indices(imgdata.shape)
        # print(ix.shape)
        # print(fullres_left[:, 0].shape)

        if (weights is None):
            weights = numpy.ones_like(imgdata, dtype=float)
        #        weights = bgsub

        fiber_specs = numpy.full((self.n_fibers, self.full_y.shape[0]), fill_value=numpy.nan)

        if fibers is None:
            fibers = numpy.arange(self.n_fibers)

        y,x = numpy.indices(imgdata.shape, dtype=float)
        left = self.fullres_left.T
        center = self.fullres_centers.T
        right = self.fullres_right.T

        profile_fit_width = 0
        profile_fit_mode = None
        profile_fit_modes = ['fit', 'clip']
        clipped = imgdata.copy()
        if (extraction_mode.startswith("profile")):
            try:
                items = extraction_mode.split(".")
                profile_fit_mode = items[1]
                profile_fit_width = int(items[2])
            except:
                self.logger.warning("Unable to interpret extraction mode '%s'" % (extraction_mode))
                pass
            if (profile_fit_mode not in profile_fit_modes):
                self.logger.warning("Illegal profile mode (%s), allowed are: %s; reverting to 'clip'" % (profile_fit_mode, ", ".join(profile_fit_modes)))
                profile_fit_mode = 'clip'

        for fiberid in fibers:  # range(n_fibers):
            #in_this_fiber = (ix > self.fullres_left[:, fiber_id].reshape((-1, 1))) & (
            #        ix <= self.fullres_right[:, fiber_id].reshape((-1, 1))) &

            cutout_left = numpy.floor(numpy.min(left[fiberid])).astype(int)
            cutout_right = numpy.ceil(numpy.max(right[fiberid])).astype(int)
            strip_x = x[:, cutout_left:cutout_right+1]
            strip_left = left[fiberid].reshape((-1,1))
            strip_center = center[fiberid].reshape(-1,1)
            strip_right = right[fiberid].reshape((-1,1))
            strip_dx = strip_x - strip_center
            strip_image = imgdata[:, cutout_left:cutout_right+1].copy()
            strip_mask = (strip_x > strip_left) & (strip_x <= (strip_right+1))
            self.logger.debug("extracting spectrum for fiber %d (x=%d...%d)" % (fiberid+1, cutout_left, cutout_right))

            strip_weights = weights[:, cutout_left:cutout_right+1].copy()
            if (optimize and fiberid in self.fiber_profiles):
                dx,profile = self.fiber_profiles[fiberid]
                strip_weights = numpy.interp(strip_dx, dx, profile)

            # mask out pixels outside the assigned range
            strip_image[~strip_mask] = numpy.nan
            strip_weights[~strip_mask] = numpy.nan

            #in_this_fiber = self.get_fiber_mask(imgdata, fiber_id) & (weights > 0)

            #_mf = weights.copy()
            #_spec = imgdata.copy()
            #_mf[~in_this_fiber] = numpy.nan
            #_spec[~in_this_fiber] = numpy.nan

            # pyfits.HDUList([
            #     pyfits.PrimaryHDU(),
            #     pyfits.ImageHDU(data=_mf)
            # ]).writeto("pre_integration_fiber=%d.fits" % (fiber_id), overwrite=True)

            # weighted = numpy.nansum(_spec * _mf, axis=1) / numpy.nansum(_mf, axis=1)
            if (extraction_mode == 'optimal'):
                extraction = numpy.nansum(strip_image * strip_weights, axis=1) / numpy.nansum(strip_weights, axis=1)
            elif (extraction_mode == 'median'):
                normalized = strip_image / strip_weights
                extraction = numpy.nanmedian(normalized, axis=1)
            elif (extraction_mode.startswith("profile")):
                ny = strip_image.shape[0]
                extraction = numpy.full((strip_image.shape[0]), fill_value=numpy.nan, dtype=float)
                profile_x, profile = self.fiber_profiles[fiberid]
                # dy = profile_fit_width
                for y in range(ny):
                    # get rows of spectra
                    y1 = numpy.max([0, y-profile_fit_width])
                    y2 = numpy.min([ny, y+profile_fit_width+1])
                    dx = strip_dx[y1:y2, :]
                    flux = strip_image[y1:y2, :]

                    # normalize each row with the profile intensity
                    _prof = numpy.interp(dx, profile_x, profile)
                    flux_prof = flux/_prof

                    # reject outliers
                    good = numpy.isfinite(flux_prof)
                    for i in range(3):
                        try:
                            _stats = numpy.nanpercentile(flux_prof[good], [16,50,84])
                            _med = _stats[1]
                            _sigma = 0.5*(_stats[2]-_stats[0])
                            good = good & (flux_prof > (_med-3*_sigma)) & (flux_prof < (_med+3*_sigma))
                        except:
                            continue

                    if (profile_fit_mode == 'fit'):
                        extraction[y] = _med
                    elif (profile_fit_mode == 'clip'):
                        useful = good[profile_fit_width]
                        clip_profile = _prof[profile_fit_width][useful]
                        clip_flux = flux[profile_fit_width][useful]
                        extraction[y] = numpy.nansum(clip_flux * clip_profile) / numpy.nansum(clip_profile)

                        # save output of clipped data
                        strip_clip = clipped[:, cutout_left:cutout_right+1]
                        strip_clip[y, :][~good[profile_fit_width]] = numpy.nan

                #end for _y
            # end processing single fiber

            fiber_specs[fiberid] = extraction
            # pyfits.PrimaryHDU(data=in_this_fiber.astype(int)).writeto("fibermask_%d.fits" % (fiber_id+1), overwrite=True)

            if (plot):
                fig, ax = plt.subplots()
                # ax.imshow(in_this_fiber.astype(int), origin='lower')
                # ax.scatter(full_y, weighted, s=0.2, label='weighted')
                # ax.scatter(full_y, _sum/4, s=0.2, label='sum')
                ax.plot(self.full_y, extraction, lw=0.3)
                # ax.legend()
                ax.set_ylim((vmin, vmax))
                fig.savefig("fiber_extraction_%03d.png" % (fiberid+1))

        return fiber_specs, clipped

    def get_fiber_spacing(self):
        frc = self.fullres_centers
        # total_dx = frc[frc.shape[0] // 2, self.n_fibers-1] - frc[frc.shape[0] // 2, 0]
        total_dx = self.get_mean_fiber_position(fiber_id=self.n_fibers-1) - self.get_mean_fiber_position(fiber_id=0)
        dx = total_dx / (self.n_fibers-1)
        return dx

    def get_mean_fiber_position(self, fiber_id):
        if (fiber_id == 'all'):
            return numpy.mean(self.fullres_centers, axis=0)
        else:
            return numpy.mean(self.fullres_centers[:, fiber_id])

    def get_sky_fiber_ids(self):
        if (self.sky_fiber_ids is not None):
            return self.sky_fiber_ids
        raise ValueError("No sky fiber IDs defined")

    def reorder_fibers(self, fiberspecs):
        return fiberspecs.copy()

    def grating_from_header(self, *args, **kwargs):
        return Grating(*args, **kwargs)

    @classmethod
    def load_raw_file(cls, filename, logger=None):
        if (logger is not None):
            logger.info("Loading file %s (%s)" % (filename, cls.name))
        hdulist = pyfits.open(filename)

        data = hdulist[cls.input_ext_data].data.astype(float)
        if (cls.input_transpose):
            data = data.T
        if (cls.input_flipx):
            data = data[:, ::-1]
        if (cls.input_flipy):
            data = data[::-1, :]

        header = hdulist[cls.input_ext_header]

        return data, header

# class SparsepakFiberSpecs( GenericFiberSpecs ):
#     n_fibers = 82
#
#     _sky_fibers = [22, 16, 2, 37, 54, 80, 70]
#     sky_fiber_ids = numpy.array(_sky_fibers, dtype=int) - 1
#
#     pass

