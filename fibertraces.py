

import numpy
import logging
import scipy.ndimage
import scipy.signal
import pandas
import matplotlib.pyplot as plt


class GenericFiberSpecs(object):

    n_fibers = -1

    def __init__(self, logger=None):
        if (self.n_fibers < 0):
            raise ValueError("Invalid number of fibers (%d) -- don't use the base class!" % (self.n_fibers))

        if (logger is None):
            logger = logging.getLogger('FiberSpecs')
        self.logger = logger

        return

    def find_trace_fibers(self, trace_image):

        if (trace_image is None):
            raise ValueError("Need to provide a trace_image!")
        self.trace_image = trace_image


        self.full_y = numpy.arange(trace_image.shape[0])
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
        left_edge = 80  ## adjust this for binning, assuming 4x3
        right_edge = 570
        w = 10

        left = numpy.mean(min_filter[:, left_edge - w:left_edge + w], axis=1).reshape((-1, 1))
        right = numpy.mean(min_filter[:, right_edge - w:right_edge + w], axis=1).reshape((-1, 1))
        slope = (right - left) / (right_edge - left_edge)
        iy, ix = numpy.indices(trace_image.shape)
        gradient_2d = (ix - left_edge) * slope + left

        # subtract the modeled background
        bgsub = trace_image - gradient_2d
        bgsub[bgsub < 0] = 0
        self.bgsub = bgsub  # TODO: fix this with proper variable name

        # Now trace the fibers
        self.logger.debug("Tracing ridge-lines of each fiber")
        dy = 5  ## adjust for binning
        traces = pandas.DataFrame()
        all_peaks = []
        all_traces_y = []
        for y in range(dy, bgsub.shape[0], 2 * dy):
            prof = numpy.nanmean(bgsub[y - dy:y + dy, :], axis=0)
            peak_intensity = numpy.mean(prof)
            # print(y, peak_intensity)

            peaks, peak_props = scipy.signal.find_peaks(prof, height=0.5 * peak_intensity, distance=3)
            if (peaks.shape[0] != self.n_fibers):  # adjust for other instruments -- 82 is for sparsepak
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
        self.fullres_left = numpy.full((y_dim, self.n_fibers), fill_value=numpy.NaN)
        self.fullres_right = numpy.full((y_dim, self.n_fibers), fill_value=numpy.NaN)
        self.fullres_centers = numpy.full((y_dim, self.n_fibers), fill_value=numpy.NaN)
        for fiber_id in range(self.n_fibers):
            for meas, full in zip([all_lefts, all_rights, centers],
                                  [self.fullres_left, self.fullres_right, self.fullres_centers]):
                polyfit = numpy.polyfit(all_traces_y, meas[:, fiber_id], deg=2)
                full[:, fiber_id] = numpy.polyval(polyfit, self.full_y)

        self.logger.info("All done tracing fibers")

    def extract_fiber_spectra(self, imgdata, weights=None, vmin=0, vmax=75000, fibers=None, plot=False):
        # extract all fibers
        iy, ix = numpy.indices(imgdata.shape)
        # print(ix.shape)
        # print(fullres_left[:, 0].shape)

        if (weights is None):
            weights = numpy.ones_like(imgdata, dtype=float)
        #        weights = bgsub

        fiber_specs = numpy.full((self.n_fibers, self.full_y.shape[0]), fill_value=numpy.NaN)

        if fibers is None:
            fibers = numpy.arange(self.n_fibers)

        for fiber_id in fibers:  # range(n_fibers):
            in_this_fiber = (ix > self.fullres_left[:, fiber_id].reshape((-1, 1))) & (
                    ix < self.fullres_right[:, fiber_id].reshape((-1, 1))) & (weights > 0)

            _mf = weights.copy()
            _spec = imgdata.copy()
            _mf[~in_this_fiber] = numpy.NaN
            _spec[~in_this_fiber] = numpy.NaN

            # pyfits.HDUList([
            #     pyfits.PrimaryHDU(),
            #     pyfits.ImageHDU(data=_mf)
            # ]).writeto("pre_integration_fiber=%d.fits" % (fiber_id), overwrite=True)

            weighted = numpy.nansum(_spec * _mf, axis=1) / numpy.nansum(_mf, axis=1)
            # print(weighted.shape)
            fiber_specs[fiber_id] = weighted

            _sum = numpy.nansum(_spec, axis=1)

            if (not plot):
                continue

            fig, ax = plt.subplots()
            # ax.imshow(in_this_fiber.astype(int), origin='lower')
            # ax.scatter(full_y, weighted, s=0.2, label='weighted')
            # ax.scatter(full_y, _sum/4, s=0.2, label='sum')
            ax.plot(self.full_y, weighted, lw=0.3)
            # ax.legend()
            ax.set_ylim((vmin, vmax))

        return fiber_specs

    def get_fiber_spacing(self):
        frc = self.fullres_centers
        # total_dx = frc[frc.shape[0] // 2, self.n_fibers-1] - frc[frc.shape[0] // 2, 0]
        total_dx = self.get_mean_fiber_position(fiber_id=self.n_fibers-1) - self.get_mean_fiber_position(fiber_id=0)
        dx = total_dx / (self.n_fibers-1)
        return dx

    def get_mean_fiber_position(self, fiber_id):
        return numpy.mean(self.fullres_centers[:, fiber_id])



class SparsepakFiberSpecs( GenericFiberSpecs ):
    n_fibers = 82
    pass

