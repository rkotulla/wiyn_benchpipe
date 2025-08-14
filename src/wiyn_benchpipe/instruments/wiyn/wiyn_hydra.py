import numpy
import astropy.coordinates as coord
import astropy.units as u
import astropy.io.fits as pyfits
import astropy.stats

from ...fibertraces import GenericFiberSpecs
from .wiyn_bench import wiyn_grating_from_header, WIYNBenchFiberSpecs

class WiynHydraFiberSpecs(WIYNBenchFiberSpecs):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.sky_fiber_ids = []

        if (self.header is not None):
            _n_fibers = 0
            for i in range(1000):
                slfib_key = "SLFIB%d" % (i+1)
                if (slfib_key not in self.header):
                    break
                slfib = self.header[slfib_key]
                print(slfib)
                if (slfib.strip() == ""):
                    break
                if (slfib.lower().find("broken fiber") > 0):
                    self.logger.debug("Marking fiber %d as broken", i+1)
                    continue
                elif (slfib.lower().find("sky") > 0):
                    self.sky_fiber_ids.append(i-1)
                items = slfib.split()
                fiber_type = int(items[1])
                # if (fiber_type >= 0):
                _n_fibers += 1
            #self.n_fibers = _n_fibers
            self.logger.info("Updating #fibers from header: %d" % self.n_fibers)
            self.logger.info("Number of sky fibers: %d" % (len(self.sky_fiber_ids)))

    def get_sky_fiber_ids(self, filelist=None, *args, **kwargs):
        sky_fibers = []
        if (filelist is not None):
            filename = filelist[0]
            hdulist = pyfits.open(filename)
            hdr = hdulist[0].header
            fiberid = 0
            for i in range(1000):
                slfib_key = "SLFIB%d" % (i+1)
                if (slfib_key not in self.header):
                    break
                slfib = hdr[slfib_key]
                if (slfib.lower().find("sky") > 0):
                    sky_fibers.append(fiberid)
                elif (slfib.lower().find("broken fiber") > 0 or
                      slfib.lower().find("gap") > 0):
                    continue
                fiberid += 1
        sky_fibers = [self.n_fibers - i for i in sky_fibers]
        self.logger.info("Using %d sky fibers (%s)" % (len(sky_fibers), ", ".join(["%d" % i for i in sky_fibers])))
        return sky_fibers

    def interpolate_missing_fibers(self):
        self.logger.info("Interpolating missing fibers...")
        centers = self.fullres_centers


        y_center = centers.shape[0]//2

        midframe_centers = centers[y_center, :]
        print(midframe_centers)

        y0 = numpy.min(midframe_centers)
        spacings = numpy.diff(midframe_centers)

        mean_spacing, median_spacing, std_spacing = astropy.stats.sigma_clipped_stats(spacings)
        # print(mean_spacing, median_spacing, std_spacing)

        fiberid = numpy.arange(centers.shape[1])
        n_fibers = (numpy.max(midframe_centers) - y0) / median_spacing
        filled_fiberid = numpy.round((midframe_centers - y0) / median_spacing)

        observed_fiber_ids = numpy.arange(centers.shape[1])
        corrected_fiber_ids = []
        actual_fiber_ids = []

        self.logger.debug("Checking for missing fibers based on trace position & spacing")
        for f in range(numpy.ceil(n_fibers).astype(int) + 1):
            # check if we already have a fiber at this position

            model_pos = f * median_spacing + y0
            diff = (midframe_centers - model_pos)
            abs_diff = numpy.fabs(diff)
            if (numpy.min(abs_diff) < 0.4 * median_spacing):
                # print("Found this fiber")
                found_at = numpy.argmin(abs_diff)
                # print(found_at)
                act_fiber = observed_fiber_ids[numpy.argmin(abs_diff)]
                actual_fiber_ids.append(act_fiber)
            else:
                # print("Found gap")
                actual_fiber_ids.append(-1)
            corrected_fiber_ids.append(f)

        #centers = hdulist['CENTER'].data
        lefts = self.fullres_left
        rights = self.fullres_right
        peaks = self.fullres_peaks

        empty_count = 0
        prev = 0
        empty_list = []

        corrected_centers = []
        corrected_lefts = []
        corrected_rights = []
        corrected_peaks = []

        good_traces = []
        for i, afi in enumerate(actual_fiber_ids):
            if (afi < 0):
                empty_count += 1
                empty_list.append(i)
                empty_left = prev
                empty_right = prev+1
            else:
                for empty in range(empty_count):

                    frac = (empty+1.) / (empty_count+1.)
                    corrected_centers.append( centers[:, empty_left] + frac * (centers[:, empty_right] - centers[:, empty_left]) )
                    corrected_peaks.append(     peaks[:, empty_left] + frac * (  peaks[:, empty_right] -   peaks[:, empty_left]) )
                    good_traces.append(False)
                    print(empty_left + frac, end=' ')
                empty_count = 0
                empty_list = []
                prev = afi
                print(afi, end=' ')
                #for x in range(len(observed_traces)):

                corrected_centers.append(centers[:, afi])
                corrected_peaks.append(peaks[:, afi])
                good_traces.append(True)

                    # corrected_centers.append(centers[:, afi])
        print("\n done!")

        # Now we have
        corrected_centers = numpy.array(corrected_centers).T
        corrected_peaks = numpy.array(corrected_peaks).T

        middle_centers = 0.5*(corrected_centers[:,1:]+corrected_centers[:,:-1])
        corrected_lefts = numpy.hstack([lefts[:,:1], middle_centers])
        corrected_rights = numpy.hstack([middle_centers, rights[:,-1:]])
        print(corrected_centers.shape, corrected_lefts.shape, corrected_rights.shape)

        # move all trace information back to actual variables
        self.fullres_peaks = corrected_peaks
        self.fullres_centers = corrected_centers
        self.fullres_left = corrected_lefts
        self.fullres_right = corrected_rights

        # also update the number of fibers
        self.n_fibers = corrected_centers.shape[1]
        self.logger.info("Done interpolating missing traces, full fiber count is now %d" % (self.n_fibers))


class WiynHydraRedFiberSpecs( WiynHydraFiberSpecs ):
    n_fibers = 89
    name = "Hydra Red Cable @ WIYN"
    ref_fiber_id = 41

    trace_minx = 120
    trace_maxx = 2520

class WiynHydraBlueFiberSpecs( WiynHydraFiberSpecs ):
    n_fibers = 87
    name = "Hydra Blue Cable @ WIYN"
    ref_fiber_id = 45

    trace_minx = 120
    trace_maxx = 2520

