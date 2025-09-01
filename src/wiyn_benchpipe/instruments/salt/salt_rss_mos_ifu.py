
import numpy

from ...grating import Grating
from ...fibertraces import GenericFiberSpecs


class SALT_RSS_IFU_FiberSpecs( GenericFiberSpecs ):
    n_fibers = 327
    ref_fiber_id = 175
    name = "RSS-VIS IFU @ SALT"

    # TODO: Handle different binnings here
    trace_minx = 90
    trace_maxx = 1950

    def grating_from_header(self, header, *args, **kwargs):
        grating = header['GRATING']
        if (grating == 'PG0900'):
            return RSS_SALT_RSS_PG0900(header, *args, **kwargs)



class RSS_SALT_RSS ( Grating ):
    name = "SALT-RSSVIS-PG0900"

    ccd_npixels_x = 4102
    ccd_npixels_y = 6344
    ccd_x_bin = 1
    ccd_y_bin = 1
    ccd_pixelsize = 18e-6   # 12 micron pixels
    lines_per_mm = 900

    collimator_focal_length = 629e-3    # TODO: Check & correct these numbers
    camera_focal_length = 229e-3
    camera_magnification = 1.5 #collimator_focal_length / camera_focal_length

    def __init__(self, header, midline_x=None, grating_angle=None):
        super().__init__(header=header, midline_x=midline_x)

        # Read all relevant keywords from header
        self.header = header
        self.grating_order = 1 # ???? header['GRATORD']
        self.grating_angle = 0
        if (grating_angle is not None):
            self.grating_angle = grating_angle
        elif ('GR-ANGLE' in header):
            self.grating_angle = header['GR-ANGLE']
        elif ('GRTILT' in header):
            self.grating_angle = header['GRTILT']
        else:
            self.logger.critical("Unable to get grating angle from FITS header")
            raise ValueError("Unable to get grating angle from FITS")
        #self.grating_angle = grating_angle if grating_angle is not None else header['GR-ANGLE'] # checked
        self.camera_collimator_angle = header['CAMANG']
        self.logger.debug("angle setup: grating: %.4f; cam: %.4f; order: %d" % (
            self.grating_angle, self.camera_collimator_angle, self.grating_order))

        ccdsum = header['CCDSUM'].strip().split()
        self.ccd_x_bin = int(ccdsum[0])
        self.ccd_y_bin = int(ccdsum[1])
        self.logger.debug("CCD config: bin-X: %d; bin-Y: %d" % (self.ccd_x_bin, self.ccd_y_bin))

        if (midline_x is None):
            midline_x = self.ccd_npixels_x / self.ccd_x_bin / 2.
        self.midline_x = midline_x

        self.line_spacing = 1e7 / self.lines_per_mm
        # print("line spacing:", self.line_spacing)
        self.output_angle = self.grating_angle + self.camera_collimator_angle
        self.grating_camera_distance = self.collimator_focal_length
            #0.795 # should be 0.776 based on design, but the other value yields better results

        self.ccd_n_pixels_binned = self.ccd_npixels_y / self.ccd_y_bin
        self.ccd_pixelsize_binned = self.ccd_pixelsize * self.ccd_y_bin

        self.y = numpy.arange(self.ccd_n_pixels_binned)
        self.y0 = self.y - (self.ccd_n_pixels_binned / 2) # relativ to center of chip

        self.alpha = numpy.deg2rad(self.grating_angle)
        self.beta = numpy.deg2rad(self.camera_collimator_angle - self.grating_angle)

        self.compute()

    # def compute(self):
    #     self.logger.info("Computing wavelength solution using grating equation")
    #     self.alpha = numpy.deg2rad(self.grating_angle)
    #     self.beta = numpy.deg2rad(self.camera_collimator_angle - self.grating_angle)
    #
    #     # calculate central wavelength
    #     self.central_wavelength = self.line_spacing / self.grating_order * (numpy.sin(self.alpha) + numpy.sin(self.beta))
    #     self.logger.info("central wavelength = %f" % (self.central_wavelength))
    #
    #     # full wavelength solution (WL for each y-value)
    #     angle_offset = numpy.arctan(self.y0 * self.ccd_pixelsize_binned / self.grating_camera_distance) * self.camera_magnification
    #     self.wavelength_solution = self.line_spacing / self.grating_order * (
    #         numpy.sin(self.alpha) + numpy.sin(self.beta - angle_offset)
    #     )
    #
    #     # we can also provide a quick polynomial fit
    #     self.wl_polyfit = numpy.polyfit(self.y0, self.wavelength_solution, deg=2)
    #
    #     self.wl_blueedge = numpy.min(self.wavelength_solution)
    #     self.wl_rededge = numpy.max(self.wavelength_solution)
    #
    #     return

class RSS_SALT_RSS_PG0900 (RSS_SALT_RSS):
    name = "SALT-RSSVIS-PG0900"
    lines_per_mm = 900