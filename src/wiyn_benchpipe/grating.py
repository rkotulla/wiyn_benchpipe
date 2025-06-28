import numpy
import logging



class Grating(object):
    name = "default"
    lines_per_mm = 1

    ccd_npixels_y = 0 # in spectral direction
    ccd_npixels_x = 0 # horizon direction
    ccd_pixelsize = 0   # 12 micron pixels

    central_wavelength = numpy.nan
    camera_magnification = 2.78

    grating_order = 0
    grating_angle = 0
    camera_collimator_angle = 0
    ccd_x_bin = 1
    ccd_y_bin = 1
    midline_x = 0

    y0 = 0

    def __init__(self, header, midline_x=None):
        self.logger = logging.getLogger(self.name)


    def setup_from_header(self, header):
        return

    def compute(self):
        self.logger.info("Computing wavelength solution using grating equation")
        return

    def wavelength_from_xy(self, x=None, x0=None, y=None, y0=None):
        return 0

    def compute_wl_offset(self, y, dx):
        return 0


