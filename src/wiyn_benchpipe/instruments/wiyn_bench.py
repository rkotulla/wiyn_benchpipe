
import numpy
import logging

from ..grating import Grating
from ..fibertraces import GenericFiberSpecs

class WIYNBenchGrating( Grating ):
    name = "default"
    lines_per_mm = 1

    ccd_npixels_y = 4000 # in spectral direction
    ccd_npixels_x = 2808 # horizon direction
    ccd_pixelsize = 12e-6   # 12 micron pixels

    central_wavelength = numpy.nan
    camera_magnification = 2.78

    def __init__(self, header, midline_x=None):
        self.logger = logging.getLogger(self.name)

        self.header = header
        self.grating_order = header['GRATORD']
        self.grating_angle = header['GRATANGL']
        self.camera_collimator_angle = header['CAMANGLE']
        self.logger.debug("angle setup: grating: %.4f; cam: %.4f; order: %d" % (
            self.grating_angle, self.camera_collimator_angle, self.grating_order))

        self.ccd_x_bin = header['CCDXBIN']
        self.ccd_y_bin = header['CCDYBIN']
        self.logger.debug("CCD config: bin-X: %d; bin-Y: %d" % (self.ccd_x_bin, self.ccd_y_bin))

        if (midline_x is None):
            midline_x = self.ccd_npixels_x / self.ccd_x_bin / 2.
        self.midline_x = midline_x

        self.line_spacing = 1e7 / self.lines_per_mm
        print(self.line_spacing)
        self.output_angle = self.grating_angle - self.camera_collimator_angle
        self.grating_camera_distance = 0.795 # should be 0.776 based on design, but the other value yields better results

        self.ccd_n_pixels_binned = self.ccd_npixels_y / self.ccd_y_bin
        self.ccd_pixelsize_binned = self.ccd_pixelsize * self.ccd_y_bin

        self.y = numpy.arange(self.ccd_n_pixels_binned)
        self.y0 = self.y - (self.ccd_n_pixels_binned / 2) # relativ to center of chip

        self.compute()

    def compute(self):
        self.logger.info("Computing wavelength solution using grating equation")
        self.alpha = numpy.deg2rad(self.grating_angle)
        self.beta = numpy.deg2rad(self.grating_angle - self.camera_collimator_angle)

        # calculate central wavelength
        self.central_wavelength = self.line_spacing / self.grating_order * (numpy.sin(self.alpha) + numpy.sin(self.beta))
        self.logger.info("central wavelength = %f" % (self.central_wavelength))

        # full wavelength solution (WL for each y-value)
        angle_offset = numpy.arctan(self.y0 * self.ccd_pixelsize_binned / self.grating_camera_distance) * self.camera_magnification
        self.wavelength_solution = self.line_spacing / self.grating_order * (
            numpy.sin(self.alpha) + numpy.sin(self.beta - angle_offset)
        )

        # we can also provide a quick polynomial fit
        self.wl_polyfit = numpy.polyfit(self.y0, self.wavelength_solution, deg=2)

        self.wl_blueedge = numpy.min(self.wavelength_solution)
        self.wl_rededge = numpy.max(self.wavelength_solution)

        return

    def wavelength_from_xy(self, x=None, x0=None, y=None, y0=None):
        if (y0 is None):
            if (y is None):
                y0 = 0
            else:
                y0 = y - (self.ccd_npixels_y / 2 / self.ccd_x_bin)

        if (x0 is None):
            if (x is None):
                x0 = 0
            else:
                x0 = x - self.midline_x #(self.ccd_npixels_x / 2 / self.ccd_x_bin)

        x0_phys = x0 * self.ccd_x_bin * self.ccd_pixelsize
        y0_phys = y0 * self.ccd_y_bin * self.ccd_pixelsize
        angle_dx = numpy.arctan(x0_phys / self.grating_camera_distance) * self.camera_magnification
        angle_dy = numpy.arctan(y0_phys / self.grating_camera_distance) * self.camera_magnification
        wavelength = self.line_spacing / self.grating_order * numpy.cos(angle_dx) * (
            numpy.sin(self.alpha) + numpy.sin(self.beta - angle_dy)
        )
        return wavelength

    def compute_wl_offset(self, y, dx):
        y_2d, x_2d = numpy.indices((self.ccd_npixels_y, self.ccd_npixels_x))
        y_2d0 = (y_2d - (self.ccd_npixels_y / 2)) * self.ccd_pixelsize
        x_2d0 = (x_2d - (self.ccd_npixels_x / 2)) * self.ccd_pixelsize
        dr = numpy.hypot(x_2d0, y_2d0)
        angle_offset = numpy.arctan(dr / self.grating_camera_distance) * 2.78
        wavelength = self.line_spacing / self.grating_order * (
            numpy.sin(self.alpha) + numpy.sin(self.beta - angle_offset)
        )
        return wavelength


class Grating_Echelle316(WIYNBenchGrating):
    lines_per_mm = 316
    name = "316@63"



def wiyn_grating_from_header(header, **kwargs):

    grating_name = header['GRATNAME']
    if (grating_name == '316@63.4'):
        return Grating_Echelle316(header, **kwargs)
    else:
        return Grating(header, **kwargs)




class WIYNBenchFiberSpecs( GenericFiberSpecs ):
    n_fibers = 82
    name = "SparsePak @ WIYN"
    ref_fiber_id = 41


    def grating_from_header(self, header):
        return wiyn_grating_from_header(header)
