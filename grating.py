import numpy
import logging

class Grating(object):
    name = "default"
    lines_per_mm = 1

    ccd_npixels = 4000 # in spectral direction
    ccd_pixelsize = 12e-6   # 12 micron pixels

    central_wavelength = numpy.nan

    def __init__(self, header):
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

        self.line_spacing = 1e7 / self.lines_per_mm
        print(self.line_spacing)
        self.output_angle = self.grating_angle - self.camera_collimator_angle
        self.grating_camera_distance = 0.776

        self.ccd_n_pixels_binned = self.ccd_npixels / self.ccd_y_bin
        self.ccd_pixelsize_binned = self.ccd_pixelsize * self.ccd_y_bin

        self.y = numpy.arange(self.ccd_n_pixels_binned)
        self.y0 = self.y - self.ccd_n_pixels_binned / 2

        self.compute()

    def compute(self):
        self.logger.info("Computing wavelength solution using grating equation")
        self.alpha = numpy.deg2rad(self.grating_angle)
        self.beta = numpy.deg2rad(self.grating_angle - self.camera_collimator_angle)

        # calculate central wavelength
        self.central_wavelength = self.line_spacing / self.grating_order * (numpy.sin(self.alpha) + numpy.sin(self.beta))
        self.logger.info("central wavelength = %f" % (self.central_wavelength))

        # full wavelength solution (WL for each y-value)
        angle_offset = numpy.arctan(self.y0 * self.ccd_pixelsize_binned / self.grating_camera_distance) * 2.78
        self.wavelength_solution = self.line_spacing / self.grating_order * (
            numpy.sin(self.alpha) + numpy.sin(self.beta - angle_offset)
        )

        # we can also provide a quick polynomial fit
        self.wl_polyfit = numpy.polyfit(self.y0, self.wavelength_solution, deg=2)

        self.wl_blueedge = numpy.min(self.wavelength_solution)
        self.wl_rededge = numpy.max(self.wavelength_solution)

        return

class Grating_Echelle316(Grating):
    lines_per_mm = 316
    name = "316@63"



def grating_from_header(header, **kwargs):

    grating_name = header['GRATNAME']
    if (grating_name == '316@63.4'):
        return Grating_Echelle316(header, **kwargs)
    else:
        return Grating(header, **kwargs)
