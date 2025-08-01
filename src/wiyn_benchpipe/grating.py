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
    camera_angle = 0

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
                y0 = y - (self.ccd_npixels_y / 2 / self.ccd_y_bin)

        if (x0 is None):
            if (x is None):
                x0 = 0
            else:
                x0 = x - self.midline_x  # (self.ccd_npixels_x / 2 / self.ccd_x_bin)

        x0_phys = x0 * self.ccd_x_bin * self.ccd_pixelsize
        y0_phys = y0 * self.ccd_y_bin * self.ccd_pixelsize
        angle_dx = numpy.arctan(x0_phys / self.grating_camera_distance) * self.camera_magnification
        angle_dy = numpy.arctan(y0_phys / self.grating_camera_distance) * self.camera_magnification
        wavelength = self.line_spacing / self.grating_order * numpy.cos(angle_dx) * (
                numpy.sin(self.alpha) + numpy.sin(self.beta - angle_dy)
        )
        return wavelength

    def y_from_wavelength(self, wavelength, return_y0=True, x0=0.0):
        x0_phys = x0 * self.ccd_x_bin * self.ccd_pixelsize
        angle_dx = numpy.arctan(x0_phys / self.grating_camera_distance) * self.camera_magnification

        sin_parts = wavelength / self.line_spacing * self.grating_order / numpy.cos(angle_dx)
        sin_beta_dy = sin_parts - numpy.sin(self.alpha)
        angle_dy = self.beta - numpy.arcsin(sin_beta_dy)

        y0_phys = numpy.tan(angle_dy / self.camera_magnification) * self.grating_camera_distance
        y0 = y0_phys / self.ccd_pixelsize / self.ccd_y_bin

        if (return_y0):
            return y0
        y = y0 + (self.ccd_npixels_y / 2 / self.ccd_y_bin)
        return y

    def compute_wl_offset(self, y, dx):
        y_2d, x_2d = numpy.indices((self.ccd_npixels_y//self.ccd_y_bin, self.ccd_npixels_x//self.ccd_x_bin))
        y_2d0 = (y_2d - (self.ccd_npixels_y / self.ccd_y_bin / 2)) * self.ccd_pixelsize * self.ccd_y_bin
        x_2d0 = (x_2d - (self.ccd_npixels_x / self.ccd_x_bin / 2)) * self.ccd_pixelsize * self.ccd_x_bin
        dr = numpy.hypot(x_2d0, y_2d0)
        angle_offset = numpy.arctan(dr / self.grating_camera_distance) * self.camera_magnification
        wavelength = self.line_spacing / self.grating_order * (
            numpy.sin(self.alpha) + numpy.sin(self.beta - angle_offset)
        )
        return wavelength

    def report(self):
        self.logger.info("setup: grating: %.2f deg // camera: %.2f deg // order: %d" % (
            self.grating_angle, self.camera_angle, self.grating_order
        ))
        self.logger.info("central wavelength: %f, dispersion %.3f A/px (range: %.1f -- %.1f)" % (
            self.central_wavelength, numpy.fabs(self.wl_polyfit[-2]),
            self.wl_blueedge, self.wl_rededge))


