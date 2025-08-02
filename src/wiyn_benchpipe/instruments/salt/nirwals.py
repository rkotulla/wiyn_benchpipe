import numpy
import astropy.coordinates as coord
import astropy.units as u
import astropy.io.fits as pyfits

from ...fibertraces import GenericFiberSpecs
from ...grating import Grating

class NirwalsFiberSpecs( GenericFiberSpecs ):
    n_fibers = 248
    ref_fiber_id = list(numpy.arange(110,120))
    name = "NIRWALS @ SALT"
    trace_minx = 4
    trace_maxx = 2044

    def grating_from_header(self, *args, **kwargs):
        return NirwalsGrating(*args, **kwargs)

    @classmethod
    def load_raw_file(cls, filename, logger=None):
        if (logger is not None):
            logger.info("Loading NIRWals frame %s" % (filename))
        max_good_rate = 25000

        hdulist = pyfits.open(filename)
        data = hdulist['SCI'].data.astype(float)

        corr = data.T
        corr[corr > max_good_rate] = max_good_rate

        return corr, hdulist[0].header




class NirwalsGrating( Grating ):
    name = "NirwalsSpec"

    ccd_npixels_x = 2048
    ccd_npixels_y = 2048
    ccd_x_bin = 1
    ccd_y_bin = 1
    ccd_pixelsize = 18e-6   # 12 micron pixels
    lines_per_mm = 950

    collimator_focal_length = 629e-3 #mm
    camera_focal_length = 229e-3
    camera_magnification = collimator_focal_length / camera_focal_length  # 2.7467248908296944
    # camera_magnification =
    def __init__(self, header, midline_x=None, grating_angle=None):
        # print("###\n"*3, "### NIRWALS GRATING", "\n###"*3)
        super().__init__(header=header, midline_x=midline_x)

        # Read all relevant keywords from header
        self.header = header
        self.grating_order = 1 # ???? header['GRATORD']
        self.grating_angle = 0
        if (grating_angle is not None):
            self.grating_angle = grating_angle
        elif ('GRRANGLE' in header):
            self.grating_angle = header['GRRANGLE']
        elif ('GR-ANGLE' in header):
            self.grating_angle = header['GR-ANGLE']
        elif ('GRTILT' in header):
            self.grating_angle = header['GRTILT']
        else:
            self.logger.critical("Unable to get grating angle from FITS header")
            raise ValueError("Unable to get grating angle from FITS")

        self.camera_angle = header['CAMANG']

        cam_focus = header['CFCFOCUS'] # microns
        self.camera_magnification = self.collimator_focal_length / (self.camera_focal_length - cam_focus*1e-6)

        #self.grating_angle = grating_angle if grating_angle is not None else header['GR-ANGLE'] # checked
        self.camera_collimator_angle = self.grating_angle #   header['GR-ANGLE'] # ??? CAMANGLE']
        self.logger.debug("angle setup: grating: %.4f; cam: %.4f; order: %d" % (
            self.grating_angle, self.camera_collimator_angle, self.grating_order))

        self.ccd_x_bin = 1
        self.ccd_y_bin = 1
        self.logger.debug("CCD config: bin-X: %d; bin-Y: %d" % (self.ccd_x_bin, self.ccd_y_bin))

        if (midline_x is None):
            midline_x = self.ccd_npixels_x / self.ccd_x_bin / 2.
        self.midline_x = midline_x

        self.line_spacing = 1e7 / self.lines_per_mm
        # print("line spacing:", self.line_spacing)
        self.output_angle = self.grating_angle #+ self.camera_collimator_angle
        self.grating_camera_distance = self.collimator_focal_length


        self.ccd_n_pixels_binned = self.ccd_npixels_y / self.ccd_y_bin
        self.ccd_pixelsize_binned = self.ccd_pixelsize * self.ccd_y_bin

        self.y = numpy.arange(self.ccd_n_pixels_binned)
        self.y0 = self.y - (self.ccd_n_pixels_binned / 2) # relativ to center of chip

        self.alpha = numpy.deg2rad(self.grating_angle)
        self.beta = numpy.deg2rad(self.grating_angle) #self.camera_angle - self.grating_angle)

        self.compute()

    # def compute(self):
    #     self.logger.info("Computing wavelength solution using grating equation")
    #     self.alpha = numpy.deg2rad(self.grating_angle)
    #     self.beta = numpy.deg2rad(self.camera_angle - self.grating_angle)
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

    # def wavelength_from_xy(self, x=None, x0=None, y=None, y0=None):
    #     if (y0 is None):
    #         if (y is None):
    #             y0 = 0
    #         else:
    #             y0 = y - (self.ccd_npixels_y / 2 / self.ccd_x_bin)
    #
    #     if (x0 is None):
    #         if (x is None):
    #             x0 = 0
    #         else:
    #             print("xxx", x, self.midline_x)
    #             x0 = x - self.midline_x #(self.ccd_npixels_x / 2 / self.ccd_x_bin)
    #
    #     x0_phys = x0 * self.ccd_x_bin * self.ccd_pixelsize
    #     y0_phys = y0 * self.ccd_y_bin * self.ccd_pixelsize
    #     angle_dx = numpy.arctan(x0_phys / self.grating_camera_distance) * self.camera_magnification
    #     angle_dy = numpy.arctan(y0_phys / self.grating_camera_distance) * self.camera_magnification
    #     wavelength = self.line_spacing / self.grating_order * numpy.cos(angle_dx) * (
    #         numpy.sin(self.alpha) + numpy.sin(self.beta - angle_dy)
    #     )
    #     return wavelength

    # def compute_wl_offset(self, y, dx):
    #     y_2d, x_2d = numpy.indices((self.ccd_npixels_y, self.ccd_npixels_x))
    #     y_2d0 = (y_2d - (self.ccd_npixels_y / 2)) * self.ccd_pixelsize
    #     x_2d0 = (x_2d - (self.ccd_npixels_x / 2)) * self.ccd_pixelsize
    #     dr = numpy.hypot(x_2d0, y_2d0)
    #     angle_offset = numpy.arctan(dr / self.grating_camera_distance) * 2.78
    #     wavelength = self.line_spacing / self.grating_order * (
    #         numpy.sin(self.alpha) + numpy.sin(self.beta - angle_offset)
    #     )
    #     return wavelength

