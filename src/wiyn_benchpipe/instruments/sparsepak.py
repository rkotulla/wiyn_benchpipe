import numpy
import astropy.coordinates as coord
import astropy.units as u

from ..fibertraces import GenericFiberSpecs

class SparsepakFiberSpecs( GenericFiberSpecs ):
    n_fibers = 82
    name = "SparsePak @ WIYN"
    ref_fiber_id = 41

    _sky_fibers = [22, 16, 2, 37, 54, 80, 70]
    sky_fiber_ids = numpy.array(_sky_fibers, dtype=int) - 1

    fiberpos = [
        #   1          2          3          4           5          6          7          8         9          0
        (- 4,  0), ( 12, 22), (-14,  9), (-10,  3), (-12,  6), (-12, 12), (-10,  9), (-12,  0), (-14,  3), (-14, -9),  #  1-10
        (-10, -9), (-12,-12), (-14, -3), (-12, -6), (-10, -3), (  0, 22), ( -8, 12), ( -2,  9), ( -4, 12), ( -6,  9),  # 11-20
        ( -4,  6), (-12, 22), ( -6,  3), ( -6, -3), ( -2, -3), ( -2,  1), ( -8,  6), ( -8,  0), ( -4,-12), ( -8, -6),  # 21-30
        (  0, -2), ( -2, -9), ( -4, -6), ( -6, -9), ( -8,-12), (  8, 12), ( 24, 22), (  0, 12), (  4, 12), (  2,  9),  # 31-40
        ( -2,  3), (  0,  6), (  2,  1), (  0,  4), (  4,  0), ( -2, -1), (  0,  2), (  0, -4), (  2, -9), (  4,-12),  # 41-50
        (  2, -3), (  0,  0), (  0,-12), ( 26, 13), ( 10,  9), (  4,  6), ( 12, 12), (  6,  9), ( 14,  9), (  8,  6),  # 51-60
        ( 10, -3), (  2,  3), (  6,  3), (  6, -3), ( 10,  3), (  2, -1), (  6, -9), (  8, -6), (  8,-12), ( 26,-11),  # 61-70
        (  4, -6), ( 12,-12), ( 12,  0), ( 14, -3), ( 12,  6), ( 14,  3), ( 12, -6), ( 14, -9), (  0, -6), ( 26,  1),  # 71-80
        ( 10, -9), (  8,  0),
    ]

    def coordinate_from_fiber_id(self, fiber_id):
        if (fiber_id < 0 or fiber_id > self.number_of_fibers ):
            raise ValueError("Invalid fiber ID (was %d, allowed range %d..%d" % (fiber_id, 0, self.n_fibers))
        x, y = self.fiberpos[fiber_id - 1]

        cx, cy = self._offset_from_xy(x, y)

        return (cx, cy)

    def _offset_from_xy(self, x, y):
        r = 5.71 / 2.
        dx = r * numpy.cos(numpy.radians(30))
        dy = r * numpy.sin(numpy.radians(30))

        cx = x * dx
        cy = y * r
        return cx, cy

    def sky_positions(self, pointing_mode, pointing_reference, ra, dec, rotation=0):
        self.logger.info("Finding pointing positions: mode: %s, reference: %s" % (pointing_mode, str(pointing_reference)))
        if (pointing_mode == 'fiber'):
            ref_x, ref_y = self.fiberpos[pointing_reference - 1]
        elif (pointing_mode == 'pos'):
            ref_x, ref_y = pointing_reference
        else:
            self.logger.warning("Unknown pointing mode %s" % pointing_mode)
            ref_x, ref_y = pointing_reference

        reference_wcs = coord.SkyCoord(ra=ra*u.deg, dec=dec*u.deg)

        fiber_coords = []
        for fiberid in range(self.n_fibers):
            raw_x,raw_y = self.fiberpos[fiberid]
            rel_x = raw_x - ref_x
            rel_y = raw_y - ref_y
            cx,cy = self._offset_from_xy(rel_x, rel_y)

            ra_dec = reference_wcs.spherical_offsets_by(cx*u.arcsec, -cy*u.arcsec)
            fiber_coords.append(ra_dec)

        return fiber_coords

    def reorder_fibers(self, fiberspecs):
        return fiberspecs[::-1, :]

    pass
