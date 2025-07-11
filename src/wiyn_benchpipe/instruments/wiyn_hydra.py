import numpy
import astropy.coordinates as coord
import astropy.units as u

from ..fibertraces import GenericFiberSpecs
from .wiyn_bench import wiyn_grating_from_header, WIYNBenchFiberSpecs

class WiynHydraRedFiberSpecs( WIYNBenchFiberSpecs ):
    n_fibers = 83
    name = "Hydra Red Cable @ WIYN"
    ref_fiber_id = 41

class WiynHydraBlueFiberSpecs( WIYNBenchFiberSpecs ):
    n_fibers = 90
    name = "Hydra Blue Cable @ WIYN"
    ref_fiber_id = 45

