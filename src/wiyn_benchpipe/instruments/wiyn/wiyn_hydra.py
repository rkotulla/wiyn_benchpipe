import numpy
import astropy.coordinates as coord
import astropy.units as u

from ...fibertraces import GenericFiberSpecs
from .wiyn_bench import wiyn_grating_from_header, WIYNBenchFiberSpecs

class WiynHydraRedFiberSpecs( WIYNBenchFiberSpecs ):
    n_fibers = 83
    name = "Hydra Red Cable @ WIYN"
    ref_fiber_id = 41

class WiynHydraBlueFiberSpecs( WIYNBenchFiberSpecs ):
    n_fibers = 87
    name = "Hydra Blue Cable @ WIYN"
    ref_fiber_id = 45

    trace_minx = 160
    trace_maxx = 2520

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if (self.header is not None):
            _n_fibers = 0
            for i in range(1000):
                slfib_key = "SLFIB%d" % (i+1)
                if (slfib_key not in self.header):
                    break
                slfib = self.header[slfib_key]
                if (slfib.strip() == ""):
                    break
                if (slfib.lower().find("broken fiber") > 0):
                    self.logger.debug("Marking fiber %d as broken", i+1)
                    continue
                items = slfib.split()
                fiber_type = int(items[1])
                # if (fiber_type >= 0):
                _n_fibers += 1
            #self.n_fibers = _n_fibers
            self.logger.info("Updating #fibers from header: %d" % self.n_fibers)