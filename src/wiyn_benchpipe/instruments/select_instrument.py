
import astropy
import astropy.io.fits as pyfits

from .wiyn import * #SparsepakFiberSpecs
from .salt import *

from . import *

def select_instrument(file_or_hdu, *args, **kwargs):

    if (type(file_or_hdu) is str):
        try:
            hdulist = pyfits.open(file_or_hdu)
            hdr = hdulist[0].header
        except:
            raise IOError("Could not open file %s" % (file_or_hdu))
    elif (type(file_or_hdu) is astropy.io.fits.header.Header):
        hdr = file_or_hdu
    else:
        hdr = file_or_hdu[0].header

    instrument = hdr['INSTRUME']
    if (instrument == 'Bench Spectrograph'):
        fibername = hdr['FIBRNAME']
        if (fibername == 'SparsePak'):
            return SparsepakFiberSpecs(header=hdr, *args, **kwargs)
        elif (fibername == "Blue"):
            return WiynHydraBlueFiberSpecs(header=hdr, *args, **kwargs)
        else:
            raise ValueError("This fiber type (%s) is not supported" % (fibername))
    elif (instrument == 'NIRWALS'):
        return NirwalsFiberSpecs(header=hdr, *args, **kwargs)
    elif (instrument == 'RSS'):
        return SALT_RSS_IFU_FiberSpecs(header=hdr, *args, **kwargs)
    else:
        raise ValueError("Cannot identify instrument %s" % (instrument))
