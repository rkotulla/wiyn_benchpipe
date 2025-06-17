
import astropy.io.fits as pyfits

from .sparsepak import  SparsepakFiberSpecs

def select_instrument(file_or_hdu, *args, **kwargs):

    if (type(file_or_hdu) is str):
        try:
            hdulist = pyfits.open(file_or_hdu)
            hdr = hdulist[0].header
        except:
            raise IOError("Could not open file %s" % (file_or_hdu))
    else:
        hdr = file_or_hdu[0].header

    instrument = hdr['INSTRUME']
    if (instrument == 'Bench Spectrograph'):
        fibername = hdr['FIBRNAME']
        if (fibername == 'SparsePak'):
            return SparsepakFiberSpecs(*args, **kwargs)
        else:
            raise ValueError("This fiber type (%s) is not supported" % (fibername))
    else:
        raise ValueError("Cannot identify instrument %s" % (instrument))
