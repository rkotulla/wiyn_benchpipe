#!/usr/bin/env python3

import sys
import logging
import argparse
import multiparlog as mplog
import astropy.io.fits as pyfits

from .benchspek import BenchSpek

def wiyn_benchpipe(cmdline_args=None):

    if (cmdline_args is None):
        cmdline_args = sys.argv[1:]

    mplog.setup_logging(debug_filename="nirwals_debug.log",
                        log_filename="nirwals_reduce.log")
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)

    logger = logging.getLogger("WIYN-BenchPipe")
    # logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(filename)-15s [ %(funcName)-30s ] :: %(message)s')


    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config',
                        type=str, default='setup.json')
    parser.add_argument('--rawdir', dest='raw_dir',
                        type=str, default='raw/')
    args = parser.parse_args(args=cmdline_args)

    benchspec = BenchSpek(args.config, args.raw_dir)
    # print(json.dumps(benchspec.config, indent=2))

    benchspec.calibrate(save=True)
    benchspec.reduce()




def region_file_from_output(cmdline_args=None):

    if (cmdline_args is None):
        cmdline_args = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('fits_file')
    parser.add_argument('region_file')
    args = parser.parse_args(args=cmdline_args)

    hdulist = pyfits.open(args.fits_file)
    header = hdulist[0].header

    with open(args.region_file, 'w') as reg:
        print("""\
# Region file format: DS9 version 4.1
global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
fk5""", file=reg)
        for i in range(1000):
            try:
                ra = header['F%03d_RA' % (i+1)]
                dec = header['F%03d_DEC' % (i+1)]
                print("""circle(%f,%f,2.8")""" % (ra,dec), file=reg)
                print(ra,dec)
            except Exception as e:
                print("error",e)

                break