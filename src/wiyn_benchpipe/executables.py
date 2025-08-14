#!/usr/bin/env python3

import sys
import logging
import argparse
import multiparlog as mplog
import astropy.io.fits as pyfits
import astropy.units as u
import matplotlib.pyplot as plt
import pandas

from .benchspek import BenchSpek
from .instruments import *

def wiyn_benchpipe(cmdline_args=None):

    if (cmdline_args is None):
        cmdline_args = sys.argv[1:]

    mplog.setup_logging(debug_filename="wiyn_benchpipe_debug.log",
                        log_filename="wiyn_benchpipe_reduce.log")

    # disable nuisance logs from matplotlib (likely related to missing fonts etc and not actually a problem
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)

    #logger = logging.getLogger()
    #logger.setLevel(logging.DEBUG)

    logger = logging.getLogger("WIYN-BenchPipe")
    # logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(filename)-15s [ %(funcName)-30s ] :: %(message)s')


    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config',
                        type=str, default='setup.json')
    parser.add_argument('--rawdir', dest='raw_dir',
                        type=str, default=None)
    parser.add_argument('--debug', dest='debug', action='store_true', default=False)
    args = parser.parse_args(args=cmdline_args)

    if (args.debug):
        logger.warning("ENABLING DEBUG MODE")

    benchspec = BenchSpek(args.config, args.raw_dir, debug=args.debug)
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



def sparsepak_simulator(cmdline_args=None):

    if (cmdline_args is None):
        cmdline_args = sys.argv[1:]

    mplog.setup_logging(debug_filename="wiyn_benchpipe_debug.log",
                        log_filename="wiyn_benchpipe_reduce.log")
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)

    logger = logging.getLogger("SparsePakSimulator")
    parser = argparse.ArgumentParser()
    parser.add_argument('--ra', dest='ra', type=float)
    parser.add_argument('--dec', dest='dec', type=float)
    parser.add_argument('--ref', dest='ref', type=str)
    parser.add_argument('--mode', dest='mode', type=str)
    parser.add_argument('--ds9', dest='ds9', type=str)
    parser.add_argument('--label', dest='label', action='store_true', default=False)
    args = parser.parse_args(args=cmdline_args)

    pointing_mode = args.mode
    pointing_reference = args.ref
    ra = args.ra
    dec = args.dec

    if (pointing_reference is None):
        logger.warning("Missing pointing reference")
        return None
    if (pointing_mode is None):
        logger.warning("No pointing data information found")
        return None
    elif (pointing_mode == 'fiber'):
        pointing_reference = int(args.ref)
    elif (pointing_mode == "pos"):
        pointing_reference = [float(x) for x in args.ref.split(",")]

    rotation = 0
    if (ra is None or dec is None):
        logger.warning("Missing reference coordinates (RA:%s, Dec:%s)" % (str(ra), str(dec)))
        return None

    logger.info("Pointing info == mode:%s ref:%s -- RA:%.5f DEC:%.5f rot=%.1f" % (
        pointing_mode, pointing_reference, ra, dec, rotation
    ))
    raw_traces = SparsepakFiberSpecs()
    fiber_coords = raw_traces.sky_positions(
        pointing_mode, pointing_reference, ra, dec, rotation)

    with open(args.ds9, mode='w') as ds9_file:
        print("""\
# Region file format: DS9 version 4.1
global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
fk5""", file=ds9_file)
        for fiber,coord in enumerate(fiber_coords):
            ra = coord.ra.to(u.degree).value
            dec = coord.dec.to(u.degree).value
            print("""circle(%f,%f,2.8")""" % (ra, dec), file=ds9_file)
            if (args.label):
                print("""# text(%f,%f) text={%d}""" % (ra, dec, fiber+1), file=ds9_file)


    #print(fiber_coords)






def grating_preview(cmdline_args=None):

    if (cmdline_args is None):
        cmdline_args = sys.argv[1:]

    mplog.setup_logging(debug_filename="wiyn_benchpipe_debug.log",
                        log_filename="wiyn_benchpipe_reduce.log")
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)

    logger = logging.getLogger("GratingModel2D")
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs='+',
                        help="list of files with setups to preview")
    parser.add_argument('--linelist', dest='linelist', type=str, default=None)
    args = parser.parse_args(args=cmdline_args)

    for filename in args.files:
        hdulist = pyfits.open(filename)
        instrument = select_instrument(file_or_hdu=hdulist)
        logger.info("Instrument: %s" % (instrument.name))

        grating = instrument.grating_from_header(hdulist[0].header)
        logger.info("Grating: %s" % (grating.name))
        logger.info("Wavelength range: %.2f -- %2f" % (grating.wl_blueedge, grating.wl_rededge))
        logger.info("Dispersion: %f A/px" % (numpy.fabs(grating.wl_polyfit[1])))

        y = numpy.linspace(0, grating.ccd_npixels_y / grating.ccd_y_bin, 100)
        wl = grating.wavelength_from_xy(y=y)

        fig, ax = plt.subplots()
        ax.plot(y, wl)

        try:
            if (args.linelist is not None):
                logger.info("Reading line list from %s" % (args.linelist))
                line_df = pandas.read_csv(args.linelist, comment='#')
                line_wl = line_df['wavelength']
                line_y = grating.y_from_wavelength(line_wl, return_y0=False)
                in_range = (line_wl >= numpy.min(wl)) & (line_wl < numpy.max(wl))
                ax.scatter(line_y[in_range], line_wl[in_range], label='Lines')
                for y,w in zip(line_y[in_range], line_wl[in_range]):
                    print(w,y)
                    _ = ax.text(x=y,y=w,s=" %.1f"%w, rotation=45, fontsize='x-small', ha='left', va='baseline')
                    print(_)

                ax.legend()
        except Exception as e:
            mplog.log_exception()

        plot_fn = filename[:-5] + "_gratingmodel.png"
        fig.tight_layout()
        fig.savefig(plot_fn, dpi=200)

        logger.info("Results saved to %s\n" % (plot_fn))




def grating_model_2d(cmdline_args=None):

    if (cmdline_args is None):
        cmdline_args = sys.argv[1:]

    mplog.setup_logging(debug_filename="wiyn_benchpipe_debug.log",
                        log_filename="wiyn_benchpipe_reduce.log")
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)

    logger = logging.getLogger("GratingModel2D")
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs='+',
                        help="list of files with setups to preview")
    parser.add_argument('--linelist', dest='linelist', type=str, default=None)
    args = parser.parse_args(args=cmdline_args)

    for filename in args.files:
        hdulist = pyfits.open(filename)
        instrument = select_instrument(file_or_hdu=hdulist)
        logger.info("Instrument: %s" % (instrument.name))

        grating = instrument.grating_from_header(hdulist[0].header)
        logger.info("Grating: %s" % (grating.name))

        y = numpy.linspace(0, grating.ccd_npixels_y / grating.ccd_y_bin, 100)
        x = numpy.linspace(0, grating.ccd_npixels_x / grating.ccd_x_bin, 100)

        reg_filename = filename[:-5] + "_wlmodel2d.reg"
        reg = open(reg_filename, "w")
        print("""\
# Region file format: DS9 version 4.1
global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
image""", file=reg)

        blue_edge = grating.wl_blueedge
        red_edge = grating.wl_rededge
        logger.info("Wavelength range: %.2f (blue) to %.2f (red)" % (blue_edge, red_edge))

        x0 = x - 968 #grating.ccd_npixels_x / grating.ccd_x_bin / 2
        for i_wl, wl in enumerate(numpy.linspace(blue_edge, red_edge, 20)):
            y0 = grating.y_from_wavelength(wl, x0=x0, return_y0=True)
            y = y0 + 1024 #1650
            print("# composite(%f,%f,0) || composite=%d" % (x[0], y[0], i_wl+1), file=reg)
            for i in range(y.shape[0]-1):
                print("line(%f, %f, %f, %f) || " % (x[i], y[i], x[i+1], y[i+1]), file=reg)
            print("text(%f,%f) # text={%.1f}" % (x[-1], y[-1], wl), file=reg)


        # wl = grating.wavelength_from_xy(y=y)
        #
        # fig, ax = plt.subplots()
        # ax.plot(y, wl)
        #
        # try:
        #     if (args.linelist is not None):
        #         logger.info("Reading line list from %s" % (args.linelist))
        #         line_df = pandas.read_csv(args.linelist, comment='#')
        #         line_wl = line_df['wavelength']
        #         line_y = grating.y_from_wavelength(line_wl, return_y0=False)
        #         in_range = (line_wl >= numpy.min(wl)) & (line_wl < numpy.max(wl))
        #         ax.scatter(line_y[in_range], line_wl[in_range], label='Lines')
        #         for y,w in zip(line_y[in_range], line_wl[in_range]):
        #             print(w,y)
        #             _ = ax.text(x=y,y=w,s=" %.1f"%w, rotation=45, fontsize='x-small', ha='left', va='baseline')
        #             print(_)
        #
        #         ax.legend()
        # except Exception as e:
        #     mplog.log_exception()
        #
        # plot_fn = filename[:-5] + "_gratingmodel.png"
        # fig.tight_layout()
        # fig.savefig(plot_fn, dpi=200)
        #
        # logger.info("Results saved to %s" % (plot_fn))


def centers_to_ds9_region_with_labels(centers, ds9_filename, reverse, color='green', good_trace=None, every=10):

    y = numpy.arange(centers.shape[0])
    print(centers.shape)

    ds9 = open(ds9_filename, "w")
    print("""\
    # Region file format: DS9 version 4.1
    global color=%s dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
    physical""" % (color), file=ds9)

    for fiberid in range(centers.shape[1]):
        # print(fiberid)
        fiberlabel = fiberid + 1 if not reverse else centers.shape[1]-fiberid

        xy = numpy.array([centers[:,fiberid], y])[:, ::every].T
        #xy = xy[:, ::every]
        #print(xy.shape)

        #print(xy.shape)

        segments = ['%.2f,%d' % (x[0]+1,x[1]+1) for x in xy]
        #print(segments)

        if (good_trace is not None and not good_trace[fiberid]):
            # extras = ["" for _real in good_trace if _real else " # background"]
            start_end = numpy.array(["line(%s  ,  %s) # background " % (s,e) for (s,e) in zip(segments[:-1], segments[1:])])
        else:
            start_end = numpy.array(["line(%s  ,  %s)" % (s,e) for (s,e) in zip(segments[:-1], segments[1:])])
        #print(start_end)

        # line_points = xy.T.ravel()
        # line_str = ",".join(["%.2f" % f for f in line_points[:30]])
        # print(line_points[:30])
        # print(line_str)
        # print("line(%s)" % line_str, file=ds9)

        n_labels = 10
        label_every = xy.shape[0] / n_labels
        i_label = ((numpy.arange(n_labels) + 0.5) * label_every).astype(int)
        # print(i_label)

        # index all lines
        i_line = numpy.arange(xy.shape[0]-1)
        #print("i-line:", i_line.shape)

        # figure out where to show lines or labels
        show_label = numpy.zeros_like(i_line, dtype=bool)
        #print(i_label[:50])
        #print(show_label[:50])
        for i in i_label:
            show_label[i_line == i] = True
        #print(show_label[:50])
        show_line = ~show_label #i_line != i_label

        #print(show_label.shape, show_line.shape, start_end.shape)
        #print(show_line[:10])
        #print(start_end[show_line])
        all_lines = "\n".join(start_end[show_line])
        print(all_lines, file=ds9)

        label_pos = 0.5*(xy[i_label,:] + xy[i_label+1,:])
        #print(label_pos)
        #for _i_label in i_label[show_label]:
        for (_x,_y) in label_pos:
            print("text(%.2f,%.2f) # text={%d}" % (_x+1,_y+1, fiberlabel), file=ds9)
        #break
        #line_points =

    ds9.close()
def traces_to_ds9(cmdline_args=None):

    if (cmdline_args is None):
        cmdline_args = sys.argv[1:]

    mplog.setup_logging(debug_filename="wiyn_benchpipe_debug.log",
                        log_filename="wiyn_benchpipe_reduce.log")
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)

    logger = logging.getLogger("Traces-to-ds9")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest='input', type=str, default="fiber_tracers_fullres.fits")
    parser.add_argument("--reverse", dest='reverse', action='store_true', default=False)
    parser.add_argument("--type", dest='tracetype', default='center', type=str)
    parser.add_argument("--output", dest='output', default='trace.reg', type=str)
    parser.add_argument("--color", dest='color', default='green', type=str)
    args = parser.parse_args(args=cmdline_args)

    logger.info("Reading trace data from %s ..." % (args.input))
    hdulist = pyfits.open(args.input)
    # hdulist.info()

    valid_trace_types = ["CENTER", "LEFT", "RIGHT", "PEAKS"]
    if (args.tracetype.upper() not in valid_trace_types):
        logger.error("Illegal trace type (%s), must be one of %s" % (args.tracetype, ", ".join(valid_trace_types)))
        return -1

    data = hdulist[args.tracetype.upper()].data
    logger.info("Writing ds9 regions to %s" % (args.output))
    centers_to_ds9_region_with_labels(
        centers=data, ds9_filename=args.output, reverse=args.reverse,
        color=args.color)

    return 0
