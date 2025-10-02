#!/usr/bin/env python3

import sys
import os
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




def nirwals_flux_calibration_scaling(cmdline_args=None):

    if (cmdline_args is None):
        cmdline_args = sys.argv[1:]

    mplog.setup_logging(debug_filename="wiyn_benchpipe_debug.log",
                        log_filename="wiyn_benchpipe_reduce.log")
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)

    logger = logging.getLogger("Traces-to-ds9")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest='input', type=str)
    parser.add_argument("--reference", dest='reference', type=str)
    parser.add_argument("--step", dest='step', default=20, type=float)
    parser.add_argument("--window", dest='window', default=75, type=float)
    parser.add_argument("--fibers", dest='fibers', default=None, type=str)
    parser.add_argument("--fiberfrac", dest='fiber_fraction', default=0.2, type=float)
    parser.add_argument("--plot", dest='plot', default=None, type=str)
    parser.add_argument("--csv", dest='csv', default=None, type=str)
    args = parser.parse_args(args=cmdline_args)

    # open reference spectrum
    ref_hdu = pyfits.open(args.reference)
    ref_hdu.info()
    ref = ref_hdu[0].data
    hdr = ref_hdu[0].header
    ref_wl = ref[0] * 1e4 # convert from micron to A
    ref_flux = ref[1]

    # open input file
    sci_hdu = pyfits.open(args.input)
    sci_hdr = sci_hdu[0].header
    sci = sci_hdu[0].data
    # print(repr(sci_hdr))
    x = numpy.arange(sci.shape[1])
    sci_wl = x * sci_hdr['CD1_1'] + sci_hdr['CRVAL1']

    # find fibers with most flux
    total_fiber_flux =  numpy.nanmean(sci, axis=1)

    if (args.fibers is None):
        peak_flux = numpy.max(total_fiber_flux)
        select = total_fiber_flux > 0.1*peak_flux
    else:
        select_fibers = [int(f)-1 for f in args.fibers.split(",")]
        select = numpy.zeros((sci.shape[0]))
        for f in select_fibers:
            select[f] = True
    print("using %d fibers" % (numpy.sum(select)))

    # add up all selected fibers
    integrated_spectrum = numpy.sum(sci[select], axis=0)

    # add up all flux in windows, and find scaling ratio
    windowsize = args.window # Angstroems
    steps = args.step
    calib_df = pandas.DataFrame()
    for i, central_wl in enumerate(numpy.arange(sci_wl[0], sci_wl[-1], steps)):
        min_wl = numpy.max([sci_wl[0], central_wl-windowsize])
        max_wl = numpy.min([sci_wl[-1], central_wl+windowsize])

        in_sci_window = (sci_wl > min_wl) & (sci_wl < max_wl)
        in_ref_window = (ref_wl > min_wl) & (ref_wl < max_wl)

        calib_df.loc[i, 'min_wl'] = min_wl
        calib_df.loc[i, 'max_wl'] = max_wl
        calib_df.loc[i, 'central_wl'] = central_wl

        calib_df.loc[i, 'sci_flux'] = numpy.nanmean(integrated_spectrum[in_sci_window])
        calib_df.loc[i, 'ref_flux'] = numpy.nanmean(ref_flux[in_ref_window])
    calib_df['sci2ref'] = calib_df['ref_flux'] / calib_df['sci_flux']

    # now make a plot
    fig, axs = plt.subplots(nrows=5, ncols=1, figsize=(12,16))
    axs[0].set_title("Reference spectrum")
    axs[0].plot(ref_wl, ref_flux, lw=0.5)
    axs[0].set_xlabel(r"Wavelength [$\AA$]")
    axs[0].set_ylabel(r"calibrated lux")

    axs[1].set_title("NIRWALS spectrum")
    axs[1].plot(sci_wl, integrated_spectrum, lw=1, label='spectrum')
    axs[1].plot(calib_df['central_wl'], calib_df['sci_flux'], label='smoothed', alpha=0.5, lw=3)
    # axs[1].scatter(calib_df['central_wl'], calib_df['sci_flux'], label='smoothed')
    axs[1].set_xlim((sci_wl[0], sci_wl[-1]))
    axs[1].set_xlabel(r"Wavelength [$\AA$]")
    axs[1].set_ylabel(r"observed flux [counts/second]")
    axs[1].legend()

    axs[2].set_title("Reference spectrum, overlapping wavelength range")
    axs[2].plot(ref_wl, ref_flux, lw=1, label='spectrum')
    axs[2].plot(calib_df['central_wl'], calib_df['ref_flux'], label='smoothed', alpha=0.5, lw=3)
    axs[2].set_xlim((sci_wl[0], sci_wl[-1]))
    axs[2].set_xlabel(r"Wavelength [$\AA$]")
    axs[2].set_ylabel(r"Flux")
    axs[2].legend()

    axs[3].set_title("flux calibration factor")
    axs[3].plot(calib_df['central_wl'], calib_df['ref_flux']/calib_df['sci_flux'])
    axs[3].set_xlim((sci_wl[0], sci_wl[-1]))
    axs[3].set_xlabel(r"Wavelength [$\AA$]")
    axs[3].set_ylabel(r"scaling factor [reference / science]")

    full_scale = numpy.interp(sci_wl, calib_df['central_wl'], calib_df['sci2ref'])
    axs[4].plot(sci_wl, integrated_spectrum*full_scale, lw=1, label="NIRWALS, calibrated")
    axs[4].plot(ref_wl, ref_flux, lw=1, label="reference")
    axs[4].set_xlim((sci_wl[0], sci_wl[-1]))
    axs[4].legend()
    axs[4].set_xlabel(r"Wavelength [$\AA$]")
    axs[4].set_ylabel(r"calibrated flux")

    fig.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.05, wspace=0.4, hspace=0.4)

    bn, _ = os.path.splitext(args.input)
    plot_fn = args.plot
    if (plot_fn is None):
        plot_fn = "%s__fluxcal.pdf" % (bn)
    fig.savefig(args.plot)

    csv_fn = args.csv
    if (csv_fn is None):
        csv_fn = "%s__fluxcal.csv" % (bn)
    calib_df.to_csv(csv_fn, index=False)