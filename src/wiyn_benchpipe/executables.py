#!/usr/bin/env python3

import logging
import argparse
import multiparlog as mplog

from .benchspek import BenchSpek

def wiyn_benchpipe():

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
    args = parser.parse_args()

    benchspec = BenchSpek(args.config, args.raw_dir)
    # print(json.dumps(benchspec.config, indent=2))

    benchspec.calibrate(save=True)
    benchspec.reduce()

