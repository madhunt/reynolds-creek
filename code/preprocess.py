#!/usr/bin/python3

import utils, plot_utils
import argparse, glob, obspy, os
import shutil
from obspy.io.mseed.util import shift_time_of_file


def main(path_home, filter_options=None, gem_list=None, shift_list=None):

    # (1) load in raw data
    print("Loading Raw Data")
    path_raw = os.path.join(path_home, "data", "raw")
    data_raw = utils.load_data(path_raw, gem_include=None, gem_exclude=None, 
                     filter_type=None, **filter_options)

    # (2) plot raw data
    print("Plotting Raw Data")
    plot_utils.plot_traces(data_raw, path_home, "Raw Data")

    # (3) shift traces (if needed)
    files_raw = glob.glob(os.path.join(path_raw, "*.mseed" ))
    path_mseed = os.path.join(path_home, "data", "mseed")

    if gem_list != None:
        # loop through all data
        for src_path in files_raw:
            file = os.path.split(src_path)[1]
            dst_path = os.path.join(path_mseed, file)

            # if station needs to be shifted
            if any(station in file for station in gem_list):
                i = [i for i, s in enumerate(gem_list) if s in file][0]
                print(f"Shifting Station {gem_list[i]}")
                shift_time_of_file(input_file=src_path, 
                                   output_file=dst_path, 
                                   timeshift=shift_list[i])
            #TODO MOVE OUT OF IF STATEMENT???
            else:
                # move over the raw data
                shutil.copy(src=src_path, dst=dst_path)

    # (4) replot traces
    #filt_freq_str = f"{filter_options['freqmin']}_{filter_options['freqmax']}"
    return


if __name__ == "__main__":
    #NOTE: for VS Code, these are in launch.json file as "args"
    parser = argparse.ArgumentParser(
        description="Plot and pre-process data from data/raw. Save to data/mseed.")
    
    # path to this file
    path_curr = os.path.dirname(os.path.realpath(__file__))
    path_home = os.path.abspath(os.path.join(path_curr, '..'))
    
    parser.add_argument("-d", "--dir",
            dest="path_home",
            type=str,
            default=path_home,
            help="Path to top-level directory.")
    parser.add_argument("-f", "--freqs",
            nargs=2,
            dest="freqs",
            metavar=("freqmin", "freqmax"),
            type=float,
            help="Min and max frequencies for bandpass filter.")
    parser.add_argument("-s", "--gem-shift",
            dest="gem_shift",
            type=str,
            action="append",
            nargs='+',
            help="-s TOP11,2 -s TOP44,2 -s TOP23,4")

    args = parser.parse_args()
    

    #FIXME kinda ugly
    if args.gem_shift != None:
        gem_list = [s.split(",")[0] for row in args.gem_shift for s in row]
        shift_list = [float(s.split(",")[1]) for row in args.gem_shift for s in row]
    else:
        gem_list = None
        shift_list = None

    main(path_home=args.path_home, 
         filter_options=dict(freqmin=args.freqs[0], 
                             freqmax=args.freqs[1]),
        gem_list=gem_list, shift_list=shift_list)
