#!/usr/bin/python3

import utils, plot_utils
import argparse, glob, obspy, os
import shutil
from obspy.io.mseed.util import shift_time_of_file


def main(path_home, #filter_options=None, 
         shift_list_gem=None, shift_list_time=None):
         #date_start=None, date_end=None):

    # (1) load in raw data
    print("Loading Raw Data")
    path_raw = os.path.join(path_home, "data", "raw")
    data_raw = utils.load_data(path_raw, gem_include=None, gem_exclude=None, 
                    filter_type=None)#, filter_options=None)
                    #TODO add time_start and time_stop

    # (2) plot raw data
    print("Plotting Raw Data")
    plot_utils.plot_traces(data_raw, path_home, "Raw Data")

    # (3) shift traces (if needed) and save data to mseed folder
    print("Saving mseed Data")
    files_raw = glob.glob(os.path.join(path_raw, "*.mseed" ))
    path_mseed = os.path.join(path_home, "data", "mseed")

    for src_path in files_raw:
        file = os.path.split(src_path)[1]
        dst_path = os.path.join(path_mseed, file)

        if (shift_list_gem != None) and any(station in file for station in shift_list_gem):
            # shift needed stations and save
            i = [i for i, s in enumerate(shift_list_gem) if s in file][0]
            print(f"Shifting Station {shift_list_gem[i]}")
            shift_time_of_file(input_file=src_path, 
                                output_file=dst_path, 
                                timeshift=shift_list_time[i])
        else:
            # just copy raw data
            shutil.copy(src=src_path, dst=dst_path)

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
    

    #TODO SHOULD t2 BE EXCLUSIVE? 
    # specify just 2023-10-07 with t1=2023-10-07 and t2=2023-10-08
    # OR with t1=2023-10-07 and t2=2023-10-07
    #parser.add_argument("-t1", "--time-start",
    #        dest="date_start",
    #        type=str,
    #        help=("Desired start date for data to be processed, in the form YYYY-MM-DD. \
    #              If only one day should be processed, t1=t2."))
    #parser.add_argument("-t2", "--time-end",
    #        dest="date_end",
    #        type=str,
    #        help=("Desired end date for data to be processed, in the form YYYY-MM-DD. \
    #              If only one day should be processed, t1=t2."))

    args = parser.parse_args()
    

    #FIXME kinda ugly
    if args.gem_shift != None:
        shift_list_gem = [s.split(",")[0] for row in args.gem_shift for s in row]
        shift_list_time = [float(s.split(",")[1]) for row in args.gem_shift for s in row]
    else:
        shift_list_gem = None
        shift_list_time = None

    if args.freqs != None:
        main(path_home=args.path_home, 
            filter_options=dict(freqmin=args.freqs[0], 
                                freqmax=args.freqs[1]),
            #date_start=args.date_start, date_end=args.date_end,
            shift_list_gem=shift_list_gem, shift_list_time=shift_list_time)
    else:
        main(path_home=args.path_home, 
            #filter_options=None,
            #date_start=args.date_start, date_end=args.date_end,
            shift_list_gem=shift_list_gem, shift_list_time=shift_list_time)

