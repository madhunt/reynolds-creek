#!/usr/bin/python3
import utils, plot_code.plot_utils
import argparse, datetime, os
import shutil
from obspy.io.mseed.util import shift_time_of_file
from obspy.core.utcdatetime import UTCDateTime

def main(path_home, array_str=None,
        time_start=None, time_stop=None,
        shift_list_gem=None, shift_list_time=None):
    '''
    Pre-process and clean up data files before array processing. Plot raw traces.
    INPUTS:
        path_home : str : Path to main directory. Should contain subdirectory data/raw/ containing
            raw mseed data files, and subdirs figures/ and data/mseed/ to save outputs.
        array_str : str : String of array name. Will be used to filter out data files and to save 
            any outputs. E.g. "TOP" or "JD".
        time_start : obspy UTCDateTime : UTCDateTime of first day to process. 
        time_stop : obspy UTCDateTime : Optional. UTCDateTime of final day to process. If None, 
            will just process data for full day of time_start.
        shift_list_gem : list of str : 
        shift_list_time : list of int : 
    RETURNS:
        Plots raw data with 5 traces per figure and saves at:
            path_home/figures/traces/traces_{array_str}_raw_data.png
        Shifts data files by integer seconds and saves files to:
            path_home/data/mseed/
    '''
    # (1) load in raw data
    print("Loading Raw Data")
    path_raw = os.path.join(path_home, "data", "raw")
    data_raw = utils.load_data(path_raw, array_str=array_str, 
                               gem_include=None, gem_exclude=None, 
                               time_start=time_start, time_stop=time_stop,
                               freqmin=None, freqmax=None)

    # (2) plot raw data
    print("Plotting Raw Data")
    plot_utils.plot_traces(data_raw, path_home, f"{array_str} Raw Data")

    # (3) shift traces (if needed) and save data to mseed folder
    print("Saving Pre-Processed Data (mseed)")
    
    for trace in data_raw:
        src_path = trace.stats['path']
        file = os.path.split(src_path)[1]
        dst_path = os.path.join(path_home, "data", "mseed", file)

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
    parser.add_argument("-a", "--array",
            dest="array_str",
            type=str,
            default=None,
            help="Array ID (e.g. TOP or JD)")
    parser.add_argument("-s", "--gem-shift",
            dest="gem_shift",
            type=str,
            action="append",
            nargs='+',
            help="-s TOP11,2 -s TOP44,2 -s TOP23,4")
    
    parser.add_argument("-t1", "--time_start", 
                        dest="time_start",
                        default=None,
                        type=str,
                        help="Time to start data processing in format YYYY-MM-DD")
    parser.add_argument("-t2", "--time_stop", 
                        dest="time_stop",
                        default=None,
                        type=str,
                        help="Time to stop data processing in format YYYY-MM-DD")
    args = parser.parse_args()
    
    if args.time_start != None:
        time_start = args.time_start.split("-")
        time_start = UTCDateTime(int(time_start[0]), int(time_start[1]), int(time_start[2]))
        if args.time_stop == None:
            time_stop = time_start + datetime.timedelta(hours=23, minutes=59, seconds=59)
    else:
        time_start = None
        time_stop = None
    if args.time_stop != None:
        time_stop = args.time_stop.split("-")
        time_stop = UTCDateTime(int(time_stop[0]), int(time_stop[1]), int(time_stop[2]))
        time_stop = time_stop + datetime.timedelta(hours=23, minutes=59, seconds=59)

    #FIXME kinda ugly
    if args.gem_shift != None:
        shift_list_gem = [s.split(",")[0] for row in args.gem_shift for s in row]
        shift_list_time = [float(s.split(",")[1]) for row in args.gem_shift for s in row]
    else:
        shift_list_gem = None
        shift_list_time = None
    
    main(path_home=args.path_home, array_str=args.array_str,
         time_start=time_start, time_stop=time_stop,
         shift_list_gem=shift_list_gem, shift_list_time=shift_list_time)
