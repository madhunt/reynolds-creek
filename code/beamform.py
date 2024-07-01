#!/usr/bin/python3

import plot_utils, utils
import argparse, math, os, datetime
import numpy as np
import pandas as pd
#from obspy.core.util import AttribDict
from obspy.core.utcdatetime import UTCDateTime
from obspy.signal.array_analysis import array_processing
from concurrent.futures import ProcessPoolExecutor
from matplotlib.dates import num2date

def main(path_home, process=False, 
         array_str=None,
         gem_include=None, gem_exclude=None,
         time_start=None, time_stop=None, 
         freqmin=None, freqmax=None):

    path_data = os.path.join(path_home, "data", "mseed")
    filt_freq_str = f"{freqmin}_{freqmax}"
    filt_date_str = f"{time_start.year}-{time_start.month}-{time_start.day}"
    file_str = f"{array_str}_{filt_date_str}_{filt_freq_str}"

    path_processed = os.path.join(path_home, "data", "processed", 
                    f"processed_output_{file_str}.pkl")

    # print progress to log file
    with open(os.path.join(path_home, "code", "log", "pylog.txt"), "a") as f:
        print((f"{datetime.datetime.now()} \t\t Loading and Filtering Data ")+
              (f"({file_str})"), file=f)
        print("    "+path_data, file=f)

    # load data
    data = utils.load_data(path_data, array_str=array_str,
                           gem_include=gem_include, gem_exclude=gem_exclude, 
                           time_start=time_start, time_stop=time_stop,
                           freqmin=freqmin, freqmax=freqmax)
    
    if process == True:
        # print progress to log file
        with open(os.path.join(path_home, "code", "log", "pylog.txt"), "a") as f:
            print((f"{datetime.datetime.now()} \t\t Processing Data ")+
                  (f"({file_str})"), file=f)
            print("    "+path_processed, file=f)

        # fiter and beamform 
        output = process_data(data, path_processed, 
                              time_start=time_start, time_stop=time_stop, 
                              freqmin=freqmin, freqmax=freqmax)

    else:
        # data has already been processed
        output = pd.read_pickle(path_processed)
    
    # print progress to log file
    with open(os.path.join(path_home, "code", "log", "pylog.txt"), "a") as f:
        print((f"{datetime.datetime.now()} \t\t Plotting Backazimuth ")+ 
                (f"({file_str})"), file=f)
        print("    "+os.path.join(path_home, "figures", f"backaz_{file_str}.png"), file=f)

    # plot backaz time series
    plot_utils.plot_backaz(output, path_home, 
                            f"{array_str} Array, Filtered {freqmin} to {freqmax} Hz", file_str)
    
    return


def process_data(data, path_processed, time_start=None, time_stop=None, freqmin=None, freqmax=None):
    '''
    Run obspy array_processing() function to beamform data. Save in .npy format to specified 
    location. Returns output as np array, with backazimuths from 0-360.
    INPUTS
        data : obspy stream : merged data files
        path_processed : str : path and filename to save output as .npy
        time_start : obspy UTCDateTime : if specified, time to start beamforming. If not specified, 
            will use max start time from all Gems.
        time_stop : obspy UTCDateTime : if specified, time to end beamforming. If not specified, 
            will use min end time from all Gems.
    RETURNS
        output : pd dataframe : array with 5 rows of output from array_processing. Rows are: timestamp, 
            relative power (semblance), absolute power, backazimuth (from 0-360), and slowness (in s/km).
    '''
    #FIXME doc string above

    # if times are not provided, use max/min start and end times from gems
    if time_start == None:
        time_start = max([trace.stats.starttime for trace in data])
    if time_stop == None:
        time_stop = min([trace.stats.endtime for trace in data])

    
    #FIXME can clean this up when these change
    process_kwargs = dict(
        # slowness grid (in [s/km])
        sll_x=-4.0, slm_x=4.0, sll_y=-4.0, slm_y=4.0, sl_s=0.1,
        # sliding window
        win_len=60, win_frac=0.50,
        # frequency
        frqlow=freqmin, frqhigh=freqmax, prewhiten=0,
        # output restrictions
        semb_thres=-1e9, vel_thres=-1e9, timestamp='mlabday',
        stime=time_start, etime=time_stop)
    
    output = array_processing(stream=data, **process_kwargs)

    # correct backaz from 0 to 360 (instead of -180 to +180)
    output[:,3] = [output[i][3] if output[i][3]>=0 else output[i][3]+360 
                    for i in range(output.shape[0])]

    # save output as dataframe
    output = pd.DataFrame(data=output, 
                          columns=["Time", "Semblance", "Abs Power", "Backaz", "Slowness"])
    
    # save time steps as datetime types
    output["Time"] = num2date(output["Time"])

    # save output to pickle
    output.to_pickle(path_processed)
    return output

if __name__ == "__main__":
    # parse arguments from command line
    #NOTE: for VS Code, these are in launch.json file as "args"
    parser = argparse.ArgumentParser(
        description="Run traditional shift and stack beamforming on specified mseed data.")
    
    # path to this file
    path_curr = os.path.dirname(os.path.realpath(__file__))
    path_home = os.path.abspath(os.path.join(path_curr, '..'))
    
    #TODO allow user to specify input and output dirs if they dont have the correct structure
    parser.add_argument("-d", "--dir",
            dest="path_home",
            type=str,
            default=path_home,
            help="Path to top-level directory containing data and figure sub-directories.")

    parser.add_argument("-p", "--process",
                        dest="process",
                        default=False,
                        action="store_true",
                        help="Flag if data should be processed.")
    parser.add_argument("-a", "--array",
            dest="array_str",
            type=str,
            default=None,
            help="Array ID (e.g. TOP or JD)")

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

    freqs = parser.add_mutually_exclusive_group(required=True)
    freqs.add_argument("-f", "--freq",
            nargs=2,
            dest="freq_bp",
            metavar=("freqmin", "freqmax"),
            type=float,
            help="Min and max frequencies for a single bandpass filter.")
    freqs.add_argument("-F", "--freq-range",
            nargs=2,
            dest="freq_range",
            metavar=("freqmin", "freqmax"),
            type=float,
            help="Min and max for a range of frequencies to run multiple bandpass filters in parallel.")

    group_gems = parser.add_mutually_exclusive_group(required=False)
    group_gems.add_argument("-i", "--gem-include",
            dest="gem_include",
            type=str,
            help="Gems to include in processing in comma-separated list (no spaces).")
    group_gems.add_argument("-x", "--gem-exclude",
            dest="gem_exclude",
            type=str,
            help="Gems to exclude in processing in comma-separated list (no spaces).")

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


    with open(os.path.join(path_home, "code", "log", "pylog.txt"), "a") as f:
        print("-----------------------------------NEW RUN-----------------------------------", file=f)

    if args.freq_bp != None:
        main(path_home=args.path_home, process=args.process, 
            array_str=args.array_str,
            gem_include=utils.arg_split_comma(args.gem_include),
            gem_exclude=utils.arg_split_comma(args.gem_exclude),
            time_start=time_start, time_stop=time_stop,
            freqmin=args.freq_bp[0], freqmax=args.freq_bp[1])

    elif args.freq_range != None:
        # multiple frequencies specified
        #TODO what if you don't want octaves??
        pwr_min = math.log(args.freq_range[0], 2)
        pwr_max = math.log(args.freq_range[1], 2)
        freq_list = 2** np.arange(pwr_min, pwr_max, 1)

        #NOTE for Borah (48 cores per node)
        with ProcessPoolExecutor(max_workers=48) as pool:
            args_list = [[args.path_home, args.process, 
                          args.array_str,
                          utils.arg_split_comma(args.gem_include),
                          utils.arg_split_comma(args.gem_exclude),
                          time_start, time_stop,
                          freqmin, freqmax] 
                        for freqmin in freq_list for freqmax in freq_list if freqmin < freqmax]

            # run through different filters in parallel
            result = pool.map(main, *zip(*args_list))

    print("Completed Processing")
    with open(os.path.join(path_home, "code", "log", "pylog.txt"), "a") as f:
        print(f"{datetime.datetime.now()} \t\t Completed Processing", file=f)
