#!/usr/bin/python3

import plot_utils, utils
import argparse, math, os, datetime
import numpy as np
import pandas as pd
from obspy.core.util import AttribDict
from obspy.signal.array_analysis import array_processing
from concurrent.futures import ProcessPoolExecutor

def main(path_home, process=False, backaz_plot=False,
         filter_options=None, gem_include=None, gem_exclude=None):

    path_data = os.path.join(path_home, "data", "mseed")
    filt_freq_str = f"{filter_options['freqmin']}_{filter_options['freqmax']}"
    path_processed = os.path.join(path_home, "data", "processed", 
                    f"processed_output_{filt_freq_str}.pkl")

    # print progress to log file
    with open(os.path.join(path_home, "code", "log", "pylog.txt"), "a") as f:
        print(f"{datetime.datetime.now()} \t\t Loading and Filtering Data ({filt_freq_str})", file=f)
        print(path_data, file=f)

    # load data
    #TODO option for highpass or lowpass only
    data = utils.load_data(path_data, gem_include=gem_include, gem_exclude=gem_exclude, 
                     filter_type='bandpass', **filter_options)
    
    if process == True:
        # print progress to log file
        with open(os.path.join(path_home, "code", "log", "pylog.txt"), "a") as f:
            print(f"{datetime.datetime.now()} \t\t Processing Data ({filt_freq_str})", file=f)
            print(path_processed, file=f)

        # fiter and beamform 
        output = process_data(data, path_processed, time_start=None, time_end=None, filter_options=filter_options)

    else:
        # data has already been processed
        output = pd.read_pickle(path_processed)
    
    if backaz_plot == True:
        # print progress to log file
        with open(os.path.join(path_home, "code", "log", "pylog.txt"), "a") as f:
            print(f"{datetime.datetime.now()} \t\t Plotting Backazimuth ({filt_freq_str})", file=f)

        # plot backaz/slowness time series
        #FIXME
        #FIXME
        #FIXME
        # uncomment
        ##plot_utils.plot_backaz_slowness(output, path_home, filt_freq_str)

        # plot slowness space
        plot_utils.plot_slowness(output, path_home, id_str=filt_freq_str)
    
    return


def process_data(data, path_processed, time_start=None, time_end=None, filter_options=None):
    '''
    Run obspy array_processing() function to beamform data. Save in .npy format to specified 
    location. Returns output as np array, with backazimuths from 0-360.
    INPUTS
        data : obspy stream : merged data files
        path_processed : str : path and filename to save output as .npy
        time_start : obspy UTCDateTime : if specified, time to start beamforming. If not specified, 
            will use max start time from all Gems.
        time_end : obspy UTCDateTime : if specified, time to end beamforming. If not specified, 
            will use min end time from all Gems.
    RETURNS
        output : pd dataframe : array with 5 rows of output from array_processing. Rows are: timestamp, 
            relative power (semblance), absolute power, backazimuth (from 0-360), and slowness (in s/km).
    '''
    #FIXME doc string above

    # if times are not provided, use max/min start and end times from gems
    if time_start == None:
        # specify start time
        time_start = max([trace.stats.starttime for trace in data])
    if time_end == None:
        time_end = min([trace.stats.endtime for trace in data])

    
    #FIXME can clean this up when these change
    process_kwargs = dict(
        # slowness grid (in [s/km])
        sll_x=-4.0, slm_x=4.0, sll_y=-4.0, slm_y=4.0, sl_s=0.1,
        # sliding window
        win_len=60, win_frac=0.50,
        # frequency
        frqlow=filter_options['freqmin'], frqhigh=filter_options['freqmax'], prewhiten=0,
        # output restrictions
        semb_thres=-1e9, vel_thres=-1e9, timestamp='mlabday',
        stime=time_start, etime=time_end)
    
    output = array_processing(stream=data, **process_kwargs)

    # correct backaz from 0 to 360 (instead of -180 to +180)
    output[:,3] = [output[i][3] if output[i][3]>=0 else output[i][3]+360 
                    for i in range(output.shape[0])]

    # save output as dataframe
    output = pd.DataFrame(data=output, 
                          columns=["Time", "Semblance", "Abs Power", "Backaz", "Slowness"])

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
    #FIXME dont really need this as an arg bc i will always want to plot
    # maybe?
    parser.add_argument("-b", "--plot-backaz",
                        dest="backaz_plot",
                        default=False,
                        action="store_true",
                        help="Flag if backazimuth plots should be created.")
    
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
    

    if args.freq_bp != None:
        main(path_home=args.path_home, 
            process=args.process, 
            backaz_plot=args.backaz_plot,
            filter_options=dict(freqmin=args.freq_bp[0], 
                                freqmax=args.freq_bp[1]),
            gem_include=utils.arg_split_comma(args.gem_include),
            gem_exclude=utils.arg_split_comma(args.gem_exclude))
    elif args.freq_range != None:
        # multiple frequencies specified

        #TODO what if you don't want octaves??
        pwr_min = math.log(args.freq_range[0], 2)
        pwr_max = math.log(args.freq_range[1], 2)
        freq_list = 2** np.arange(pwr_min, pwr_max, 1)

        #NOTE for Borah (48 cores per node)
        with ProcessPoolExecutor(max_workers=48) as pool:

            args_list = [[args.path_home, 
                        args.process, 
                        args.backaz_plot,
                        dict(freqmin=freqmin, 
                            freqmax=freqmax),
                        utils.arg_split_comma(args.gem_include),
                        utils.arg_split_comma(args.gem_exclude)]
                        for freqmin in freq_list for freqmax in freq_list if freqmin < freqmax]

            # run through different filters in parallel
            result = pool.map(main, *zip(*args_list))

    print("Completed Processing")
    with open(os.path.join(path_home, "code", "log", "pylog.txt"), "a") as f:
        print(f"{datetime.datetime.now()} \t\t Completed Processing", file=f)
