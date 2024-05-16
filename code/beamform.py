#!/usr/bin/python3

import plot_utils, utils
import argparse, glob, obspy, os
import numpy as np
import pandas as pd
from obspy.core.util import AttribDict
from obspy.signal.array_analysis import array_processing
#from concurrent.futures import ProcessPoolExecutor

def main(path_home, process=False, backaz_plot=False,
         filter_options=None, gem_include=None, gem_exclude=None):


    path_data = os.path.join(path_home, "data", "mseed")
    filt_freq_str = f"{filter_options['freqmin']}_{filter_options['freqmax']}"
    path_processed = os.path.join(path_data, "processed", 
                    f"processed_output_{filt_freq_str}.pkl")

    # load data
    print("Loading and Filtering Data")
    #TODO option for highpass or lowpass only
    data = utils.load_data(path_data, gem_include=gem_include, gem_exclude=gem_exclude, 
                     filter_type='bandpass', **filter_options)
    
    if process == True:
        # fiter and beamform 
        print("Processing Data")
        output = process_data(data, path_processed, time_start=None, time_end=None)
    else:
        # data has already been processed
        print("Loading Data")
        output = pd.read_pickle(path_processed)
        #output = np.load(path_processed)
    
    if backaz_plot == True:
        print("Plotting Backazimuth and Slowness")
        plot_utils.plot_backaz_slowness(output, path_home, filt_freq_str)
    
    return


def process_data(data, path_processed, time_start=None, time_end=None):
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
        output : np array : array with 5 rows of output from array_processing. Rows are: timestamp, 
            relative power (semblance), absolute power, backazimuth (from 0-360), and slowness (in s/km).
    '''
    # if times are not provided, use max/min start and end times from gems
    if time_start == None:
        # specify start time
        time_start = max([trace.stats.starttime for trace in data])
    if time_end == None:
        time_end = min([trace.stats.endtime for trace in data])

    #FIXME probably can clean this up when these change
    process_kwargs = dict(
        # slowness grid (in [s/km])
        sll_x=-4.0, slm_x=4.0, sll_y=-4.0, slm_y=4.0, sl_s=0.1,
        # sliding window
        win_len=10, win_frac=0.50,
        # frequency
        frqlow=2.0, frqhigh=10.0, prewhiten=0,
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
    #np.save(path_processed, output)
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
    parser.add_argument("-b", "--plot-backaz",
                        dest="backaz_plot",
                        default=False,
                        action="store_true",
                        help="Flag if backazimuth plots should be created.")
    parser.add_argument("-f", "--freqs",
            nargs=2,
            dest="freqs",
            metavar=("freqmin", "freqmax"),
            type=float,
            help="Min and max frequencies for bandpass filter.")

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
    
    #TODO move this to utils
    def arg_split_comma(arg):
        if arg != None:
            arg = [s for s in arg.split(",")]
            return arg
        else:
            # return None
            return arg

    main(path_home=args.path_home, 
         process=args.process, 
         backaz_plot=args.backaz_plot,
         filter_options=dict(freqmin=args.freqs[0], 
                             freqmax=args.freqs[1]),
        gem_include=arg_split_comma(args.gem_include),
        gem_exclude=arg_split_comma(args.gem_exclude))

    # run through different filters in parallel
#    with ProcessPoolExecutor(max_workers=4) as pool:
#        f_list = [0.5, 1, 2, 4, 8, 10, 15, 20, 25, 30, 35, 40]
#        args_list = [ [args.input_path, True, False, True, dict(freqmin=freqmin, freqmax=freqmax)] 
#                     for freqmin in f_list for freqmax in f_list if freqmin <= 2*freqmax]
#        # now call main() with each set of args in parallel
#        # map loops through each set of args
#        result = pool.map(main, *zip(*args_list))

    print("Completed Processing")
