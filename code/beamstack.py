#!/usr/bin/python3

import plot_utils, utils
import argparse, math, os, datetime, glob
import numpy as np
import pandas as pd
import obspy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from obspy.core.utcdatetime import UTCDateTime

def main():

    #date_list = ["2023-10-6", "2023-10-7"]

    day1 = 6
    day2 = 7
    time_start = UTCDateTime(f'2023-10-0{day1}T00:00:00')
    time_stop = UTCDateTime(f'2023-10-0{day2}T23:59:59')

    array_str = "JDNA"
    freqmin = 24.0
    freqmax = 32.0
    freq_str = f"{freqmin}_{freqmax}"

    # load processed infrasound data
    path_harddrive = os.path.join("/", "media", "mad", "LaCie 2 LT", "research", 
                                  "reynolds-creek")
    path_processed = os.path.join(path_harddrive, "data", "processed")
    output = pd.DataFrame()
    for day in np.arange(day1, day2+1, 1):
        date_str = f'2023-10-{day}'
        file = os.path.join(path_processed, f"processed_output_{array_str}_{date_str}_{freq_str}.pkl")
        output_tmp = pd.read_pickle(file)
        output = pd.concat([output, output_tmp])
    
    # load traces
    path_data = os.path.join(path_harddrive, "data", "mseed")
    path_harddrive = os.path.join("/", "media", "mad", "LaCie 2 LT", "research", 
                                  "reynolds-creek")
    path_coords = os.path.join("/", "home", "mad", "Documents", "research", 
                                  "reynolds-creek", "data", "gps", "RCEW_all_coordinates.csv")
    data = utils.load_data(path_data, path_coords, array_str=array_str,
                           gem_include=None, gem_exclude=None,
                           time_start=time_start, time_stop=time_stop,
                           freqmin=freqmin, freqmax=freqmax)
    
    # calculate beamstack spectrum

    # calculate cross spectrum
    # data = st = stream, win_len = 60 s


    # loop through data time
    #len

    #for data_section in 
    



    # will need freqs, power, and semblance







    return

if __name__ == "__main__":
    main()