#!/usr/bin/python3

import argparse

import glob, obspy, os
import numpy as np
from obspy.core.util import AttribDict
from obspy.signal.array_analysis import array_processing
import pandas as pd

from concurrent.futures import ProcessPoolExecutor

import plot_utils

def main(path_home, process=False, trace_plot=False, backaz_plot=False,
         filter_options=None):


    print("entered main")

    #FIXME path stuff
    #path_curr = os.path.dirname(os.path.realpath(__file__))
    #path_home = os.path.abspath(os.path.join(path_curr, '..'))
    path_data = os.path.join(path_home, "data")
    
    filt_freq_str = f"{filter_options['freqmin']}_{filter_options['freqmax']}"
    path_processed = os.path.join(path_data, "processed", 
                    f"processed_output_{filt_freq_str}.npy")

    #FIXME arguments into main or better
    #gem_list = ['138', '170', '155', '136', '150']#, '133']  # hopefully back az towards south
    gem_list=None
    

    filter_type = 'bandpass'
    # load data
    print("Loading and Filtering Data")
    data = load_data(path_data, gem_list=gem_list, 
                     filter_type=filter_type, **filter_options)
    
    # plot individual traces
    if trace_plot == True:
        print("Plotting Traces")
        plot_utils.plot_traces(data, path_home, filt_freq_str)

    if process == True:
        print("Processing Data")
        # fiter and beamform 
        output = process_data(data, path_processed, time_start=None, time_end=None)
    else:
        print("Loading Data")
        # data has already been processed
        output = np.load(path_processed)
    
    if backaz_plot == True:
        print("Plotting Backazimuth and Slowness")
        plot_utils.plot_backaz_slowness(output, path_home, filt_freq_str)
    
    return


#TODO add option to specify a date range
def load_data(path_data, gem_list=None, filter_type=None, **filter_options):
    '''
    Loads in and pre-processes array data.
        Loads all miniseed files in a specified directory into an obspy stream. 
        Assigns coordinates to all traces. If specified, filters data. If specified, 
        only returns a subset of gems (otherwise, returns full array).
    INPUTS
        path_data : str : Path to data folder. Should contain all miniseed files 
            under 'mseed' dir, and coordinates in .csv file(s).
        gem_list : list of str : Optional. If specified, should list Gem SNs 
            of interest. If `None`, will return full array.
        filter_type : str : Optional. Obspy filter type. Includes 'bandpass', 
            'highpass', and 'lowpass'.
        filter_options : dict : Optional. Obspy filter arguments. For 'bandpass', 
            contains freqmin and freqmax. For low/high pass, contains freq.
    RETURNS
        data : obspy stream : Stream of data traces for full array, or specified 
            Gems. Stats include assigned coordinates.
    '''
    # paths to mseed and coordinates
    path_mseed = os.path.join(path_data, "mseed", "*.mseed")
    path_coords = glob.glob(os.path.join(path_data, "gps", "*.csv" ))#"20240114_Bonfire_Gems.csv")

    # import data as obspy stream
    data = obspy.read(path_mseed)

    # import coordinates
    coords = pd.DataFrame()
    for file in path_coords:
        coords = pd.concat([coords, pd.read_csv(file)])
    coords["Name"] = coords["Name"].astype(str) # SN of gem
    
    # get rid of any stations that don't have coordinates
    data_list = [trace for trace in data.traces 
                    if trace.stats['station'] in coords["Station"].to_list()]
    # convert list back to obspy stream
    data = obspy.Stream(traces=data_list)

    # assign coordinates to stations
    for _, row in coords.iterrows():
        sn = row["Station"]
        for trace in data.select(station=sn):
            trace.stats.coordinates = AttribDict({
                'latitude': row["Latitude"],
                'longitude': row["Longitude"],
                'elevation': row["Elevation"]
            }) 
    
    if filter_type != None:
        # filter data
        data = data.filter(filter_type, **filter_options)

    # merge dates (discard overlaps and leave gaps)
    data = data.merge(method=0)
    
    if gem_list == None:
        # use full array
        return data
    else:
        # only use specified subset of gems
        data_subset = [trace for trace in data.traces if trace.stats['station'] in gem_list]
        data_subset = obspy.Stream(traces=data_subset)
        return data_subset
    
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

    # save output to .npy file
    np.save(path_processed, output)
    return output

if __name__ == "__main__":


    parser = argparse.ArgumentParser(
        description="Run traditional beamforming on specified mseed data.")
    parser.add_argument("-i", "--input",
            dest="input_path",
            type=str,
            help="path to dir with correct structure blah blah")
    args = parser.parse_args()

    #print(args.input_path)


    #main(args.input_path, True, False, True, dict(freqmin=0.5, freqmax=25))



    # run through different filters in parallel
    with ProcessPoolExecutor(max_workers=4) as pool:
        
        f_list = [0.5, 1, 2, 4, 8, 10, 15, 20, 25, 30, 35, 40]
        args_list = [ [args.input_path, True, False, True, dict(freqmin=freqmin, freqmax=freqmax)] 
                     for freqmin in f_list for freqmax in f_list if freqmin<freqmax]

        # now call main() with each set of args in parallel
        # map loops through each set of args
        result = pool.map(main, *zip(*args_list))


    print("test")

