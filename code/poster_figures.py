#!/usr/bin/python3

import datetime, glob, obspy, os
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.dates as mdates
import numpy as np
from obspy.core.util import AttribDict
from obspy.signal.array_analysis import array_processing
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

def main(trace_plot=False, backaz_plot=False, filename=None):#, filter_options=None):

    #FIXME path stuff
    path_curr = os.path.dirname(os.path.realpath(__file__))
    path_home = os.path.abspath(os.path.join(path_curr, '..'))
    path_data = os.path.join(path_home, "data")
    
    #filt_freq_str = f"{filter_options['freqmin']}_{filter_options['freqmax']}"
    path_processed = os.path.join(path_data, "processed", filename)

    #FIXME arguments into main or better
    gem_list = ['138', '170', '155', '136', '150']#, '133']  # hopefully backaz towards south
    #FIXME removed 133 because data didnt start until 01-15T02
    

    filter_type = 'bandpass'
    # load data
    print("Loading and Filtering Data")
    data = load_data(path_data, gem_list=gem_list, 
                     filter_type=None)
    
    # plot individual traces
    if trace_plot == True:
        print("not plotting traces")
        #print("Plotting Traces")
        #plot_traces(data, path_home, filt_freq_str)

   # if process == True:
   #     print("Processing Data")
   #     # fiter and beamform 
   #     output = process_data(data, path_processed, time_start=None, time_end=None)
    print("Loading Data")
    # data has already been processed
    output = np.load(path_processed)
    
    if backaz_plot == True:
        print("Plotting Backazimuth")
        plot_poster_backaz(output, path_home)#, filt_freq_str)
    
    return

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
    path_coords = glob.glob(os.path.join(path_data, "*.csv" ))#"20240114_Bonfire_Gems.csv")

    # import data as obspy stream
    data = obspy.read(path_mseed)

    # import coordinates
    coords = pd.DataFrame()
    for file in path_coords:
        coords = pd.concat([coords, pd.read_csv(file)])
    coords["Name"] = coords["Name"].astype(str) # SN of gem
    
    # get rid of any stations that don't have coordinates
    data_list = [trace for trace in data.traces 
                    if trace.stats['station'] in coords["Name"].to_list()]
    # convert list back to obspy stream
    data = obspy.Stream(traces=data_list)

    # assign coordinates to stations
    for _, row in coords.iterrows():
        sn = row["Name"]
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
    
def plot_traces(data, path_home, filt_freq_str):
    '''
    Plots individual traces from each Gem and saves at 
    path_home/figures/traces_{freqmin}_{freqmax}.png.
    INPUTS
        data : obspy stream : merged stream with data from all Gems to plot
        path_home : str : path to main dir. Figure will be saved in "figures" subdir.
        filt_freq_str : str : string with data filter frequencies ("freqmin_freqmax")
    RETURNS
        Saves figure at path_home/figures/traces.png
    '''
    # define number of traces
    n = len(data)
    #TODO make separate figures if plotting full array (if n > certain number)
    fig, ax = plt.subplots(n, 1, sharex=True, sharey=True, tight_layout=True)
    color = cm.rainbow(np.linspace(0, 1, n))

    for i, trace in enumerate(data):
        ax[i].plot(trace.times("matplotlib"), trace.data, c=color[i])
        ax[i].grid()
        ax[i].set_ylabel(trace.stats["station"])
        ax[i].xaxis_date()

        #TODO change this -- how to do this better?
        ax[i].set_ylim([-100, 100])

        #TODO make this better, ok for now
        # plot fire start time
        ax[i].axvline(datetime.datetime(2024, 1, 14, 22, 45), 
                     color='k', linestyle='--', linewidth=2)

    # label and format bottom x-axis
    ax[n-1].set_xlabel("UTC Time")
    fig.autofmt_xdate()
    fig.suptitle(f"Individual Gem Traces (filtered {filt_freq_str})")

    plt.savefig(os.path.join(path_home, "figures", f"traces_{filt_freq_str}.png"), dpi=500)
    plt.close()
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

    # save output to .npy file
    np.save(path_processed, output)
    return output

def plot_poster_backaz(output, path_home):#, filt_freq_str):
    #TODO make this able to reuse for other data #FIXME
    
    #TODO have this as a func with user input slow min/max at some point
    # only plot backaz with slowness near 3 s/km
    slow_min = 2.8
    slow_max = 3.2
    output_constrain = []
    for col in output.T:
        col_constrain = [col[i] for i in range(len(col)) if slow_min < output[:,4][i] < slow_max]
        col_constrain = np.array(col_constrain)
        output_constrain.append(col_constrain)
    output_constrain = np.array(output_constrain).T


    fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=[10, 8])

    im = ax.scatter(output_constrain[:,0], output_constrain[:,3], c=output_constrain[:,1],
                    alpha=0.6, edgecolors='none', cmap='plasma',
                    vmin=min(output_constrain[:,1]), vmax=max(output_constrain[:,1]))
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Semblance", fontsize=12)

    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(24)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    
    #TODO make this better, ok for now
    # plot fire start time
    ax.axvline(datetime.datetime(2024, 1, 14, 22, 45), 
                    color='r', alpha=0.7, linestyle='-', linewidth=2, label="Time of bonfire start")
    # plot notable landmarks
    #ax.axhline(y=235) # goat falls
    ax.axhline(y=173, color='k', alpha=1, linestyle=':', linewidth=2, label="Direction to bonfire") 
    ax.axhline(y=240, color='k', alpha=0.7, linestyle='-.', linewidth=2, label="Direction to Goat Falls")
    ax.axhline(y=48, color='k', alpha=0.7, linestyle="--", linewidth=2, label="Direction to Stanley, ID")

    #ax[1].set_xlim([datetime.datetime(2024, 1, 15, 2, 30), datetime.datetime(2024, 1, 15, 4)])
    ax.set_ylabel("Direction of arrival, clockwise from N [$^o$]", fontsize=15)
    ax.set_ylim([0, 360])
    ax.set_yticks(ticks=np.arange(0, 360+60, 60))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    ax.set_xlabel("UTC Time", fontsize=15)
    fig.autofmt_xdate()

    ax.legend(loc="upper right", fontsize=12)

    fig.suptitle(f"Direction of Arrival (0.5 to 4 Hz)", fontsize=18)
    plt.savefig(os.path.join(path_home, "figures", "poster", f"backaz_0.5_4.png"), dpi=500)
    #plt.show()
    plt.close()
    return

#TODO think about this... there's probably a neater way to consolidate and reuse this func
# will leave for now and wait for other plotting needs
def simple_beamform_plot(plot_type, output, fig, ax):
    '''
    Plots backazimuth or slowness on given figure, along with colorbar. Assumes output array 
    includes columns in the same order as output of array_processing().
    INPUTS
        plot_type : str : 'backaz' or 'slowness'
        output : np array : array with 5 rows containing output of array_processing() function. 
            This includes time, semblance, abs power, backazimuth, and slowness.
        fig, ax : pyplot handles(?) : handles(?) to figure and axes
    RETURNS
        im : handle(?) : handle to image
    '''
    if plot_type == 'backaz':
        yvar = output[:,3]
        ax.set_ylabel("Backazimuth [$^o$]")
        ax.set_ylim([0, 360])
        ax.set_yticks(ticks=np.arange(0, 360+60, 60))
    elif plot_type == 'slowness':
        yvar = output[:,4]
        ax.set_ylabel("Slowness [s/km]")
        ax.set_yticks(ticks=np.arange(0, int(max(output[:,4]))+1, 1))
    else:
        raise Exception("Plot type not supported!")

    im = ax.scatter(output[:,0], yvar, c=output[:,1],
                    alpha=0.6, edgecolors='none', cmap='plasma',
                    vmin=min(output[:,1]), vmax=max(output[:,1]))
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Semblance")
    return im

if __name__ == "__main__":

    main(trace_plot=False,
        backaz_plot=True,
        filename="processed_output_0.5_4.npy")

    ## run through different filters in parallel
    #with ProcessPoolExecutor(max_workers=4) as pool:
        
        #f_list = [0.5, 1, 2, 4, 8, 10, 15, 20, 25, 30, 35, 40]
        #args_list = [ [True, True, True, dict(freqmin=freqmin, freqmax=freqmax)] 
                     #for freqmin in f_list for freqmax in f_list if freqmin<freqmax]
        #print(args_list)

        ## now call main() with each set of args in parallel
        ## map loops through each set of args
        #result = pool.map(main, *zip(*args_list))
    
