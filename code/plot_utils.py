#!/usr/bin/python3

import datetime, os
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.dates as mdates
import numpy as np
import math
import scipy as sci


def plot_traces(data, path_home, id_str):
    '''
    Plots individual traces, with a max of 5 subplots per figure.
    INPUTS
        data : obspy stream : Merged stream with data from all Gems to plot
        path_home : str : Path to main dir. Figure will be saved in "figures" subdir.
        id_str : str : Additional ID to put in figure title/filename. For instance, 
            id_str = "Raw Data" will add "Raw Data" to the title and "raw_data" to 
            the filename. 
    RETURNS
        Saves figure at path_home/figures/
    '''
    # define number of traces
    N = len(data)
    num_fig = int(math.ceil(N/5))
    color = cm.rainbow(np.linspace(0, 1, N))

    # downsample data for faster plotting
    data = data.split().decimate(10).merge(method=0)

    # choose ylim values from min/max of all data
    y_max = max(data.max())
    y_min = min(data.max()) # no data.min(), but max() includes most negative vals

    for n in np.arange(N, step=5):
        curr_fig = int(np.ceil(n/5)+1)
        print(f"    Plotting Figure {curr_fig} of {num_fig}")

        fig, ax = plt.subplots(5, 1, sharex=True, sharey=True, tight_layout=True)

        for i in range(5):
            if n+i < len(data):
                trace = data[n+i]

                ax[i].plot(trace.times("matplotlib"), trace.data, c=color[n+i])
                ax[i].set_ylabel(trace.stats["station"])
                ax[i].grid()
                ax[i].xaxis_date()
                ax[i].set_ylim(y_min, y_max)
            else:
                fig.delaxes(ax[i])
        # label and format bottom x-axis
        ax[-1].set_xlabel("UTC Time")
        fig.autofmt_xdate()
        fig.suptitle(f"Individual Gem Traces for {id_str} ({curr_fig}/{num_fig})")

        id_str_file = id_str.lower().replace(" ", "_")
        plt.savefig(os.path.join(path_home, "figures", "traces",        
                                 f"traces_{id_str_file}_{curr_fig}.png"), dpi=500)
        plt.close()
    return


#TODO get coords and plot stations
# do this in pre-processing
def plot_station_map():

    return


def plot_slowness(output, path_home, id_str):
    '''
    Creates slowness-space plot for output from beamforming. 
    INPUTS
        output : pandas dataframe : Result from beamforming processing. Contains timestamp,
            semblance, abs power, backazimuth (0-360), and slowness (s/km).
        path_home : str : Path to main dir. Figure will be saved in "figures" subdir.
        id_str : str : Additional ID to put in figure title/filename. For instance, 
            id_str = "0.5_4.0" will 
    RETURNS
        Saves figure at path_home/figures/
    '''
    #FIXME
    #FIXME
    #FIXME
    #FIXME this does not work currently. need to think about what I actually want here
    # filter by time so not all points are plotted 
    
    # only filter out points with slowness close to 3 s/km
    slow_min = 2.9
    slow_max = 3.1
    output = output[output["Slowness"].between(slow_min, slow_max)]

    slist = np.arange(-4, 4, 0.1)
    lim = slist.max()
    
    theta = np.deg2rad( (-1*output['Backaz'].to_numpy() + 90)%360 )
    r = output['Slowness'].to_numpy()
    z = output['Semblance'].to_numpy()
    

    # define colors
    N = 100#len(output)
    color = plt.cm.rainbow(np.linspace(0, 1, N))

    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': 'polar'})
    for i in range(100): 
        ax.scatter(theta[i], r[i], color=color[i])

    plt.xlabel('$s_x$ (s/km)')
    plt.ylabel('$s_y$ (s/km)')
    plt.title("Slowness Spectrum")
    plt.show()


    #id_str_file = id_str.lower().replace(" ", "_")
    #plt.savefig(os.path.join(path_home, "figures", f"slowness_{id_str_file}.png"), dpi=500)
    #plt.close()
    return


def plot_backaz(output, path_home, subtitle_str, file_str=None):
    '''
    Plot backazimuth over time from output of array processing. 
    INPUTS:
        output : pandas df : Result from beamforming with the columns 
            Time (datetime), Semblance, Abs Power, Backaz (0-360), and Slowness.
        path_home : str : Path to main dir. Figure will be saved in "figures" subdir.
        subtitle_str : str : Subtitle for plot. Usually contains bandpass frequencies 
            (e.g. "Filtered 24-32 Hz")
        file_str : str or None : String to append on end of filename to uniquely save figure 
            (e.g. "24.0_32.0"). If None, function returns a handle to the figure and axes, and does 
            NOT save the figure. 
    RETURNS:
        If file_str=None, returns handle to the figure and axes. Figure is NOT saved.
        Otherwise, figure is saved as path_home/figures/backaz_{file_str}.png
    '''
    # sort by ascending semblance so brightest points are plotted on top
    output = output.sort_values(by="Semblance", ascending=True)

    # constrain data to only plot points with slownesses near 3 s/km
    slow_min = 2.5
    slow_max = 3.5
    output = output[output["Slowness"].between(slow_min, slow_max)]

    # create figure
    fig, ax = plt.subplots(1, 1, figsize=[7, 5], tight_layout=True)
    im = ax.scatter(output["Time"], output['Backaz'], c=output["Semblance"],
                    alpha=0.7, edgecolors='none', cmap='plasma',
                    vmin=min(output["Semblance"]), vmax=max(output["Semblance"]))
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Semblance")

    # format y-axis
    ax.set_ylabel("Backazimuth [$^o$]")
    ax.set_ylim([0, 360])
    ax.set_yticks(ticks=np.arange(0, 360+60, 60))

    # format x-axis
    ax.set_xlabel("Mountain Time (Local)")
    ax.set_xlim([output["Time"].min(), output["Time"].max()])
    hours_num = (output["Time"].max() - output["Time"].min()).total_seconds() / 3600
    tick_spacing = int(hours_num // 15) # make x-axis look nice (good number of ticks)
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(24), interval=tick_spacing))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M", tz="US/Mountain"))
    fig.autofmt_xdate()

    # add titles
    fig.suptitle(f"Backazimuth")
    ax.set_title(subtitle_str, fontsize=10)

    if file_str == None:
        return fig, ax
    else: 
        # save figure
        plt.savefig(os.path.join(path_home, "figures", f"backaz_{file_str}.png"), dpi=500)
        plt.close()
        return


