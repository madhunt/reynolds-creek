#!/usr/bin/python3

import datetime, os
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.dates as mdates
import numpy as np
import math


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
        plt.savefig(os.path.join(path_home, "figures", f"traces_{id_str_file}_{curr_fig}.png"), dpi=500)
        plt.close()
    return


def plot_backaz_slowness(output, path_home, filt_freq_str):
    #TODO make this able to reuse for other data #FIXME
    
    #TODO have this as a func with user input slow min/max at some point
    # only plot backaz with slowness near 3 s/km
    slow_min = 2.9
    slow_max = 3.1
    output_constrain = output[output["Slowness"].between(slow_min, slow_max)]

    fig, ax = plt.subplots(2, 1, tight_layout=True, sharex=True)
    im0 = simple_beamform_plot('backaz', output_constrain, fig, ax[0])
    im1 = simple_beamform_plot('slowness', output, fig, ax[1])

    for ax_i in ax:
        ax_i.xaxis.set_major_locator(mdates.HourLocator(byhour=range(24), interval=2))
        ax_i.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        
    ax[1].set_xlabel("UTC Time")
    fig.autofmt_xdate()
    fig.suptitle(f"Backazimuth and Slowness (filtered {filt_freq_str})")

    plt.savefig(os.path.join(path_home, "figures", f"backaz_slowness_{filt_freq_str}.png"), dpi=500)
    plt.close()
    return


#TODO think about this... there's probably a neater way to consolidate and reuse this func
# will leave for now and wait for other plotting needs
def simple_beamform_plot(plot_type, output, fig, ax):
    '''
    Plots beackazimuth or slowness on given figure, along with colorbar. Assumes output array 
    includes columns in the same order as output of array_processing().
    INPUTS
        plot_type : str : 'backaz' or 'slowness'
        output : np array : array with 5 rows containing output of array_processing() function. 
            This includes time, semblance, abs power, backazimuth, and slowness.
        fig, ax : pyplot handles(?) : handles(?) to figure and axes
    RETURNS
        im : handle(?) : handle to image
    '''
    
    # sort by semblance to show on plot better
    output = output.sort_values(by="Semblance", ascending=True)

    if plot_type == 'backaz':
        yvar = output["Backaz"]
        ax.set_ylabel("Backazimuth [$^o$]")
        ax.set_ylim([0, 360])
        ax.set_yticks(ticks=np.arange(0, 360+60, 60))
    elif plot_type == 'slowness':
        yvar = output["Slowness"]
        ax.set_ylabel("Slowness [s/km]")
        ax.set_yticks(ticks=np.arange(0, int(max(output["Slowness"]))+1, 1))
    else:
        raise Exception("Plot type not supported!")

    im = ax.scatter(output["Time"], yvar, c=output["Semblance"],
                    alpha=0.7, edgecolors='none', cmap='plasma',
                    vmin=min(output["Semblance"]), vmax=max(output["Semblance"]))
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Semblance")
    return im
