#!/usr/bin/python3

import datetime, os
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.dates as mdates
import numpy as np


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


def plot_backaz_slowness(output, path_home, filt_freq_str):
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

    fig, ax = plt.subplots(2, 1, tight_layout=True, sharex=True)
    im0 = simple_beamform_plot('backaz', output_constrain, fig, ax[0])
    
    # plot notable landmarks
    #ax[0].axhline(y=240)
    #ax[0].axhline(y=42)
    im1 = simple_beamform_plot('slowness', output, fig, ax[1])

    for ax_i in ax:
        ax_i.xaxis.set_major_locator(mdates.HourLocator(byhour=range(24)))
        ax_i.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        
        #TODO make this better, ok for now
        # plot fire start time
        ax_i.axvline(datetime.datetime(2024, 1, 14, 22, 45), 
                     color='k', linestyle='--', linewidth=2)

    #ax[1].set_xlim([datetime.datetime(2024, 1, 15, 2, 30), datetime.datetime(2024, 1, 15, 4)])
    ax[1].set_xlabel("UTC Time")
    fig.autofmt_xdate()
    fig.suptitle(f"Backazimuth and Slowness (filtered {filt_freq_str})")
    plt.savefig(os.path.join(path_home, "figures", f"backaz_slowness_{filt_freq_str}.png"), dpi=500)
    plt.close()
    #plt.show()
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
