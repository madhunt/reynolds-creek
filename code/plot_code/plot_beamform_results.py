#!/usr/bin/python3

import os, sys, datetime, pytz
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.cm as cm
import matplotlib.dates as mdates

# import files from dir above
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils, settings, crossbeam

def beamform_results_vs_time(path_processed, path_figures, t0, tf):
    '''
    Plot beamforming results (slowness, semblance, backazimuth, and absolute power) 
    for all five arrays in several frequency bands.
    These plots can be used to determine times when signal is coherent based on the 
    beamfomring outputs. 
    '''

    # array names and number of sensors
    array_dict = {'TOP':42, 
                  'JDNA':3, 
                  'JDNB':3, 
                  'JDSA':4, 
                  'JDSB':4}
    # frequency bands
    freq_list = [(0.5, 2.0), (2.0, 4.0), (4.0, 8.0), (24.0, 32.0)]
    # variables (yaxis bounds, tickmarks, and subtitles)
    var_dict = {'Slowness':[(-0.1,6), (0,2,4,6), "Slowness\n(s/km)"], 
                  'Adj Semblance':[(-0.2,1.2), (0,0.5,1), "Adjusted\nSemblance"], 
                  'Backaz':[(-10,370), (0,90,180,270,360), "Backaz\n($^o$)"],
                  'Abs Power':[(1e8,8e13), (1e8,1e10,1e12), "Absolute\nPower"]}

    for array_str in array_dict:
        print(array_str)

        # loop through frequency bands
        for freq_tup in freq_list:
            print(f"\tfreq: {freq_tup}")
            # initialize figure for each array for each freq band   
            fig, ax = plt.subplots(nrows=4, ncols=2, width_ratios=[4,1], figsize=[12,6])
            freq_str = f"{freq_tup[0]}_{freq_tup[1]}"

            # load processed beamfom data
            output = utils.load_beamform_results(path_processed, array_str, freq_str, t0, tf)
            # calculate adjusted semblance
            N = array_dict[array_str]
            output["Adj Semblance"] = N/(N-1) * (output["Semblance"] - 1/N)

            # plot each veriable in output results
            for i, var_str in enumerate(var_dict):
                
                # configure y-axis
                if var_str == "Abs Power":
                    ax[i,0].set_yscale("log")
                else:
                    ax[i,0].set_yticks(ticks=var_dict[var_str][1], labels=[str(i) for i in var_dict[var_str][1]])
                ax[i,0].set_ylim(var_dict[var_str][0])
                ax[i,0].set_ylabel(var_dict[var_str][2], fontsize=16)
                # set tick label size
                ax[i,0].tick_params(axis='both', which='major', labelsize=16)
                ax[i,1].tick_params(axis='both', which='major', labelsize=16)

                # plot all points
                ax[i,0].plot(output.index, output[var_str], 'k.')
                # plot points within slowness bounds as red
                test_slow = lambda arr: arr[arr['Slowness'].between(2.0, 3.5)]
                output_test_slow = test_slow(output)
                ax[i,0].plot(output_test_slow.index, output_test_slow[var_str], 'r.')

                # plot histogram of each variable
                if var_str == "Abs Power":
                    _, bins  = np.histogram(output[var_str], bins=8)
                    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
                    ax[i,1].hist(output[var_str], orientation='horizontal', bins=logbins)
                    ax[i,1].set_yscale("log")
                    ax[i,1].set_ylim([10e8,10e13])
                else:
                    ax[i,1].hist(output[var_str], orientation='horizontal')

                # remove y-axis ticklabels from histograms
                ax[i,1].set_yticklabels([])
                # make all plots share x-axis
                if i > 0:
                    ax[i,0].sharex(ax[i-1,0])
                    ax[i,1].sharex(ax[i-1,1])
                # hide x-label for all but last subplot
                if i < len(ax)-1:
                    ax[i,0].set_xticklabels([])
                    ax[i,0].xaxis.set_visible(False)
                    ax[i,1].set_xticklabels([])
                    ax[i,1].xaxis.set_visible(False)

            # set and format x-axis
            ax[len(ax)-1,0].set_xlim([t0, tf])
            ax[len(ax)-1,0].xaxis.set_major_locator(mdates.HourLocator(byhour=range(24), interval=1, tz=pytz.timezone("US/Mountain")))
            ax[len(ax)-1,0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=pytz.timezone("US/Mountain")))
            ax[len(ax)-1,1].set_xticks([0,100,200],[0,100,200])
           
            # add subtitle
            #fig.suptitle(f"{array_str}, {freq_str}", fontsize=20)
            fig.suptitle(f"{array_str}, {freq_str}, {str(t0.date())}", fontsize=20)
            fig.tight_layout()
            fig.align_ylabels()
            # squish plots together
            plt.subplots_adjust(wspace=0.1, hspace=0)
            plt.savefig(os.path.join(path_figures, "beamform_results", f"{array_str}_{freq_str}_{str(t0.date())}.png"))
            plt.close()
    return



def slowness_space_plots(path_processed, path_figures, t0, tf):

    # array names and number of sensors
    array_list = ['TOP', 'JDNA', 'JDNB', 'JDSA', 'JDSB']
    # frequency bands
    freq_list = [(0.5, 2.0), (2.0, 4.0), (4.0, 8.0), (24.0, 32.0)]

    fig, ax = plt.subplots(nrows=4, ncols=5, figsize=[12,10])
    #ax = ax.flat

    for j, array_str in enumerate(array_list):
        print(array_str)

        # loop through frequency bands
        for i, freq_tup in enumerate(freq_list):
            print(f"\tfreq: {freq_tup}")
            # initialize figure for each array for each freq band   
            freq_str = f"{freq_tup[0]}_{freq_tup[1]}"

            # load processed beamfom data
            output = utils.slowness_to_sx_sy(path_processed, array_str, freq_str, t0, tf)

            # plot slowness points
            time_int = output.index.astype(int)
            im = ax[i,j].scatter(output["sx"], output["sy"], 
                                 c=time_int, cmap="plasma")
            
            # plot circle at horizontal sound speed
            theta = np.arange(0, 360, 0.1)
            slow_sound = 1/0.343 #s/km
            ax[i,j].plot(slow_sound*np.cos(np.deg2rad(theta)), slow_sound*np.sin(np.deg2rad(theta)), 'k-')

            # plot center axes
            ax[i,j].hlines(y=0, xmin=-5, xmax=5, linewidth=0.5, color='k')
            ax[i,j].vlines(x=0, ymin=-5, ymax=5, linewidth=0.5, color='k')
            
            if i == 0:
                # set title as array string for each column
                ax[i,j].set_title(array_str)
            if i != 3:
                ax[i,j].xaxis.set_visible(False)
            else:
                ax[i,j].set_xlabel("$s_x$ (s/km)")

            if j == 0:
                # set y-label and title for each row
                row_title = f"{freq_tup[0]} - {freq_tup[1]} Hz"
                ax[i,j].set_ylabel(f"{row_title}\n$s_y$ (s/km)")
            else:
                ax[i,j].yaxis.set_visible(False)

            ax[i,j].set_xlim([-4.5, 4.5])
            ax[i,j].set_ylim([-4.5, 4.5])
            ax[i,j].set_aspect('equal', adjustable='box')


    fig.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    

    # add colorbar
    fig.subplots_adjust(right=0.87)
    cbar_ax = fig.add_axes([0.9, 0.17, 0.02, 0.6])
    cbar = fig.colorbar(im, cax=cbar_ax, aspect=12)
    cbar.ax.invert_yaxis()
    n = len(time_int)
    #idx = [int(n/4-15), int(n/2), int(3*n/4+15)]
    idx = [int(n/4), int(n/2), int(3*n/4)]
    cbar.set_ticks(time_int[idx])
    cbar.set_ticklabels([t.strftime("%H:%M") for t in output.index[idx]])
    cbar.set_label(f"Time on {str(t0.date())}", fontsize=12)



    #plt.subplots_adjust(wspace=0.1, hspace=0)
    # save plot
    plt.savefig(os.path.join(path_figures, "beamform_results", f"a_slowness_space_{str(t0.date())}.png"))
    plt.close()
    return







if __name__ == "__main__":
    # TIMES FOR 06 OCT
    t0 = datetime.datetime(2023, 10, 6, 18, 0, 0, tzinfo=pytz.UTC)
    tf = datetime.datetime(2023, 10, 6, 23, 0, 0, tzinfo=pytz.UTC)
    # TIMES FOR 07 OCT
    #t0 = datetime.datetime(2023, 10, 7, 16, 0, 0, tzinfo=pytz.UTC)
    #tf = datetime.datetime(2023, 10, 7, 21, 0, 0, tzinfo=pytz.UTC)
    #beamform_results_vs_time(settings.path_processed, settings.path_figures, t0, tf)
    

    # slowness space plots
    t0 = datetime.datetime(2023, 10, 6, 20, 30, 0, tzinfo=pytz.UTC)
    tf = datetime.datetime(2023, 10, 6, 21, 0, 0, tzinfo=pytz.UTC)
    slowness_space_plots(settings.path_processed, settings.path_figures, t0, tf)

