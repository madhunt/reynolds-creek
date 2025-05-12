'''
Plot beamforming results (slowness, semblance, backazimuth, and absolute power) 
for all five arrays in several frequency bands.
These plots can be used to determine times when signal is coherent based on the 
beamfomring outputs. 
'''
#!/usr/bin/python3

import os, sys, datetime, pytz
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.cm as cm
import matplotlib.dates as mdates

# import files from dir above
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils, settings, crossbeam

def main(path_processed, path_figures, t0, tf):

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

            # plot each veriable in outpur results
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


if __name__ == "__main__":
    # TIMES FOR 06 OCT
    t0 = datetime.datetime(2023, 10, 6, 18, 0, 0, tzinfo=pytz.UTC)
    tf = datetime.datetime(2023, 10, 6, 23, 0, 0, tzinfo=pytz.UTC)
    # TIMES FOR 07 OCT
    #t0 = datetime.datetime(2023, 10, 7, 16, 0, 0, tzinfo=pytz.UTC)
    #tf = datetime.datetime(2023, 10, 7, 21, 0, 0, tzinfo=pytz.UTC)

    main(settings.path_processed, settings.path_heli, 
         settings.path_station_gps, settings.path_figures,
         t0, tf)
