#!/usr/bin/python3

import os, sys, datetime, pytz
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.dates as mdates
from matplotlib.ticker import LogFormatter


# import files from dir above
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils, settings, crossbeam


def main(path_processed, path_heli, path_station_gps, path_figures,
         t0, tf):


    # array names and number of sensors
    array_dict = {'TOP':42, 
                  'JDNA':3, 
                  'JDNB':3, 
                  'JDSA':4, 
                  'JDSB':4}
    # frequency bands
    freq_list = [(0.5, 2.0), (2.0, 4.0), (4.0, 8.0), (24.0, 32.0)]
    # properties and bounds, subtitles
    props_dict = {'Slowness':[(-0.1,6), (0,2,4,6), "Slowness\n(s/km)"], 
                  'Adj Semblance':[(-0.2,1.2), (0,0.5,1), "Adj\nSemblance"], 
                  'Backaz':[(-10,370), (0,90,180,270,360), "Backaz\n($^o$)"],
                  'Abs Power':[(0,3.5e12), (0,1e12,2e12,3e12), "Abs\nPower"]}

    for array_str in array_dict:
        print(array_str)
        
        # loop through different frequencies
        for freq_tup in freq_list:
            print(freq_tup)
            # initialize figure for each array for each freq band   
            fig, ax = plt.subplots(nrows=4, ncols=2, width_ratios=[4,1], figsize=[12,6])
            freq_str = f"{freq_tup[0]}_{freq_tup[1]}"

            # LOAD PROCESSED BACKAZIMUTH DATA
            output = utils.load_backaz_data(path_processed, array_str, freq_str, t0, tf)
            # sort by ascending semblance so brightest points are plotted on top
            output = output.sort_values(by="Semblance", ascending=True)
            # calculate adjusted semblance
            N = array_dict[array_str]
            output["Adj Semblance"] = N/(N-1) * (output["Semblance"] - 1/N)

            # plot each result
            for i, props_str in enumerate(props_dict):
                # plot property
                ax[i,0].plot(output.index, output[props_str], 'k.')
                ax[i,0].set_ylim(props_dict[props_str][0])
                ax[i,0].set_yticks(ticks=props_dict[props_str][1], labels=[str(i) for i in props_dict[props_str][1]])
                                   #fontsize=16)
                if props_str == "Abs Power":
                    ax[i,0].set_yticks(ticks=props_dict[props_str][1], 
                                    labels=[f"{i:.1E}" for i in props_dict[props_str][1]])
                ax[i,0].set_ylabel(props_dict[props_str][2], fontsize=16)

                # set tick label size
                ax[i,0].tick_params(axis='both', which='major', labelsize=16)
                ax[i,1].tick_params(axis='both', which='major', labelsize=16)
                if props_str == "Abs Power":
                    ax[i,0].yaxis.set_major_formatter(LogFormatter(10, labelOnlyBase=False)) 

                # color points based on tests
                #NOTE slowness filter here
                test_slow =  lambda arr: arr[arr['Slowness'].between(2.0, 3.5)]
                output_test_slow = test_slow(output)
                ax[i,0].plot(output_test_slow.index, output_test_slow[props_str], 'r.')

                # plot histogram of property
                ax[i,1].hist(output[props_str], orientation='horizontal')
                #TODO FIXME to make this a nice rounded value 
                #ax[i,1].set_title(f"mean {np.mean(output[props_str])}")
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
            fig.suptitle(f"{array_str}, {freq_str}", fontsize=20)
            #fig.suptitle(f"{props_str}, {freq_str}, {str(t0.date())}", fontsize=16)
            fig.tight_layout()
            fig.align_ylabels()
            # squish plots together
            plt.subplots_adjust(wspace=0.1, hspace=0)

            plt.savefig(os.path.join(path_figures, "beamform_results", f"{array_str}_{freq_str}_{str(t0.date())}.png"))
            plt.close()
    return

    
def calcs(path_processed, path_heli, path_station_gps, path_figures,
         t0, tf, freqmin, freqmax):
    # do some quick hillslope calculations
    top_24_32 = utils.load_backaz_data(path_processed, "TOP", "24.0_32.0", t0, tf)
    slow_avg = top_24_32["Slowness"].mean() #s/km
    speed_avg = 1 / slow_avg                #km/s

    jdsb_24_32 = utils.load_backaz_data(path_processed, "JDSB", "24.0_32.0", t0, tf)
    t15 = datetime.datetime(2023, 10, 6, 21, 15, 0, tzinfo=pytz.UTC)
    t16 = datetime.datetime(2023, 10, 6, 22, 0, 0, tzinfo=pytz.UTC)

    max_slow = jdsb_24_32[t15:t16]["Slowness"].max()






    return

    



if __name__ == "__main__":
    # choose times when helicopter is moving so we can compare points
    #freqmin = 4.0
    #freqmax = 8.0
    #freqmin = 24.0
    #freqmax = 32.0

    # TIMES FOR 06 OCT
    t0 = datetime.datetime(2023, 10, 6, 18, 0, 0, tzinfo=pytz.UTC)
    tf = datetime.datetime(2023, 10, 6, 23, 0, 0, tzinfo=pytz.UTC)
    # TIMES FOR 07 OCT
    #t0 = datetime.datetime(2023, 10, 7, 16, 0, 0, tzinfo=pytz.UTC)
    #tf = datetime.datetime(2023, 10, 7, 21, 0, 0, tzinfo=pytz.UTC)

    main(settings.path_processed, settings.path_heli, 
         settings.path_station_gps, settings.path_figures,
         t0, tf)

    #calcs(settings.path_processed, settings.path_heli, 
    #     settings.path_station_gps, settings.path_figures,
    #     t0, tf)