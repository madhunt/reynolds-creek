#!/usr/bin/python3

import os, sys, datetime, pytz
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.dates as mdates

# import files from dir above
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils, settings, crossbeam


def main(path_processed, path_heli, path_station_gps, path_figures,
         t0, tf, freqmin, freqmax):

    

    def plot_result(result_str, bounds=None):

        array_list = ['TOP', 'JDNA', 'JDNB', 'JDSA', 'JDSB']
        freq_str = f"{freqmin}_{freqmax}"
        fig, ax = plt.subplots(nrows=5, ncols=2, sharey=True, width_ratios=[3,1], figsize=[12,9])

        for i, array_str in enumerate(array_list):
            # LOAD PROCESSED BACKAZIMUTH DATA
            output = utils.load_backaz_data(path_processed, array_str, freq_str, t0, tf)
            # sort by ascending semblance so brightest points are plotted on top
            output = output.sort_values(by="Semblance", ascending=True)

            ax[i,0].plot(output.index, output[result_str], 'k.')
            
            if bounds != None:
                ax[i,0].hlines(y=bounds[0], xmin=min(output.index), xmax=max(output.index), color='r', linestyle='--')
                ax[i,0].hlines(y=bounds[1], xmin=min(output.index), xmax=max(output.index), color='r', linestyle='--')
            
            ax[i,0].set_title(array_str)

            ax[i,1].hist(output[result_str], orientation='horizontal')
            ax[i,1].set_title(f"mean {np.mean(output[result_str])}")


            if i > 0:
                ax[i,0].sharex(ax[i-1,0])
                ax[i,1].sharex(ax[i-1,1])
            if i < 4:
                # hide x-label for all but last subplot
                ax[i,0].xaxis.label.set_visible(False)
        

        ax[4,0].set_xlim([t0, tf])
        ax[4,0].xaxis.set_major_locator(mdates.HourLocator(byhour=range(24), interval=1, tz=pytz.timezone("US/Mountain")))
        ax[4,0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=pytz.timezone("US/Mountain")))
        fig.suptitle(f"{result_str}, {freq_str}, {str(t0.date())}", fontsize=16)
        fig.tight_layout()

        plt.savefig(os.path.join(path_figures, "beamform_results", f"{result_str}_{freq_str}_{str(t0.date())}.png"))

        return
    
    plot_result("Slowness", bounds=[2.0, 3.2])
    plot_result("Semblance")
    plot_result("Abs Power")
    plot_result("Backaz")


    return




if __name__ == "__main__":
    # choose times when helicopter is moving so we can compare points
    freqmin = 2.0
    freqmax = 8.0
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
         t0, tf, freqmin, freqmax)
