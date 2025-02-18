#!/usr/bin/python3
'''

'''
import obspy, os, sys, pytz, datetime
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as patches
import pandas as pd

# import files from dir above
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils, settings, plot_utils


def main(path_output, path_station_gps, path_figures, array_str):


    # run crossbeam and save median ints as pkl 
    # for ENTIRE burn window 
    median_ints = pd.read_csv(path_output)
    median_ints["Time"] = pd.to_datetime(median_ints["Time"])
    median_ints = median_ints.set_index("Time")

    # then plot northing/easting of heli over time
    # and northing/easting of triangulated points over time

    # now plot data filtered to 15 min
    t0_list = [datetime.datetime(2023, 10, 7, 16, 0)+n*datetime.timedelta(minutes=15) 
               for n in range((22-16)*4)]
    for t0 in t0_list:
        tf = t0 + datetime.timedelta(minutes=15)
        fig, ax = plt.subplots(nrows=3, ncols=1, height_ratios=[1, 1, 3], 
                            tight_layout=True, figsize=[9,12])

        # set y-limits to encompass data within one standard deviation
        def y_limits(df_col):
            mean = df_col.mean()
            std = df_col.std()
            ymin = mean - std
            ymax = mean + std
            return [ymin, ymax]
        
        ax[0].plot(median_ints.index, median_ints["Easting"], "k.")
        ax[0].plot(median_ints.index, median_ints["True Easting"], "r-", alpha=0.5)
        ax[0].set_title("Easting")
        #ax[0].set_ylim(y_limits(median_ints["Easting"]))
        ax[0].set_ylim([median_ints["True Easting"].min(), median_ints["True Easting"].max()])

        ax[1].plot(median_ints.index, median_ints["Northing"], "k.")
        ax[1].plot(median_ints.index, median_ints["True Northing"], "r-", alpha=0.5)
        ax[1].set_title("Northing")
        #ax[1].set_ylim(y_limits(median_ints["Northing"]))
        ax[1].set_ylim([median_ints["True Northing"].min(), median_ints["True Northing"].max()])

        ax[1].xaxis.set_major_locator(mdates.HourLocator(byhour=range(24), interval=1, tz=pytz.timezone("US/Mountain")))
        ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=pytz.timezone("US/Mountain")))
        ax[0].sharex(ax[1])


        ax[2].plot(median_ints["True Easting"], median_ints["True Northing"],
                "-", color="purple", alpha=0.5)
        # add array reference points to plot
        for array_str in ["TOP", "JDNA", "JDSA", "JDNB", "JDSB"]:
            x, y = utils.station_coords_avg(path_station_gps, array_str=array_str, latlon=False)
            ax[2].plot(x, y, 'b^', markersize=10)
        data_filt = lambda df: df.between_time(t0.time(), tf.time())
        ax[2].plot(data_filt(median_ints)["True Easting"], 
                data_filt(median_ints)["True Northing"], 
                "-", color="purple")
        # and box around upper plots
        ax[0].vlines(x=[t0, tf], ymin=median_ints["True Easting"].min(), ymax=median_ints["True Easting"].max(), colors="purple")
        ax[1].vlines(x=[t0, tf], ymin=median_ints["True Northing"].min(), ymax=median_ints["True Northing"].max(), colors="purple")



        filename = f"helipath_{t0}_{tf}.png"
        plt.savefig(os.path.join(path_figures, "scratch", filename), dpi=500)
        plt.close()





    return

if __name__ == "__main__":
    main(os.path.join(settings.path_output,
                      "median_intersections_24.0-32.0Hz_20231007-16-00_20231007-21-00.csv"),
         settings.path_station_gps,
         settings.path_figures,
         array_str="TOP")