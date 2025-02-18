#!/usr/bin/python3
'''

'''
import obspy, os, sys, pytz, datetime
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as colors
import matplotlib.cm as cm
import pandas as pd

# import files from dir above
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils, settings, plot_utils


def main(path_output, path_station_gps, path_figures, array_str):


    # load in intersection points
    median_ints = pd.read_csv(path_output)
    median_ints["Color"] = median_ints.index
    median_ints["Time"] = pd.to_datetime(median_ints["Time"])
    median_ints = median_ints.set_index("Time")

    # now plot data filtered to 15 min
    #FIXME HACK this should use index[0] and index[-1]
    t0_list = [datetime.datetime(2023, 10, 7, 16, 0)+n*datetime.timedelta(minutes=15) 
               for n in range((21-16)*4)]
    #t0_list = [datetime.datetime(2023, 10, 6, 18, 0)+n*datetime.timedelta(minutes=15) 
    #           for n in range((23-18)*4)]
    for t0 in t0_list:
        # initialize figure
        fig, ax = plt.subplots(nrows=3, ncols=1, height_ratios=[1, 1, 3], 
                            tight_layout=True, figsize=[9,12])
        
        # choose colormap for all subplots
        cmap = cm.hsv

        # (0) PLOT EASTING VALUES
        ax[0].scatter(median_ints.index, median_ints["Easting"], 
                      c=median_ints["Color"], cmap=cmap, vmin=0, vmax=len(median_ints))
        ax[0].plot(median_ints.index, median_ints["True Easting"], "k-", alpha=1)
        ax[0].set_title("Easting")
        #ax[0].set_ylim(y_limits(median_ints["Easting"], median_ints["True Easting"]))
        ax[0].set_ylim([median_ints["True Easting"].min()-1e3, median_ints["True Easting"].max()+1e3])

        # (1) PLOT NORHTING VALUES
        ax[1].scatter(median_ints.index, median_ints["Northing"],
                      c=median_ints["Color"], cmap=cmap, vmin=0, vmax=len(median_ints))
        ax[1].plot(median_ints.index, median_ints["True Northing"], "k-", alpha=1)
        ax[1].set_title("Northing")
        #ax[1].set_ylim(y_limits(median_ints["Northing"], median_ints["True Northing"]))
        ax[1].set_ylim([median_ints["True Northing"].min()-5e3, median_ints["True Northing"].max()+5e3])

        ax[1].xaxis.set_major_locator(mdates.HourLocator(byhour=range(24), interval=1, tz=pytz.timezone("US/Mountain")))
        ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=pytz.timezone("US/Mountain")))
        ax[0].sharex(ax[1])


        # (2) PLOT MAP VIEW
        # add array reference points to plot
        for array_str in ["TOP", "JDNA", "JDSA", "JDNB", "JDSB"]:
            x, y = utils.station_coords_avg(path_station_gps, array_str=array_str, latlon=False)
            ax[2].plot(x, y, 'k^', markersize=10, zorder=3)

        # filter data to 15 min subsection
        tf = t0 + datetime.timedelta(minutes=15)
        data_filt = median_ints.between_time(t0.time(), tf.time())

        # plot helicopter
        ax[2].plot(median_ints["True Easting"], median_ints["True Northing"], 
                   alpha=0.5, color="black", zorder=0)
        ax[2].plot(data_filt["True Easting"], data_filt["True Northing"], 
                   alpha=1, color="black", zorder=2)

        # plot infrasound
        ax[2].scatter(median_ints["Easting"], median_ints["Northing"], 
                      alpha=0.3, edgecolors="black",
                      c=median_ints["Color"], cmap=cmap, vmin=0, vmax=len(median_ints),
                      zorder=1)
        # plot infrasound
        ax[2].scatter(data_filt["Easting"], data_filt["Northing"], 
                alpha=1, edgecolors="black",
                c=data_filt["Color"], cmap=cmap, vmin=0, vmax=len(median_ints),
                zorder=4)
        
        # set axis limits
        ax[2].set_xlim([median_ints["True Easting"].min(), median_ints["True Easting"].max()])
        ax[2].set_ylim([median_ints["True Northing"].min(), median_ints["True Northing"].max()])
        
        # and box around upper plots
        ax[0].vlines(x=[t0, tf], ymin=median_ints["Easting"].min(), ymax=median_ints["Easting"].max(),
                     linestyle="--", colors="black")
        ax[1].vlines(x=[t0, tf], ymin=median_ints["Northing"].min(), ymax=median_ints["Northing"].max(), 
                     linestyle="--", colors="black")



        fig.suptitle(os.path.basename(path_output))

        filename = f"helipath_{t0}_{tf}.png"
        print(filename)
        plt.savefig(os.path.join(path_figures, "scratch", filename), dpi=500)
        plt.close()





    return

if __name__ == "__main__":
    main(os.path.join(settings.path_output,
                      #"median_intersections_2.0-8.0Hz_20231006-18-00_20231006-23-00.csv"),
                      "median_intersections_2.0-8.0Hz_20231007-16-00_20231007-21-00.csv"),
                      #"median_intersections_24.0-32.0Hz_20231007-16-00_20231007-21-00.csv"),
                      #"median_intersections_24.0-32.0Hz_20231006-18-00_20231006-23-00.csv"),
         settings.path_station_gps,
         settings.path_figures,
         array_str="TOP")