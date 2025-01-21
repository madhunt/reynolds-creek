#!/usr/bin/python3
'''
Plots results of Monte-Carlo simulations for uncertainty in backazimuth.
Figures were used on 2025 AGU poster. Calculations were performed with 
code adapted from beamform.py (see GEOPH 522 project code).
'''
import os, datetime, pytz, sys
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# import files from dir above
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils, settings

def main(path_processed_uncert, path_heli, path_station_gps, path_figures,
         freqmin, freqmax, gps_perturb_scale):

    # define arrays and plot subtitles
    array_list = ["TOP", "JDSA", "JDNA", "JDNB", "JDSB"]
    subtitle_list = ["TOP (200x140 m, 44 sensors)", 
                     "JDSA (15x14 m, 3 sensors)", 
                     "JDNA (8.5x7 m, 4 sensors)", 
                     "JDNB (8x6.5 m, 4 sensors)", 
                     "JDSB (8x6.5 m, 4 sensors)"]

    fig, ax = plt.subplots(nrows=5, ncols=2, width_ratios=[3, 1], figsize=[12,11])
    freq_str = f"{freqmin}_{freqmax}"

    for i, array_str in enumerate(array_list):
        # LOAD PROCESSED BACKAZIMUTH DATA
        path_files = glob.glob(os.path.join(path_processed_uncert, f"output_{array_str}_gps_{gps_perturb_scale}*"))
        output_all = np.empty(shape=[179, 5, len(path_files)])
        for j, file in enumerate(path_files):
            output = np.load(file)
            output_all[:,:,j] = output
        
        # calculate mean backaz at each point in time
        # output contains 5 columns (Time, Semblance, Abs Power, Backaz, Slowness)
        output_mean = np.mean(output_all, axis=2)
        output_std = np.std(output_all, axis=2)

        # plot boxplot
        ax[i,1].boxplot(output_std[:,3], vert=False, widths=0.7,
                        medianprops=dict(color='red'))
        median = np.median(output_std[:,3])
        ax[i,1].text(median, 1.38, f"Median = {np.round(median, 2)}")
        ax[i,1].set_yticks([], [])
        ax[4,1].set_xlabel("Standard Deviation ($^o$)", fontsize=12)
        ax[0,1].set_title("Distribution of\nBackazimuth Uncertainty")
        ax[i,1].set_xlim([-5, 180])
        if i > 0:
            ax[i,0].sharex(ax[i-1,0])

        # convert to dataframe
        output = pd.DataFrame(data=output_mean, 
                            columns=["Time", "Semblance", "Abs Power", "Backaz", "Slowness"])
        # save time steps as datetime type
        output["Time"] = mdates.num2date(output["Time"])

        # add STD to dataframe
        output["Backaz Std"] = output_std[:,3]

        # sort by ascending semblance so brightest points are plotted on top
        # constrain data to only plot points with slownesses near 3 s/km
        output = output[output["Slowness"].between(2.5, 3.5)]

        # LOAD HELICOPTER TRACK DATA
        data = utils.adsb_kml_to_df(path_heli, latlon=True)
        # convert heli coords to dist/azimuth from current array
        data_heli = utils.coords_to_az(path_station_gps, data, array_str)
        # mask data points at the end of a long data gap (for plotting purposes)
        data_heli['Masked Azimuth'] = np.ma.masked_where(data_heli["Time"].diff() > datetime.timedelta(minutes=15), 
                                                         data_heli["Azimuth"])
        # remove any duplicate values
        data_heli = data_heli.set_index('Time')
        data_heli = data_heli[~data_heli.index.duplicated(keep='first')]

        # PLOT HELICOPTER DATA
        output = output.sort_values(by="Time", ascending=True)
        if i == 0:
            # make sure one axis has a label for legend entry
            ax[i,0].plot(data_heli.index, data_heli['Masked Azimuth'], '--', color='grey', 
                    alpha=1, label="True Helicopter Location")
            ax[i,0].plot(output["Time"], (output["Backaz"] + output["Backaz Std"]), '-', c='red',
                       label="Backazimuth Standard Deviation")
            ax[i,0].plot(output["Time"], (output["Backaz"] - output["Backaz Std"]), '-', c='tomato')
            ax[i,0].plot(output["Time"], output["Backaz"], 'o', c='black',
                       label="Mean Backazimuth")
        else:
            ax[i,0].plot(data_heli.index, data_heli['Masked Azimuth'], '--', color='grey', 
                    alpha=1)
            ax[i,0].plot(output["Time"], (output["Backaz"] + output["Backaz Std"]), '-', c='red')
            ax[i,0].plot(output["Time"], (output["Backaz"] - output["Backaz Std"]), '-', c='tomato')
            ax[i,0].plot(output["Time"], output["Backaz"], 'o', c='black')

        # SUBPLOT FORMATTING
        ax[i,0].set_title(subtitle_list[i], fontsize=12)
        if i < 4:
            # hide x-label for all but last subplot
            ax[i,0].xaxis.label.set_visible(False)
            ax[i,1].xaxis.label.set_visible(False)

        # format subplot y-axis
        ax[i,0].set_ylim([0, 360])
        ax[i,0].set_yticks(ticks=np.arange(0, 360+60, 90))
    
    # AXIS-SPECIFIC FORMATTING
    # add y-label on middle subplot
    ax[2,0].set_ylabel('Backazimuth [$^o$]', fontsize=12)
    # format x-axis on bottom subplot
    ax[4,0].set_xlabel("Local Time (US/Mountain) on 2023-10-07", fontsize=12)

    # set time axis limits
    time_min = datetime.datetime(year=2023, month=10, day=7, hour=10, minute=0, tzinfo=pytz.timezone("US/Mountain"))
    time_max = datetime.datetime(year=2023, month=10, day=7, hour=11, minute=30, tzinfo=pytz.timezone("US/Mountain"))
    ax[4,0].set_xlim([time_min, time_max])
    ax[4,0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=pytz.timezone("US/Mountain")))

    # FIGURE FORMATTING
    fig.suptitle(f"GPS Accuracy $\\pm${gps_perturb_scale} m:\nProcessed Backazimuth, Filtered {freqmin}-{freqmax} Hz", 
                 fontsize=16)
    # add legend outside axes on top subplot
    fig.tight_layout()
    
    #TODO FIXME LEGEND
    fig.subplots_adjust(bottom=0.1)
    fig.legend(fancybox=False, framealpha=1.0,
               ncol=3, loc='upper center',
               edgecolor="black", fontsize=12,
               bbox_to_anchor=(0.4,0.05), reverse=True)

    # SAVE FIGURE
    plt.savefig(os.path.join(path_figures, f"backaz_uncertainty_{gps_perturb_scale}m.png"), dpi=500)
    plt.close()
    return


if __name__ =="__main__":
    # settings
    freqmin = 24.0
    freqmax = 32.0
    gps_perturb_scale = 3 # m

    main(settings.path_processed_uncert, 
         settings.path_heli,
         settings.path_station_gps,
         settings.path_figures,
         freqmin, freqmax, gps_perturb_scale)
