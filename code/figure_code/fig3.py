#!/usr/bin/python3
import os, sys, datetime, pytz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.colors as colors
import matplotlib.dates as mdates
# import from personal scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import triangulate

def main():
    # figure with all array backaz in subplots
    # set parameters (TODO change these to command line inputs)
    date_list = ["2023-10-6", "2023-10-7"]#, "2023-10-6"]#, "2023-10-5"]
    freqmin = 2.0
    freqmax = 8.0

    # FIXME this is ugly but not as bad as it once was
    path_harddrive = os.path.join("/", "media", "mad", "LaCie 2 LT", "research", "reynolds-creek")
    path_home = os.path.join("/", "home", "mad", "Documents", "research", "reynolds-creek")
    # path nonsense
    path_heli = os.path.join(path_harddrive, "data", "helicopter")
    path_processed = os.path.join(path_harddrive, "data", "processed")
    path_station_gps = os.path.join(path_harddrive, "data", "gps")
    path_figures = os.path.join(path_home, "figures")

    plot_backaz_heli_allarrays(path_heli, path_processed, path_station_gps,
                     path_figures, date_list, freqmin, freqmax)
    return

def plot_backaz_heli_allarrays(path_heli, path_processed, path_station_gps,
                     path_figures, date_list, freqmin, freqmax):
    '''
    Plot calculated backazimuths overlaid with helicopter location data for 
        each array (TOP, JDNA, JDNB, JDSA, and JDSB). 
    INPUTS:

    RETURNS:
    '''
    array_list = ["TOP", "JDNA", "JDNB", "JDSA", "JDSB"]
    subtitle_list = ["TOP (42 sensors)", "JDNA (3 sensors)", "JDNB (3 sensors)", 
                     "JDSA (4 sensors)", "JDSB (4 sensors)"]

    fig, ax = plt.subplots(nrows=5, ncols=1, sharex=True, figsize=[12,9])
    freq_str = f"{freqmin}_{freqmax}"

    for i, array_str in enumerate(array_list):
        # LOAD PROCESSED BACKAZIMUTH DATA
        output = pd.DataFrame()
        for date_str in date_list:
            file = os.path.join(path_processed, f"processed_output_{array_str}_{date_str}_{freq_str}.pkl")
            output_tmp = pd.read_pickle(file)
            output = pd.concat([output, output_tmp])
        # sort by ascending semblance so brightest points are plotted on top
        output = output.sort_values(by="Semblance", ascending=True)
        # constrain data to only plot points with slownesses near 3 s/km
        output = output[output["Slowness"].between(2.5, 3.5)]

        # PLOT BACKAZIMUTHS
        # create truncated greyscale colormap
        new_cmap = truncate_colormap(plt.get_cmap("Greys"), 0.4, 1.0)
        im = ax[i].scatter(output["Time"], output['Backaz'], c=output["Semblance"],
                        alpha=1, edgecolors='none', cmap=new_cmap,
                        vmin=0, vmax=1)
        # format subplot y-axis
        ax[i].set_ylim([0, 360])
        ax[i].set_yticks(ticks=np.arange(0, 360+60, 90))

        # LOAD HELICOPTER TRACK DATA
        data = triangulate.adsb_kml_to_df(path_heli)
        # convert heli coords to dist/azimuth from current array
        data_heli = triangulate.helicoords_to_az(path_station_gps, data, array_str)
        # mask data points at the end of a long data gap (for plotting purposes)
        data_heli['Masked Azimuth'] = np.ma.masked_where(data_heli["Time"].diff() > datetime.timedelta(minutes=15), 
                                                         data_heli["Azimuth"])
        # remove any duplicate values
        data_heli = data_heli.set_index('Time')
        data_heli = data_heli[~data_heli.index.duplicated(keep='first')]

        # PLOT HELICOPTER DATA
        ax[i].plot(data_heli.index, data_heli['Masked Azimuth'], '-', color='red', 
                alpha=0.6)
        if i == 0:
            # make sure one axis has a label for legend entry
            ax[i].plot(data_heli.index, data_heli['Masked Azimuth'], '-', color='red', 
                    alpha=0.6, label="Helicopter,\nrelative\nto array")

        # SUBPLOT FORMATTING
        ax[i].set_title(subtitle_list[i], fontsize=12)
        if i < 4:
            # hide x-label for all but last subplot
            ax[i].xaxis.label.set_visible(False)
    
    # AXIS-SPECIFIC FORMATTING
    # add legend outside axes on top subplot
    fig.legend(fancybox=False, framealpha=1.0, 
               edgecolor="black", fontsize=12)
    # add y-label on middle subplot
    ax[2].set_ylabel('Backazimuth [$^o$]', fontsize=12)
    # format x-axis on bottom subplot
    ax[4].set_xlabel("Local Time (US/Mountain) on 2023-10-06", fontsize=12)

    # set time axis limits
    time_min = datetime.datetime(year=2023, month=10, day=6, hour=8, minute=0, tzinfo=pytz.timezone("US/Mountain"))
    time_max = datetime.datetime(year=2023, month=10, day=6, hour=19, minute=0, tzinfo=pytz.timezone("US/Mountain"))
    ax[4].set_xlim([time_min, time_max])
    ax[4].xaxis.set_major_locator(mdates.HourLocator(byhour=range(24), interval=1, tz=pytz.timezone("US/Mountain")))
    ax[4].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=pytz.timezone("US/Mountain")))

    # FIGURE FORMATTING
    fig.suptitle(f"Known Helicopter Locations and Processed Backazimuth\nFiltered {freqmin}-{freqmax} Hz, 2023-10-06", 
                 fontsize=16)
    fig.tight_layout()
    # add colorbar across all subplots
    fig.subplots_adjust(right=0.87)
    cbar_ax = fig.add_axes([0.9, 0.17, 0.02, 0.6])
    cbar = fig.colorbar(im, cax=cbar_ax, aspect=12)
    cbar.set_label("Semblance", fontsize=12)

    # SAVE FIGURE
    plt.savefig(os.path.join(path_figures, f"backaz_allarrays_{freq_str}.png"), dpi=500)
    plt.close()
    return

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    # copied from a helpful stack overflow comment
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

if __name__ == "__main__":
    main()