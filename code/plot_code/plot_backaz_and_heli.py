#!/usr/bin/python3
'''
Plot backazimuth over time for all arrays. 
Backaz point colors are greyscale semblance value. 
Helicopter location relative to array is overplotted as red line.
'''
import os, sys, datetime, pytz
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# import files from dir above
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils, settings

def main(path_processed, path_heli, path_station_gps, path_figures,
         t0, tf, freqmin, freqmax):
    '''
    Plot calculated backazimuths overlaid with helicopter location data for 
        each array (TOP, JDNA, JDNB, JDSA, and JDSB). 
    INPUTS
        path_XXX    : str       : Paths from settings.py
        t0          : datetime  : Datetime object with UTC timezone for figure
            lower x-axis limit.
        tf          : datetime  : Datetime object with UTC timezone for figure
            upper x-axis limit.
    RETURNS
        Figure saved at path_figures/backaz_and_heli_{freqmin}-{freqmax}Hz_{timestr(t0)}_{timestr(tf)}.png
    '''
    array_list = ["TOP", "JDNA", "JDNB", "JDSA", "JDSB"]
    subtitle_list = ["TOP (42 sensors)", "JDNA (3 sensors)", "JDNB (3 sensors)", 
                     "JDSA (4 sensors)", "JDSB (4 sensors)"]

    fig, ax = plt.subplots(nrows=5, ncols=1, sharex=True, figsize=[12,9])
    freq_str = f"{freqmin}_{freqmax}"

    for i, array_str in enumerate(array_list):
        # LOAD PROCESSED BACKAZIMUTH DATA
        output = utils.load_backaz_data(path_processed, array_str, freq_str, t0, tf)
        # sort by ascending semblance so brightest points are plotted on top
        output = output.sort_values(by="Semblance", ascending=True)
        # constrain data to only plot points with slownesses near 3 s/km
        

        #NOTE SLOWNESS FILTER HERE
        output = output[output["Slowness"].between(2, 3.5)]






        # PLOT BACKAZIMUTHS
        # create truncated greyscale colormap
        new_cmap = utils.truncate_colormap(plt.get_cmap("Greys"), 0.4, 1.0)
        im = ax[i].scatter(output.index, output['Backaz'], c=output["Semblance"],
                        alpha=1, edgecolors='none', cmap=new_cmap,
                        vmin=0, vmax=1)
        # format subplot y-axis
        ax[i].set_ylim([0, 360])
        ax[i].set_yticks(ticks=np.arange(0, 360+60, 90))

        # LOAD HELICOPTER TRACK DATA as lat/lons
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
    datestr = lambda t: t.strftime("%Y-%m-%d")
    ax[4].set_xlabel(f"Local Time (UTC-6:00) on {datestr(t0)}", fontsize=12)

    # set time axis limits
    #time_min = datetime.datetime(year=2023, month=10, day=6, hour=8, minute=0, tzinfo=pytz.timezone("US/Mountain"))
    #time_max = datetime.datetime(year=2023, month=10, day=6, hour=19, minute=0, tzinfo=pytz.timezone("US/Mountain"))
    #ax[4].set_xlim([time_min, time_max])
    ax[4].set_xlim([t0, tf])
    ax[4].xaxis.set_major_locator(mdates.HourLocator(byhour=range(24), interval=1, tz=pytz.timezone("US/Mountain")))
    ax[4].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=pytz.timezone("US/Mountain")))

    # FIGURE FORMATTING
    fig.suptitle(f"Known Helicopter Locations and Processed Backazimuth\nFiltered {freqmin}-{freqmax} Hz, {datestr(t0)}", 
                 fontsize=16)
    fig.tight_layout()
    # add colorbar across all subplots
    fig.subplots_adjust(right=0.87)
    cbar_ax = fig.add_axes([0.9, 0.17, 0.02, 0.6])
    cbar = fig.colorbar(im, cax=cbar_ax, aspect=12)
    cbar.set_label("Semblance", fontsize=12)

    # SAVE FIGURE
    timestr = lambda t: t.strftime("%Y%m%d-%H-%M")
    filename = f"backaz_and_heli_{freqmin}-{freqmax}Hz_{timestr(t0)}_{timestr(tf)}.png"
    plt.savefig(os.path.join(path_figures, filename), dpi=500)
    plt.close()
    return


if __name__ == "__main__":
    # settings for AGU figure 2-8 Hz -------------------------------
    #freqmin = 2.0
    #freqmax = 8.0
    ## these UTC times give Mountain 09:00-20:00
    #t0 = datetime.datetime(2023, 10, 6, 15, 0, 0, tzinfo=pytz.UTC)
    #tf = datetime.datetime(2023, 10, 7, 2, 0, 0, tzinfo=pytz.UTC)

    # settings for AGU figure 24-23 Hz -----------------------------
    #freqmin = 24.0
    #freqmax = 32.0
    ## these UTC times give Mountain 10:00-15:00
    #t0 = datetime.datetime(2023, 10, 7, 16, 0, 0, tzinfo=pytz.UTC)
    #tf = datetime.datetime(2023, 10, 7, 21, 0, 0, tzinfo=pytz.UTC)



    freqmin = 24.0
    freqmax = 32.0
    t0 = datetime.datetime(2023, 10, 7, 16, 0, 0, tzinfo=pytz.UTC)
    tf = datetime.datetime(2023, 10, 7, 21, 0, 0, tzinfo=pytz.UTC)


    
    main(settings.path_processed, 
         settings.path_heli, 
         settings.path_station_gps, 
         settings.path_figures,
         t0, tf, freqmin, freqmax)
