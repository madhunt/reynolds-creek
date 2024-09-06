#!/usr/bin/python3

import os, sys, datetime, pytz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.colors as colors

import matplotlib.dates as mdates

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import triangulate
import plot_utils

def main():
    # figure with all array backaz in subplots
    # set parameters (TODO change these to command line inputs)
    date_list = ["2023-10-7"]#, "2023-10-6"]#, "2023-10-5"]
    freqmin = 24.0
    freqmax = 32.0
    freq_str = f"{freqmin}_{freqmax}"

    array_list = ['TOP', 'JDNA', 'JDNB', 'JDSA', 'JDSB']

    # path to this file
    path_curr = os.path.dirname(os.path.realpath(__file__))
    path_home = os.path.abspath(os.path.join(path_curr, '..', '..'))




    path_heli = os.path.join(path_home, "data", "helicopter")
    # load processed infrasound data for all days of interest
    path_processed = os.path.join("/", "media", "mad", "LaCie 2 LT", "research", 
                                "reynolds-creek", "data", "processed")
    path_figures = os.path.join(path_home, "figures")
    plot_backaz_heli(path_home, path_heli, path_processed, 
                     path_figures, date_list, array_list, freq_str)




    return

def plot_backaz_heli(path_home, path_heli, path_processed, 
                     path_figures, date_list, array_list, freq_str):
    '''
    Plot calculated backazimuths overlaid with helicopter location data for 
    each array (TOP, JDNA, JDNB, JDSA, and JDSB). 
    INPUTS:

    RETURNS:
    '''

    subtitle_list = ["TOP (44 sensors)", "JDNA (4 sensors)", "JDNB (4 sensors)", 
                     "JDSA (4 sensors)", "JDSB (4 sensors)"]



    # load in helicopter data
    data = triangulate.adsb_kml_to_df(path_heli)

    fig, ax = plt.subplots(nrows=5, ncols=1, sharex=True, figsize=[14,9])#, tight_layout=True, figsize=[14,9])

    for i, array_str in enumerate(array_list):
        # convert heli coords to dist/azimuth from array
        data_heli = triangulate.helicoords_to_az(path_home, data, array_str)
        # mask data points at the end of a long data gap (for plotting)
        data_heli['Masked Azimuth'] = np.ma.masked_where(data_heli["Time"].diff() > datetime.timedelta(minutes=15), 
                                                         data_heli["Azimuth"])
        # set index as time
        data_heli = data_heli.set_index('Time')
        # remove any duplicate values
        data_heli = data_heli[~data_heli.index.duplicated(keep='first')]


        output = pd.DataFrame()
        for date_str in date_list:
            file = os.path.join(path_processed, f"processed_output_{array_str}_{date_str}_{freq_str}.pkl")
            output_tmp = pd.read_pickle(file)
            output = pd.concat([output, output_tmp])

        # plot processed data on array subplot
        #fig, axi = plot_utils.plot_backaz(output=output, path_home=path_home, 
        #                    subtitle_str=f"{array_str}", file_str=None,
        #                    fig=fig, ax=ax[i])

        axi = ax[i]

        # sort by ascending semblance so brightest points are plotted on top
        output = output.sort_values(by="Semblance", ascending=True)
        # constrain data to only plot points with slownesses near 3 s/km
        output = output[output["Slowness"].between(2.5, 3.5)]

        ## create new colormap
        def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
            new_cmap = colors.LinearSegmentedColormap.from_list(
                'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
                cmap(np.linspace(minval, maxval, n)))
            return new_cmap
        cmap = plt.get_cmap("Greys")
        new_cmap = truncate_colormap(cmap, 0.5, 1.0)


        im = axi.scatter(output["Time"], output['Backaz'], c=output["Semblance"],
                        alpha=1, edgecolors='none', cmap=new_cmap,
                        vmin=0, vmax=1)
                        #vmin=min(output["Semblance"]), vmax=max(output["Semblance"]))
        #cb = fig.colorbar(im, ax=axi)
        #cb.set_label("Semblance")

        # format y-axis
        #axi.set_ylabel("Backazimuth [$^o$]")
        axi.set_ylim([0, 360])
        axi.set_yticks(ticks=np.arange(0, 360+60, 90))

        # format x-axis
        axi.set_xlabel("Local Time (US/Mountain)")
        axi.set_xlim([output["Time"].min(), output["Time"].max()])
        hours_num = (output["Time"].max() - output["Time"].min()).total_seconds() / 3600
        tick_spacing = 1#int(np.ceil((hours_num / 15))) # make x-axis look nice (good number of ticks)
        axi.xaxis.set_major_locator(mdates.HourLocator(byhour=range(24), interval=tick_spacing))
        axi.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz="US/Mountain"))
        fig.autofmt_xdate()

        # add titles
        fig.suptitle(f"Backazimuth")
        axi.set_title(subtitle_list[i])
        #ax.set_title(subtitle_str, fontsize=10)





        # plot heli data on array subplot
        axi.plot(data_heli.index, data_heli['Masked Azimuth'], '-', color='red', 
                alpha=0.6, markeredgewidth=0.0, label='Helicopter Track')
        axi.legend(loc='upper right')
        axi.set_xlim([datetime.datetime(2023, 10, 7, 9, 0, 0, tzinfo=pytz.timezone("US/Mountain")), 
                    datetime.datetime(2023, 10, 7, 16, 0, 0, tzinfo=pytz.timezone("US/Mountain"))])
        # hide x-label for all but last plot
        if i < 4:
            axi.xaxis.label.set_visible(False)


    #cb = fig.colorbar(im, ax=axi)
    #cb.set_label("Semblance")
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, aspect=20)
    cbar.set_label("Semblance")

    # set common y-label
    fig.text(0.04, 0.5, 'Backazimuth [$^o$]', va='center', rotation='vertical')

    # set tight layout
    #fig.tight_layout()


    fig.suptitle(f"Backazimuth, Data Filtered {freq_str}")
    plt.show()
    print('done')
    # save figure
    #plt.savefig(os.path.join(path_figures, f"backaz_ALLARRAYS_{freq_str}.png"), dpi=500)
    #plt.close()

    return

if __name__ == "__main__":
    main()