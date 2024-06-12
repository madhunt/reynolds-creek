#!/usr/bin/python3

# code to import helicopter data and overplot with infrasound data

import plot_utils, utils
import argparse, math, os, datetime, glob
import numpy as np
import pandas as pd
import obspy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xmltodict

def main(path_home):

    # load helicopter data
    path_heli = os.path.join(path_home, "data", "helicopter")
    
    data_heli = adsbkml_to_df(path_heli)





    # load processed data
    path_processed = os.path.join(path_home, "data", "processed",
                    "window_60s", "processed_output_24.0_32.0.pkl")
    output = pd.read_pickle(path_processed)

    # plot processed data
    plot_backaz(output, path_home, "Filtered 24-32 Hz", "24_32")



    # plot station map
    
    
    return

def plot_backaz(output, path_home, subtitle_str, file_str):
    '''
    Plot backazimuth over time from output of array processing. 
    INPUTS:
        output : pandas df : Result from beamforming with the columns 
            Time (datetime), Semblance, Abs Power, Backaz (0-360), and Slowness.
        path_home : str : Path to main dir. Figure will be saved in "figures" subdir.
        subtitle_str : str : Subtitle for plot. Usually contains bandpass frequencies 
            (e.g. "Filtered 24-32 Hz")
        file_str : str : String to append on end of filename to uniquely label figure 
            (e.g. "24.0_32.0").
    RETURNS:

    '''
    # sort by ascending semblance so brightest points are plotted on top
    output = output.sort_values(by="Semblance", ascending=True)

    # constrain data to only plot points with slownesses near 3 s/km
    slow_min = 2.7
    slow_max = 3.3
    output = output[output["Slowness"].between(slow_min, slow_max)]

    # create figure
    fig, ax = plt.subplots(1, 1, figsize=[7, 5], tight_layout=True)
    im = ax.scatter(output["Time"], output['Backaz'], c=output["Semblance"],
                    alpha=0.7, edgecolors='none', cmap='plasma',
                    vmin=min(output["Semblance"]), vmax=max(output["Semblance"]))
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Semblance")

    # format y-axis
    ax.set_ylabel("Backazimuth [$^o$]")
    ax.set_ylim([0, 360])
    ax.set_yticks(ticks=np.arange(0, 360+60, 60))

    # format x-axis
    ax.set_xlabel("Mountain Time (Local)")
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(24), interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M", tz="US/Mountain"))
    fig.autofmt_xdate()

    # add titles
    fig.suptitle(f"Backazimuth")
    ax.set_title(subtitle_str, fontsize=10)

    # save figure
    plt.savefig(os.path.join(path_home, "figures", f"backaz_{file_str}.png"), dpi=500)
    plt.close()
    return


def adsbkml_to_df(path):
    '''
    Loads in aircraft flight track (downloaded from ADS-B Exchange https://globe.adsbexchange.com/) 
    from KML as a pandas dataframe.
    INPUTS:
        path : str : Path to dir containing all KML files for one aircraft of interest.
    RETURNS:
        data_all : pandas df : Dataframe containing data from all KML files in specified dir. Columns are 
                    Time : datetime : Timestamp of location reading
                    Lat : float : Latitude of aircraft
                    Lon : float : Longitude of aircraft
                    Alt : float : Altitude of aircraft
    '''

    files_kml = glob.glob(os.path.join(path, "*.kml" ))
    data_all = pd.DataFrame()

    for file in files_kml:
        data = pd.DataFrame()
        with open(file, 'r') as file:
            xml_str = file.read()
        xml_dict = xmltodict.parse(xml_str)
        data_raw = xml_dict['kml']['Folder']['Folder']['Placemark']['gx:Track']

        # add data to pandas array 
        data['Time'] = pd.to_datetime(data_raw['when'])
        data['coord'] = data_raw['gx:coord']
        data[['Lat', 'Lon', 'Alt']] = data['coord'].str.split(' ', n=2, expand=True).astype(float)
        data = data.drop('coord', axis=1)

        # store data from multiple files
        data_all = pd.concat([data_all, data])

    return data_all

if __name__ == "__main__":
    
    # path to this file
    path_curr = os.path.dirname(os.path.realpath(__file__))
    path_home = os.path.abspath(os.path.join(path_curr, '..'))

    main(path_home=path_home)
