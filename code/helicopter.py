#!/usr/bin/python3

# code to import helicopter data and overplot with infrasound data

import plot_utils, utils
import argparse, math, os, datetime, glob
import numpy as np
import pandas as pd
import obspy
from obspy.geodetics.base import gps2dist_azimuth
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xmltodict

def main(path_home):

    freqmin = 2.0
    freqmax = 5.0
    freq_str = f"{freqmin}_{freqmax}"

    # load in helicopter data
    path_heli = os.path.join(path_home, "data", "helicopter")
    data_heli = adsb_kml_to_df(path_heli)

    # convert heli coords to dist/azimuth from array
    data_heli = helicoords_to_az(path_home, data_heli)

    # load processed infrasound data
    path_processed = os.path.join(path_home, "data", "processed",
                    "window_60s", f"processed_output_{freq_str}.pkl")
    output = pd.read_pickle(path_processed)

    # plot processed data
    fig, ax = plot_backaz(output=output, path_home=path_home, 
                          subtitle_str=f"Filtered {freqmin} to {freqmax} Hz", file_str=None)
    
    # plot heli data
    ax.plot(data_heli['Time'], data_heli['Azimuth'], '.', color='green', 
            alpha=0.5, label='Helicopter Track')
    ax.legend(loc='lower left')

    # save figure
    plt.savefig(os.path.join(path_home, "figures", f"backaz_{freq_str}.png"), dpi=500)
    plt.close()



    plt.show()



    # plot station map
    
    
    return

def plot_backaz(output, path_home, subtitle_str, file_str=None):
    '''
    Plot backazimuth over time from output of array processing. 
    INPUTS:
        output : pandas df : Result from beamforming with the columns 
            Time (datetime), Semblance, Abs Power, Backaz (0-360), and Slowness.
        path_home : str : Path to main dir. Figure will be saved in "figures" subdir.
        subtitle_str : str : Subtitle for plot. Usually contains bandpass frequencies 
            (e.g. "Filtered 24-32 Hz")
        file_str : str or None : String to append on end of filename to uniquely save figure 
            (e.g. "24.0_32.0"). If None, function returns a handle to the figure and axes, and does 
            NOT save the figure. 
    RETURNS:
        If file_str=None, returns handle to the figure and axes. Figure is NOT saved.
        Otherwise, figure is saved as path_home/figures/backaz_{file_str}.png
    '''
    # sort by ascending semblance so brightest points are plotted on top
    output = output.sort_values(by="Semblance", ascending=True)

    # constrain data to only plot points with slownesses near 3 s/km
    slow_min = 2.5
    slow_max = 3.5
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
    ax.set_xlim([output["Time"].min(), output["Time"].max()])
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(24), interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M", tz="US/Mountain"))
    fig.autofmt_xdate()

    # add titles
    fig.suptitle(f"Backazimuth")
    ax.set_title(subtitle_str, fontsize=10)

    if file_str == None:
        return fig, ax
    else: 
        # save figure
        plt.savefig(os.path.join(path_home, "figures", f"backaz_{file_str}.png"), dpi=500)
        plt.close()
        return


def adsb_kml_to_df(path):
    '''
    Loads in aircraft flight track (downloaded from ADS-B Exchange https://globe.adsbexchange.com/) 
    from KML as a pandas dataframe.  
    INPUTS: 
        path : str : Path to dir containing all KML files for one aircraft of interest.
    RETURNS: 
        data_all : pandas df : Dataframe containing data from all KML files in specified dir. Columns are 
                    Time : datetime : Timestamp of location reading
                    Latitude : float : Latitude of aircraft
                    Longitude : float : Longitude of aircraft
                    Altitude : float : Altitude of aircraft
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
        data[['Longitude', 'Latitude', 'Altitude']] = data['coord'].str.split(' ', 
                                                                              n=2, expand=True).astype(float)
        data = data.drop('coord', axis=1)

        # store data from multiple files
        data_all = pd.concat([data_all, data])
    
    # do some cleanup
    data_all = data_all.sort_values("Time").drop_duplicates()


    return data_all
    

def station_coords_avg(path_home):
    '''
    Find mean location for entire array. 
    INPUTS: 
        path_home : str : Path to main dir.
    RETURNS:
        lat : float : Average latitude for station.
        lon : float : Average longitude for station.
        elv : float : Average elevation for station.
    '''
    path_data = os.path.join(path_home, "data", "mseed")
    path_coords = glob.glob(os.path.join(path_data, "..", "gps", "*.csv" ))
    coords = pd.DataFrame()
    for file in path_coords:
        coords = pd.concat([coords, pd.read_csv(file)])
    coords["Name"] = coords["Name"].astype(str) # SN of gem

    lat = coords['Latitude'].mean()
    lon = coords['Longitude'].mean()
    elv = coords['Elevation'].mean()

    return lat, lon, elv


def helicoords_to_az(path_home, data):

    # find avg location for entire TOP array 
    coords_top = station_coords_avg(path_home)

    # use gps2dist to get distance and azimuth between heli and TOP array
    data[['Distance', 'Azimuth', 'az2']] = data.apply(lambda x: 
                                                                gps2dist_azimuth(lat1=coords_top[0], lon1=coords_top[1], 
                                                                                lat2=x["Latitude"], lon2=x["Longitude"]), 
                                                                                axis=1, result_type='expand')
    data = data.drop('az2', axis=1)

    return data
    

if __name__ == "__main__":
    
    # path to this file
    path_curr = os.path.dirname(os.path.realpath(__file__))
    path_home = os.path.abspath(os.path.join(path_curr, '..'))

    main(path_home=path_home)
