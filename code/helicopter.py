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

    date_list = ["2023-10-7", "2023-10-8", "2023-10-6"]#, "2023-10-5"]
    array_str = "JD"
    freqmin = 4.0
    freqmax = 8.0
    freq_str = f"{freqmin}_{freqmax}"

    # load in helicopter data
    path_heli = os.path.join(path_home, "data", "helicopter")
    data_heli = adsb_kml_to_df(path_heli)

    # convert heli coords to dist/azimuth from array
    data_heli = helicoords_to_az(path_home, data_heli, array_str)

    # load processed infrasound data
    # path to data on harddrive
    path_processed = os.path.join("/", "media", "mad", "LaCie 2 LT", "research", 
                                  "reynolds-creek", "results_processed", "data")
    #path_processed = os.path.join(path_home, "data", "processed", "window_60s")
    output = pd.DataFrame()
    for date_str in date_list:
        file = os.path.join(path_processed, f"processed_output_{array_str}_{date_str}_{freq_str}.pkl")
        output_tmp = pd.read_pickle(file)
        output = pd.concat([output, output_tmp])

    # plot processed data
    fig, ax = plot_utils.plot_backaz(output=output, path_home=path_home, 
                          subtitle_str=f"Filtered {freqmin} to {freqmax} Hz", file_str=None)
    
    # plot heli data
    ax.plot(data_heli['Time'], data_heli['Azimuth'], '.', color='green', 
            alpha=0.1, label='Helicopter Track')
    ax.legend(loc='lower left')

    # save figure
    plt.savefig(os.path.join(path_home, "figures", f"backaz_{array_str}_{freq_str}.png"), dpi=500)
    plt.close()



    # plot station map
    
    
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
    

def station_coords_avg(path_home, array_str):
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
    # filter by array
    coords = coords[coords["Station"].str.contains(array_str)]

    lat = coords['Latitude'].mean()
    lon = coords['Longitude'].mean()
    elv = coords['Elevation'].mean()

    return lat, lon, elv


def helicoords_to_az(path_home, data, array_str):

    # find avg location for entire array specified
    coords_top = station_coords_avg(path_home, array_str)

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
