#!/usr/bin/python3

# code to import helicopter data and overplot with infrasound data

import plot_utils, utils
import argparse, math, os, datetime, glob, pytz
import numpy as np
import pandas as pd
import scipy as sci
import obspy
from obspy.geodetics.base import gps2dist_azimuth
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from obspy.signal.array_analysis import get_geometry
from obspy.signal.util import util_geo_km, util_lon_lat
import xmltodict

def main(path_home):

    date_list = ["2023-10-6", "2023-10-7"]#, "2023-10-6"]#, "2023-10-5"]
    #array_str = "TOP"
    freqmin = 24.0
    freqmax = 32.0
    freq_str = f"{freqmin}_{freqmax}"

    def save_az_backaz():
        # save pkls of heli az and calculated backaz for each array for specified time interval
        array_list = ['TOP', 'JDNA', 'JDNB', 'JDSA', 'JDSB']
        for array_str in array_list:
            print(array_str)
            # load in helicopter data
            path_heli = os.path.join(path_home, "data", "helicopter")
            data_heli = adsb_kml_to_df(path_heli)

            # convert heli coords to dist/azimuth from array
            data_heli = helicoords_to_az(path_home, data_heli, array_str)

            # load processed infrasound data
            path_processed = os.path.join("/", "media", "mad", "LaCie 2 LT", "research", 
                                        "reynolds-creek", "data", "processed")
            output = pd.DataFrame()
            for date_str in date_list:
                file = os.path.join(path_processed, f"processed_output_{array_str}_{date_str}_{freq_str}.pkl")
                output_tmp = pd.read_pickle(file)
                output = pd.concat([output, output_tmp])
            
            filt = lambda arr: arr[(arr['Time'] > '2023-10-07T17:00:00') & (arr['Time'] < '2023-10-07T18:30:00')]
            output = filt(output).set_index('Time')
            data_heli = filt(data_heli).set_index('Time')

            # get rid of weird duplicates
            data_heli = data_heli[~data_heli.index.duplicated()]
            # resample to every 30 s (same rate as output for 60s windows with 50% overlap)
            win_len = 60
            overlap = 0.5
            spacing = int(win_len*overlap)
            heli_resamp = data_heli.resample(f'{spacing}s').mean().interpolate()

            # first time through, save heli lon/lat/alt as pkl
            if array_str == "TOP":
                print('saving heli coords')
                heli_resamp.to_pickle(os.path.join(path_home, 'data', 'test', 'heli_coords.pkl'))

            # concat with output to make df with 'true' dist/az and calc az
            all_data = pd.concat([heli_resamp['Distance'], heli_resamp['Azimuth'], output], axis=1)#.dropna() 
            all_data = all_data.rename(columns={'Distance': 'Heli Dist', 'Azimuth': 'Heli Az'})

            print('saving azs')
            all_data.to_pickle(os.path.join(path_home, 'data', 'test', f'{array_str}_az_backaz.pkl'))
        return
    
    # TEST data including "true" heli coords
    top = pd.read_pickle(os.path.join(path_home, 'data', 'test', 'TOP_az_backaz.pkl'))
    jdna = pd.read_pickle(os.path.join(path_home, 'data', 'test', 'JDNA_az_backaz.pkl'))
    jdnb = pd.read_pickle(os.path.join(path_home, 'data', 'test', 'JDNB_az_backaz.pkl'))
    jdsa = pd.read_pickle(os.path.join(path_home, 'data', 'test', 'JDSA_az_backaz.pkl'))
    jdsb = pd.read_pickle(os.path.join(path_home, 'data', 'test', 'JDSB_az_backaz.pkl'))



    # (1) get start coords and angles for each point from each array
    top_lat, top_lon, top_elv = station_coords_avg(path_home, 'TOP')
    jdna_lat, jdna_lon, jdna_elv = station_coords_avg(path_home, 'JDNA')
    jdnb_lat, jdnb_lon, jdnb_elv = station_coords_avg(path_home, 'JDNB')


    # these are line segments, not lines, since they only go in one direction
    # (2) so, calculate a reasonable end point for each line (max distance we'd accept intersection)


    # (3) calculate if the line segments intersect
    # if 4/5 intersect, that still seems valid.... think about this


    # (4) calculate least-squares best fit intersection point




    # let's try top and jdna first
    # set top as reference origin

    lat0, lon0, elv0 = station_coords_avg(path_home, 'TOP')
    lat1, lon1, elv1 = station_coords_avg(path_home, 'JDNA')

    # ref coordinates of TOP
    x0 = 0
    y0 = 0
    # coords of jdna rel to top
    x1, y1 = util_geo_km(lon0, lat0, lon1, lat1)

    int_pts = pd.DataFrame()    
    #FIXME cant use tan here these are backaz!!!!!!!!!!!!! NEED TO CONVERT 
    int_pts['m_top'] = np.tan((-1*top['Backaz']+90)%360)
    int_pts['b_top'] = y0 - int_pts['m_top']*x0

    int_pts['m_jdna'] = np.tan((-1*jdna['Backaz']+90)%360)
    int_pts['b_jdna'] = y1 - int_pts['m_jdna']*x1

    int_pts['x_int'] = (int_pts['b_jdna'] - int_pts['b_top']) / (int_pts['m_top'] / int_pts['m_jdna'])
    int_pts['y_int'] = int_pts['m_top'] * int_pts['x_int'] + int_pts['b_top']

    # remove huge outliers
    #int_pts = int_pts[(int_pts['x_int'] < int_pts['x_int'].quantile(0.90)) & 
                      #(int_pts['y_int'] < int_pts['y_int'].quantile(0.90))]
    
    # convert back to lat lon
    int_pts[['lon_int', 'lat_int']] = pd.DataFrame([util_lon_lat(lon0, lat0, x, y) for x, y in zip(int_pts['x_int'], int_pts['y_int'])], index=int_pts.index)

    # true coords
    heli_coords = pd.read_pickle(os.path.join(path_home, 'data', 'test', 'heli_coords.pkl'))


    # some test plots
    fig, ax = plt.subplots(2, 1, tight_layout=True, sharex=True)

    ax[0].plot(int_pts.index, int_pts['lon_int'], 'ro')
    ax[0].plot(heli_coords.index, heli_coords['Longitude'], 'k-')

    ax[1].plot(int_pts.index, int_pts['lat_int'], 'ro')
    ax[1].plot(heli_coords.index, heli_coords['Latitude'], 'k-')

    ax[0].set_ylim([-116.5, -117])
    ax[1].set_ylim([42.8, 43.4])

    plt.show()

    print(int_pts)


    






    geom = get_geometry(top, coordsys='lonlat', return_center=True)
    coords_rel = pd.DataFrame({'x': geom[:-1,0], 
                               'y': geom[:-1,1], 
                               'z': geom[:-1,2] + geom[-1,2]})
    fig, ax = plt.subplots(1, 1, tight_layout=True)
    ax.plot(coords_rel['x'], coords_rel['y'], 'ko')
    for i in range(len(coords_rel)):
        ax.text(x=coords_rel['x'][i], y=coords_rel['y'][i], 
                s=top[i].stats.station, ha='center', va='bottom')
    plt.axis('square')
    plt.show()
    plt.close() 

    #int_pts = pd.DataFrame()

    #lon_int = (d - c) / (a - b)

    return
    


def basic_main(path_home):
    date_list = ["2023-10-7"]#, "2023-10-6"]#, "2023-10-5"]
    array_str = "TOP"
    freqmin = 24.0
    freqmax = 32.0
    freq_str = f"{freqmin}_{freqmax}"

    # load in helicopter data
    path_heli = os.path.join(path_home, "data", "helicopter")
    data_heli = adsb_kml_to_df(path_heli)

    # convert heli coords to dist/azimuth from array
    data_heli = helicoords_to_az(path_home, data_heli, array_str)

    # load processed infrasound data
    # path to data on harddrive
    path_processed = os.path.join("/", "media", "mad", "LaCie 2 LT", "research", 
                                  "reynolds-creek", "data", "processed")
    #path_processed = os.path.join(path_home, "data", "processed", "window_60s")
    output = pd.DataFrame()
    for date_str in date_list:
        file = os.path.join(path_processed, f"processed_output_{array_str}_{date_str}_{freq_str}.pkl")
        output_tmp = pd.read_pickle(file)
        output = pd.concat([output, output_tmp])

    # plot processed data
    fig, ax = plot_utils.plot_backaz(output=output, path_home=path_home, 
                        subtitle_str=f"{array_str} Array, Filtered {freqmin} to {freqmax} Hz", file_str=None)
    
    # plot heli data
    ax.plot(data_heli['Time'], data_heli['Azimuth'], '-', color='green', 
            alpha=0.6, label='Helicopter Track')
    ax.legend(loc='upper right')

    ax.set_xlim([datetime.datetime(2023, 10, 7, 9, 0, 0, tzinfo=pytz.timezone("US/Mountain")), 
                 datetime.datetime(2023, 10, 7, 16, 0, 0, tzinfo=pytz.timezone("US/Mountain"))])

    # save figure
    plt.savefig(os.path.join(path_home, "figures", f"backaz_{array_str}_{freq_str}.png"), dpi=500)
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
    # TODO FIXME path
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
    #TODO FIXME path
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

    #main(path_home=path_home)
    basic_main(path_home=path_home)
