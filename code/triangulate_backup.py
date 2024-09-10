#!/usr/bin/python3

import plot_utils
import os, datetime, glob, pytz, itertools, xmltodict
import numpy as np
import pandas as pd
import scipy as sci
from obspy.geodetics.base import gps2dist_azimuth
from obspy.signal.util import util_geo_km, util_lon_lat
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

def main(path_home):

    # set parameters (TODO change these to command line inputs)
    date_list = ["2023-10-7"]#, "2023-10-6"]#, "2023-10-5"]
    freqmin = 24.0
    freqmax = 32.0
    freq_str = f"{freqmin}_{freqmax}"

    array_list = ['TOP', 'JDNA', 'JDNB', 'JDSA', 'JDSB']

    

    # initialize empty dfs
    all_ints = pd.DataFrame()
    all_rays = pd.DataFrame()

    # calculate intersections at each timestep for each pair of arrays
    for i, (arr1_str, arr2_str) in enumerate(itertools.combinations(array_list, 2)):
        # load in processed data and calculate center point of array (lat, lon)
        arr1, p1_latlon = load_ray_data(arr1_str, date_list, freq_str)
        arr2, p2_latlon = load_ray_data(arr2_str, date_list, freq_str)

        # change from lat,lon to x,y in km
        p1_km = np.array([0, 0])
        p2_km = util_geo_km(orig_lon=p1_latlon[1], orig_lat=p1_latlon[0], lon=p2_latlon[1], lat=p2_latlon[0])

        # calculate intersection
        int_pts_km = arr1['Backaz'].combine(arr2['Backaz'], 
                                           (lambda a1, a2: intersection(p1_km, a1, p2_km, a2)))
        
        # change back to lat/lon from km
        int_pts = pd.DataFrame([util_lon_lat(orig_lon=p1_latlon[1], orig_lat=p1_latlon[0], 
                                             x=km[0], y=km[1]) for km in int_pts_km], 
                               columns=['Lon', 'Lat'], index=int_pts_km.index)
        
        # NOTE TO MAD TOMORROW: make a plot here to visulaize km vs lat/lon intersection **************
        #******

        # save intersection points
        all_ints.index = int_pts.index
        all_ints[f'{arr1_str}_{arr2_str}_Lat'] = int_pts['Lat']#[col[1] for col in int_pts]
        all_ints[f'{arr1_str}_{arr2_str}_Lon'] = int_pts['Lon']#[col[0] for col in int_pts]

        # save rays
        all_rays.index = arr1.index
        all_rays[arr1_str] = arr1['Backaz']
        all_rays[arr2_str] = arr2['Backaz']

    # now find median intersection point
    median_ints = pd.DataFrame(index=int_pts.index)
    median_ints['Lat'] = all_ints.filter(regex='Lat').median(axis=1)
    median_ints['Lon'] = all_ints.filter(regex='Lon').median(axis=1)


    # FIXME? get rid of outliers
    #median_ints = median_ints[~((median_ints['Lat'] > median_ints['Lat'].quantile(0.99)) | 
    #                            (median_ints['Lat'] < median_ints['Lat'].quantile(0.01)) | 
    #                            (median_ints['Lon'] > median_ints['Lon'].quantile(0.99)) | 
    #                            (median_ints['Lon'] < median_ints['Lon'].quantile(0.01)) )]

    # load in true helicopter data
    path_heli = os.path.join(path_home, "data", "helicopter")
    data_heli = adsb_kml_to_df(path_heli)
    data_heli = data_heli.set_index('Time')
    # resample heli data
    data_heli = data_heli[~data_heli.index.duplicated(keep='first')]
    data_heli = data_heli.resample('30s').nearest()
    #FIXME change from nearest to something else??

    # only use a subsection of points between times of interest
    #filt = lambda arr: arr[(arr.index > '2023-10-07T17:45:00') & (arr.index < '2023-10-07T18:00:00')]
    #median_ints = filt(median_ints)
    #data_heli = filt(data_heli)

    # plot all rays
    for i, row in enumerate(all_rays.iterrows()):

        fig, ax = plt.subplots(1, 1, tight_layout=True)

        colors = cm.winter(np.linspace(0, 1, 5))
        for j, arr_str in enumerate(array_list):
            # get initial point, slope, and y-intercept of ray
            p = station_coords_avg(path_home, arr_str)
            m = np.tan(np.deg2rad(row[1][arr_str]))
            #NOTE THIS IS NOT THE SLOPE IN AZIMUTHAL COORDS***
            b = p[0] - p[1]*m
            # only plot positive end of the ray (arbitrary length 100)
            ax.plot([p[1],  100], [p[0], 100*m + b],
                     color=colors[j])
            
            # plot array centers as triangles
            ax.scatter(p[1], p[0], c='k', marker='^')
            ax.text(p[1], p[0]-0.0004, s=arr_str,
                    va='center', ha='center')
            
        #FIXME for testing purposes plot all ints
        for k in np.arange(0, 20, 2):
            ax.scatter(all_ints.loc[row[0]][k+1], all_ints.loc[row[0]][k],
                       c='green', marker='o')
        #FIXME this is not right....


        # plot median point
        med_lat = median_ints.loc[row[0]]['Lat']
        med_lon = median_ints.loc[row[0]]['Lon']
        ax.plot(med_lon, med_lat, c='orange', marker='*', 
                   markersize=10, label='Median Intersection')

        # plot helicopter "true" point
        heli_lat = data_heli.loc[row[0]]['Latitude']
        heli_lon = data_heli.loc[row[0]]['Longitude']
        ax.plot(heli_lon, heli_lat, c='red', marker='*', 
                   markersize=10, label='True Heli Location')

        ax.legend(loc='upper left')
        ax.set_xlim([-116.85, -116.74])
        ax.set_ylim([43.10, 43.18])

        plt.suptitle(row[0])
        plt.show()



        ##TODO
        # plot waveforms / pwr for each array (median?)
        # waveform for TOP? 


        print(os.path.join(path_home, "figures", "backaz_errors", f"{row[0]}timestep.png"))
        plt.savefig(os.path.join(path_home, "figures", "backaz_errors", f"{row[0]}timestep.png"), dpi=500)
        plt.close()
        #plt.show()
    # loop through time and plot backaz rays at each time
   # print('plot time')
    # also get waveforms


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

def load_ray_data(array_str, date_list, freq_str):
    # load processed infrasound data
    path_processed = os.path.join("/", "media", "mad", "LaCie 2 LT", "research", 
                                "reynolds-creek", "data", "processed")
    output = pd.DataFrame()
    for date_str in date_list:
        file = os.path.join(path_processed, f"processed_output_{array_str}_{date_str}_{freq_str}.pkl")
        output_tmp = pd.read_pickle(file)
        output = pd.concat([output, output_tmp])
    
    output = output.set_index('Time')
    
    #filt = lambda arr: arr[(arr['Time'] > '2023-10-07T17:00:00') & (arr['Time'] < '2023-10-07T18:30:00')]
    #output = filt(output).set_index('Time')

    # add column for angle (0 deg at x-axis, 90 deg at y-axis)
    #output['Angle'] = (-1 * output['Backaz'] + 90)%360

    # get start points
    lat, lon, elv = station_coords_avg(path_home, array_str)
    p_start = np.array([lat, lon])

    return output, p_start

def intersection(p1, a1, p2, a2):
    '''
    Calculates intersection point between two rays, given starting points and azimuths (from N).
    INPUTS
        p1 : np array, 2x1 : Coordinates (x,y) for start point of ray 1.
        a1 : float : Azimuth of ray 1 direction in degrees (clockwise from North).
        p2 : np array, 2x1 : Coordinates (x,y) for start point of ray 2.
        a2 : float : Azimuth of ray 2 direction in degrees (clockwise from North).
    RETURNS
        int_pt : np array, 2x1 : Coordinates (x,y) of intersection point. 
            Returns [NaN, NaN] if there is no intersection; e.g. if intersection 
            occurs "behind" ray start points or if rays are parallel. 
    '''
    # create matrix of direction unit vectors
    D = np.array([[np.sin(np.radians(a1)), -np.sin(np.radians(a2))],
                      [np.cos(np.radians(a1)), -np.cos(np.radians(a2))]])
    # create vector of difference in start coords (p2x-p1x, p2y-p1y)
    P = np.array([p2[0] - p1[0],
                  p2[1] - p1[1]])

    # solve system of equations Dt=P
    try:
        t = np.linalg.solve(D, P)
    except:
        # matrix is singular (rays are parallel)
        return np.array([np.nan, np.nan])

    # see if intersection point is actually along rays
    if t[0]<0 or t[1]<0:
        # if intersection is "behind" rays, return nans
        return np.array([np.nan, np.nan])
        
    # calculate intersection point
    int_pt = np.array([p1[0]+D[0,0]*t[0],
                       p1[1]+D[1,0]*t[0]])
    return int_pt

if __name__ == "__main__":
    
    # path to this file
    path_curr = os.path.dirname(os.path.realpath(__file__))
    path_home = os.path.abspath(os.path.join(path_curr, '..'))

    main(path_home=path_home)
    #basic_main(path_home=path_home)
