#!/usr/bin/python3
'''
Crossbeams to find intersection points between all five arrays (TOP, JD*). 
Finds median intersection point (in UTM), 
and number of total intersections/crossbeams at each timestep.
Re-samples "true" helicopter data during same time period.
Calculates distance between true helicopter point and median intersection.
'''
import os, datetime, pytz, itertools, utm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

import utils, settings

def main(path_processed, path_heli, path_station_gps, path_output,
         t0, tf, freqmin, freqmax):

    # calculate all intersections for each pair of arrays
    array_list = ['TOP', 'JDNA', 'JDNB', 'JDSA', 'JDSB']
    all_ints, all_rays = calc_array_intersections(path_processed, path_station_gps, array_list,
                                                  t0, tf, freqmin, freqmax)
    
    # calculate median intersection points
    median_ints = pd.DataFrame(index=all_ints.index)
    median_ints['Easting'] = all_ints.filter(regex='Easting').median(axis=1)
    median_ints['Northing'] = all_ints.filter(regex='Northing').median(axis=1)

    # calculate number of intersections
    median_ints['Num Int'] = all_ints.notna().sum(axis=1)/2
    median_ints['Num TOP Int'] = all_ints[all_ints.columns[all_ints.columns.str.contains("TOP")]].notna().sum(axis=1) / 2
    # find all columns WITHOUT "TOP", sum across rows where values are NOT nan, divide by 2 (lat and lon)
    median_ints['Num JD Int'] = all_ints[all_ints.columns[~all_ints.columns.str.contains("TOP")]].notna().sum(axis=1) / 2

    # load in true helicopter data as UTM coordinates
    data_heli = utils.adsb_kml_to_df(path_heli, latlon=False)
    # resample heli data
    filt = lambda arr: arr[(arr['Time'] > t0) & (arr['Time'] < tf)]
    data_heli = filt(data_heli ).set_index('Time')
    data_heli = data_heli[~data_heli.index.duplicated(keep='first')]
    data_heli = data_heli.resample('30s').nearest()

    # add true data to median_ints dataframe
    median_ints['True Easting'] = data_heli['Easting']
    median_ints['True Northing'] = data_heli['Northing']

    # calculate distance between median int and heli point (in m)
    median_ints['Distance'] = np.sqrt((median_ints['True Easting'] - median_ints['Easting'])**2 +
                                      (median_ints['True Northing'] - median_ints['Northing'])**2)
    
    # add column for plotting points through time in QGIS
    median_ints['Time Int'] = median_ints.index.astype(int)

    # save median_ints dataframe (to make figures in another file)
    if path_output != None:
        timestr = lambda t: t.strftime("%Y%m%d-%H-%M")
        filename = f"median_intersections_{freqmin}-{freqmax}Hz_{timestr(t0)}_{timestr(tf)}.csv"
        median_ints.to_csv(os.path.join(path_output, filename))
    return median_ints, all_ints, all_rays


def intersection(p1, a1, p2, a2, latlon=False):
    '''
    Calculates intersection point between two rays, given starting points and azimuths (from N).
    INPUTS
        p1      : np array, 2x1 : Coordinates for start point of ray 1.
        a1      : float         : Azimuth of ray 1 direction in degrees (clockwise from North).
        p2      : np array, 2x1 : Coordinates for start point of ray 2.
        a2      : float         : Azimuth of ray 2 direction in degrees (clockwise from North).
        latlon  : bool          : Default False (coordinates are of the form x,y).
            If True, coordinates are provided as lat,lon (y,x).
    RETURNS
        int_pt : np array, 2x1 : Coordinates of intersection point. 
            If latlon=False, coordinates are returned as x,y. 
            If latlon=True, coordinates are returned as lat,lon (y,x).
            Returns [NaN, NaN] if there is no intersection; e.g. if intersection 
            occurs "behind" ray start points or if rays are parallel. 
    '''
    if latlon == True:
        # swap the order of coordinates to lon,lat
        p1 = np.array([p1[1], p1[0]])
        p2 = np.array([p2[1], p2[0]])
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
    if latlon == True: 
        # return intersection as lat,lon
        int_pt = np.array([int_pt[1], int_pt[0]])
    return int_pt


def calc_array_intersections(path_processed, path_station_gps, array_list, t0, tf, freqmin, freqmax):
    # initialize empty dfs
    all_ints = pd.DataFrame()
    all_rays = pd.DataFrame()

    # calculate intersections at each timestep for each pair of arrays
    for i, (arr1_str, arr2_str) in enumerate(itertools.combinations(array_list, 2)):
        # load in processed data
        freq_str = f"{freqmin}_{freqmax}"
        arr1 = utils.load_backaz_data(path_processed, arr1_str, freq_str, t0, tf)
        arr2 = utils.load_backaz_data(path_processed, arr2_str, freq_str, t0, tf)
        # calculate center point of arrays (lat, lon)
        p1 = utils.station_coords_avg(path_station_gps, arr1_str, latlon=False)
        p2 = utils.station_coords_avg(path_station_gps, arr2_str, latlon=False)

        # filter by slowness (only keep backaz between reasonable slowness values)
        
        
        #NOTE SLOWNESS FILTER HERE
        slow_filt =  lambda arr: arr[arr['Slowness'].between(2.0, 3.5)]
        arr1 = slow_filt(arr1)
        arr2 = slow_filt(arr2)

        # save rays (for plotting)
        all_rays[arr1_str] = arr1['Backaz']
        all_rays[arr2_str] = arr2['Backaz']

        # calculate intersection
        int_pts = pd.DataFrame()
        int_pts['result'] = arr1['Backaz'].combine(arr2['Backaz'], 
                                           (lambda a1, a2: intersection(p1, a1, p2, a2, latlon=False)))
        int_pts['Easting'] = [x[0] for x in int_pts['result']]
        int_pts['Northing'] = [x[1] for x in int_pts['result']]
        int_pts = int_pts.drop('result', axis=1)      # clean up temp column
        
        # convert points from lat/lon to UTM
        #int_pts_notna = int_pts[int_pts['Lat'].notna()]             # remove nans
        #int_pts_notna = int_pts_notna[int_pts_notna['Lat'] <= 80]   # remove outliers
        #int_pts_utm = utm.from_latlon(int_pts_notna['Lat'].to_numpy(), int_pts_notna['Lon'].to_numpy())
        ## match points with time index in df
        #int_pts_notna['Easting'] = int_pts_utm[0]
        #int_pts_notna['Northing'] = int_pts_utm[1]
        # add back to df that includes NaN values
        #int_pts['Easting'] = int_pts_notna['Easting']
        #int_pts['Northing'] = int_pts_notna['Northing']

        # save intersection points in larger df
        all_ints[f'{arr1_str}_{arr2_str}_Easting'] = int_pts['Easting']
        all_ints[f'{arr1_str}_{arr2_str}_Northing'] = int_pts['Northing']

    return all_ints, all_rays



if __name__ == "__main__":
    # choose times when helicopter is moving so we can compare points
    freqmin = 4.0
    freqmax = 8.0
    #freqmin = 24.0
    #freqmax = 32.0

    # TIMES FOR 07 OCT
    #t0 = datetime.datetime(2023, 10, 7, 16, 0, 0, tzinfo=pytz.UTC)
    #tf = datetime.datetime(2023, 10, 7, 21, 0, 0, tzinfo=pytz.UTC)
    # TIMES FOR 06 OCT
    #t0 = datetime.datetime(2023, 10, 6, 18, 0, 0, tzinfo=pytz.UTC)
    #tf = datetime.datetime(2023, 10, 6, 23, 0, 0, tzinfo=pytz.UTC)
    # times determined from best signal
    t0 = datetime.datetime(2023, 10, 6, 20, 0, 0, tzinfo=pytz.UTC)
    tf = datetime.datetime(2023, 10, 6, 22, 30, 0, tzinfo=pytz.UTC)


    main(settings.path_processed, settings.path_heli, 
         settings.path_station_gps, settings.path_output,
         t0, tf, freqmin, freqmax)
