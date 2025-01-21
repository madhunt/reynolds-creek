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

    # save median_ints dataframe (to make figures in another file)
    timestr = lambda t: t.strftime("%Y%m%d-%H-%M")
    filename = f"median_intersections_{freqmin}-{freqmax}Hz_{timestr(t0)}_{timestr(tf)}.csv"
    median_ints.to_csv(os.path.join(path_output, filename))
    return




def intersection(p1, a1, p2, a2, latlon=True):
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
        p1 = utils.station_coords_avg(path_station_gps, arr1_str, latlon=True)
        p2 = utils.station_coords_avg(path_station_gps, arr2_str, latlon=True)

        # save rays (for plotting)
        all_rays[arr1_str] = arr1['Backaz']
        all_rays[arr2_str] = arr2['Backaz']

        # calculate intersection
        int_pts = pd.DataFrame()
        int_pts['result'] = arr1['Backaz'].combine(arr2['Backaz'], 
                                           (lambda a1, a2: intersection(p1, a1, p2, a2, latlon=True)))
        int_pts['Lat'] = [x[0] for x in int_pts['result']]
        int_pts['Lon'] = [x[1] for x in int_pts['result']]
        int_pts = int_pts.drop('result', axis=1)      # clean up temp column
        
        # convert points from lat/lon to UTM
        int_pts_notna = int_pts[int_pts['Lat'].notna()]             # remove nans
        int_pts_notna = int_pts_notna[int_pts_notna['Lat'] <= 80]   # remove outliers
        int_pts_utm = utm.from_latlon(int_pts_notna['Lat'].to_numpy(), int_pts_notna['Lon'].to_numpy())
        # match points with time index in df
        int_pts_notna['Easting'] = int_pts_utm[0]
        int_pts_notna['Northing'] = int_pts_utm[1]
        # add back to df that includes NaN values
        int_pts['Easting'] = int_pts_notna['Easting']
        int_pts['Northing'] = int_pts_notna['Northing']

        # save intersection points in larger df
        all_ints[f'{arr1_str}_{arr2_str}_Easting'] = int_pts['Easting']
        all_ints[f'{arr1_str}_{arr2_str}_Northing'] = int_pts['Northing']

    return all_ints, all_rays


#FIXME should move this function to another file or remove entirely
def debug_ray_plot(path_station_gps, path_figures, array_list, 
                   all_ints, all_rays, median_ints, data_heli):
    for i, row in enumerate(all_rays.iterrows()):
        fig, ax = plt.subplots(1, 1, tight_layout=True)
        colors = cm.winter(np.linspace(0, 1, 5))
        for j, arr_str in enumerate(array_list):
            # get initial point, slope, and y-intercept of ray
            p = utils.station_coords_avg(path_station_gps, arr_str)
            a = row[1][arr_str]
            m = 1 / np.tan(np.deg2rad(a))
            b = p[0] - p[1]*m
            # only plot positive end of the ray (arbitrary length 100)
            if a > 180:
                ax.plot([p[1], -1000], [p[0], -1000*m+b], 
                        color=colors[j])
            else: 
                ax.plot([p[1],  1000], [p[0], 1000*m + b],
                        color=colors[j])
            # plot array centers as triangles
            ax.scatter(p[1], p[0], c='k', marker='^')
            ax.text(p[1], p[0]-0.0004, s=arr_str,
                    va='center', ha='center')
        #FIXME for testing purposes plot all intersections
        for k in np.arange(0, 20, 2):
            t = row[0]
            ax.scatter(all_ints.loc[t][k+1], all_ints.loc[t][k],
                    c='green', marker='o')

        # plot median point
        med_lat = median_ints.loc[t]['Lat']
        med_lon = median_ints.loc[t]['Lon']
        ax.plot(med_lon, med_lat, c='orange', marker='*', 
                markersize=10, label='Median Intersection')

        # plot helicopter "true" point
        heli_lat = data_heli.loc[t]['Latitude']
        heli_lon = data_heli.loc[t]['Longitude']
        ax.plot(heli_lon, heli_lat, c='red', marker='*', 
                markersize=10, label='True Heli Location')

        ax.legend(loc='upper left')
        ax.set_xlim([-116.85, -116.74])
        ax.set_ylim([43.10, 43.18])

        plt.suptitle(t)
        #plt.show()
        print(os.path.join(path_figures, "backaz_errors", f"{t}_timestep.png"))
        plt.savefig(os.path.join(path_figures, "backaz_errors", f"{t}_timestep.png"), dpi=500)
        plt.close()
    return


if __name__ == "__main__":
    # choose times when helicopter is moving so we can compare points
    # TIMES FOR 24-32 HZ
    #freqmin = 24.0
    #freqmax = 32.0
    #t0 = datetime.datetime(2023, 10, 7, 17, 0, 0, tzinfo=pytz.UTC)
    #tf = datetime.datetime(2023, 10, 7, 18, 30, 0, tzinfo=pytz.UTC)
    # TIMES FOR 2-8 HZ
    freqmin = 2.0
    freqmax = 8.0
    t0 = datetime.datetime(2023, 10, 6, 20, 0, 0, tzinfo=pytz.UTC)
    tf = datetime.datetime(2023, 10, 6, 21, 0, 0, tzinfo=pytz.UTC)

    main(settings.path_processed, settings.path_heli, 
         settings.path_station_gps, settings.path_output,
         t0, tf, freqmin, freqmax)
