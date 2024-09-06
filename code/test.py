#!/usr/bin/python3

# plot animation of triangulated fire coords

import plot_utils, utils
import os, datetime, glob, pytz, itertools
import numpy as np
import pandas as pd
import scipy as sci
import obspy
from obspy.geodetics.base import gps2dist_azimuth
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from obspy.signal.array_analysis import get_geometry
from obspy.signal.util import util_geo_km, util_lon_lat
import matplotlib.animation as animation

def main(path_home):

    date_list = ["2023-10-7"]#, "2023-10-6"]#, "2023-10-5"]
    freqmin = 4.0
    freqmax = 8.0
    freq_str = f"{freqmin}_{freqmax}"

    array_list = ['TOP', 'JDNA', 'JDNB', 'JDSA', 'JDSB']

    # figure with all array backaz in subplots
    #plot_backaz_heli(path_home, date_list, array_list, freq_str)
    

    # calculate all intersections between each 2 arrays
    all_ints = pd.DataFrame()
    for (arr1_str, arr2_str) in itertools.combinations(array_list, 2):
        arr1, p1 = load_ray_data(arr1_str, date_list, freq_str)
        arr2, p2 = load_ray_data(arr2_str, date_list, freq_str)

        # change from lat/lon to x,y in km
        p1_km = np.array([0, 0])
        p2_km = util_geo_km(orig_lon=p1[0], orig_lat=p1[1], lon=p2[0], lat=p2[1])

        # calculate intersection (using angle, not backaz)
        int_pts_km = arr1['Angle'].combine(arr2['Angle'], (lambda a1, a2: intersection(p1_km, a1, p2_km, a2)))
        
        # change back to lat/lon from km
        int_pts = pd.DataFrame([util_lon_lat(p1[0], p1[1], km[0], km[1]) for km in int_pts_km], 
                               columns=['Lon', 'Lat'], index=int_pts_km.index)

        all_ints[f'{arr1_str}_{arr2_str} Lat'] = int_pts['Lat']#[col[1] for col in int_pts]
        all_ints[f'{arr1_str}_{arr2_str} Lon'] = int_pts['Lon']#[col[0] for col in int_pts]
    
    # now find median intersection point
    all_ints.index = int_pts.index
    median_ints = pd.DataFrame(index=int_pts.index)
    median_ints['Lat'] = all_ints.filter(regex='Lat').median(axis=1)
    median_ints['Lon'] = all_ints.filter(regex='Lon').median(axis=1)


    
    #############
    filt = lambda arr: arr[(arr.index > '2023-10-07T21:00:00') & (arr.index < '2023-10-07T23:59:59')]




    median_ints = filt(median_ints)
    # get rid of outliers
    #median_ints = median_ints[~((median_ints['Lat'] > median_ints['Lat'].quantile(0.99)) | 
    #                            (median_ints['Lat'] < median_ints['Lat'].quantile(0.01)) | 
    #                            (median_ints['Lon'] > median_ints['Lon'].quantile(0.99)) | 
    #                            (median_ints['Lon'] < median_ints['Lon'].quantile(0.01)) )]

    
    # scatterplot
    fig, ax = plt.subplots(3, 1, tight_layout=True, height_ratios=[3,1,1])
    #ax[0].set_xlim([median_ints['Lon'].min(), median_ints['Lon'].max()])
    #ax[0].set_ylim([median_ints['Lat'].min(), median_ints['Lat'].max()])

    # plot center of each array
    def plot_array_centers(path_home, array_str, ax):
        lat, lon, _ = station_coords_avg(path_home, array_str)
        ax[0].scatter(lon, lat, c='r', marker='^')
        ax[0].text(lon, lat, s=array_str)
        return
    for array_str in array_list:
        plot_array_centers(path_home, array_str, ax)
    

    # check xlim and ylim so all points are plotted


    # ax1 and 2 are timeseries
    ax[1].plot(median_ints.index, median_ints['Lat'], 'bo')
    ax[1].set_ylabel('Latitude')
    ax[2].plot(median_ints.index, median_ints['Lon'], 'bo')
    ax[2].set_ylabel('Longitude')
    ax[1].sharex(ax[2])
    ax[2].set_xlabel("Mountain Time (Local)")
    ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M", tz="US/Mountain"))
    fig.autofmt_xdate()

    graph1 = ax[0].scatter([], [], c='b', alpha=0.5, label='Triangulated')
    v1 = ax[1].axvline(median_ints.index[0], ls='-', color='k', lw=1)
    v2 = ax[2].axvline(median_ints.index[0], ls='-', color='k', lw=1)
    #ax[0].legend(loc='upper right')
    def animate(i):
        graph1.set_offsets(np.vstack((median_ints['Lon'][:i+1], median_ints['Lat'][:i+1])).T)
        v1.set_xdata([median_ints.index[i], median_ints.index[i]])
        v2.set_xdata([median_ints.index[i], median_ints.index[i]])
        return graph1, v1, v2
    ani = animation.FuncAnimation(fig, animate, repeat=True, interval=50, frames=(len(median_ints)-1))
    #plt.show()
    ani.save(os.path.join(path_home, "figures", f"fire_coords.gif"), dpi=200)#, writer=animation.PillowWriter(fps=30))

    return
    



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
    output['Angle'] = (-1 * output['Backaz'] + 90)%360

    # constrain data to only use points with slownesses near 3 s/km
    #slow_min = 2.8
    #slow_max = 3.2
    #output = output[output["Slowness"].between(slow_min, slow_max)]

    # get start points
    lat, lon, elv = station_coords_avg(path_home, array_str)
    # flip points since lon, lat is x, y
    p_start = np.array([lon, lat])

    return output, p_start

def intersection(p1, a1, p2, a2):
    '''
    Calculates intersection point between two rays.
    INPUTS
        l1 : np array, 2x1 : Coordinates (x,y) for start point of ray 1.
        a1 : float : Angle of direction of ray 1 (0 deg on x-axis, 90 deg on y-axis).
        l2 : np array, 2x1 : Coordinates (x,y) for start point of ray 2.
        a2 : float : Angle of direction of ray 2.
    RETURNS
        int_pt : np array, 2x1 : Coordinates (x,y) of intersection point. 
            Returns [NaN, NaN] if there is no intersection; e.g. if intersection 
            occurs "behind" ray start points or if rays are parallel. 
    '''
    # create matrix of direction unit vectors
    D = np.array([[np.cos(np.radians(a1)), -1*np.cos(np.radians(a2))],
                      [np.sin(np.radians(a1)), -1*np.sin(np.radians(a2))]])
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
