#!/usr/bin/python3
import pytest, sys, os, itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from obspy.signal.util import util_geo_km, util_lon_lat
# import from personal scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import code.crossbeam as crossbeam

def test_triangulate():
    '''
    Test entire triangulation workflow
    '''


    # try with test set of points first
    #test1 = {'p1': np.array([1, 3]),
    #         'a1': 120,
    #         'p2': np.array([5, 7]),
    #         'a2': 200,
    #         'int_x': 3.10,
    #         'int_y': 1.79}

    date_list = ["2023-10-7"]#, "2023-10-6"]#, "2023-10-5"]
    freqmin = 24.0
    freqmax = 32.0
    freq_str = f"{freqmin}_{freqmax}"
    array_list = ['TOP', 'JDNA', 'JDNB', 'JDSA', 'JDSB']
    path_harddrive = os.path.join("/", "media", "mad", "LaCie 2 LT", "research", "reynolds-creek")
    path_home = os.path.join("/", "home", "mad", "Documents", "research", "reynolds-creek")
    path_processed = os.path.join(path_harddrive, "data", "processed")
    path_station_gps = os.path.join(path_harddrive, "data", "gps")
    path_figures = os.path.join(path_home, "figures")


    # call triangulate with debug flag on


    # initialize empty dfs
    all_ints = pd.DataFrame()
    all_rays = pd.DataFrame()
    # calculate intersections at each timestep for each pair of arrays
    for i, (arr1_str, arr2_str) in enumerate(itertools.combinations(array_list, 2)):
        # load in processed data and calculate center point of array (lat, lon)
        arr1, p1_latlon = crossbeam.load_ray_data(path_processed, arr1_str, date_list, freq_str, path_station_gps)
        arr2, p2_latlon = crossbeam.load_ray_data(path_processed, arr2_str, date_list, freq_str, path_station_gps)
        # change from lat,lon to x,y in km
        #p1_km = np.array([0, 0])
        #p2_km = util_geo_km(orig_lon=p1_latlon[1], orig_lat=p1_latlon[0], 
        #                    lon=p2_latlon[1], lat=p2_latlon[0])
        # calculate intersection
        #int_pts_km = arr1['Backaz'].combine(arr2['Backaz'], 
        #                                   (lambda a1, a2: triangulate.intersection(p1_km, a1, p2_km, a2)))
        ## change back to lat/lon from km
        #int_pts = pd.DataFrame([util_lon_lat(orig_lon=p1_latlon[1], orig_lat=p1_latlon[0], 
        #                                     x=km[0], y=km[1]) for km in int_pts_km], 
        #                       columns=['Lon', 'Lat'], index=int_pts_km.index)
        p1 = [p1_latlon[1], p1_latlon[0]]
        p2 = [p2_latlon[1], p2_latlon[0]]
        int_pts_result = arr1['Backaz'].combine(arr2['Backaz'], 
                                           (lambda a1, a2: crossbeam.intersection(p1, a1, p2, a2)))
        int_pts = pd.DataFrame(index=int_pts_result.index)
        int_pts['Lon'] = [x[0] for x in int_pts_result]
        int_pts['Lat'] = [x[1] for x in int_pts_result]
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

    

    ##### PLOT
    for i, ray_row in enumerate(all_rays.iterrows()):
        fig, ax = plt.subplots(1, 1, tight_layout=True)
        colors = plt.cm.winter(np.linspace(0, 1, 5))

        # location of TOP in km
        #p0_km = np.array([0, 0])
        #p0 = triangulate.station_coords_avg(path_station_gps, 'TOP')
        for j, arr_str in enumerate(array_list):
            # get initial point, slope, and y-intercept of ray
            p = crossbeam.station_coords_avg(path_station_gps, arr_str)
            #p_km = util_geo_km(orig_lon=p0[1], orig_lat=p0[0], 
            #                    lon=p[1], lat=p[0])
            a = ray_row[1][arr_str]
            # plot slope with azimuthal angle
            m = 1 / np.tan(np.deg2rad(a))
            b = p[0] - m*p[1]
            ax.plot([p[1],  100], [p[0], 100*m + b],
                     color=colors[j])
            
            # plot array centers as triangles
            ax.scatter(p[1], p[0], c='k', marker='^')
            ax.text(p[1], p[0]-0.0004, s=arr_str,
                    va='center', ha='center')
            
        #FIXME for testing purposes plot all ints
        for k in np.arange(0, 20, 2):
            t = ray_row[0]
            # change int pts to km
            #all_ints
            ax.scatter(all_ints.loc[t][k+1], all_ints.loc[t][k],
                       c='green', marker='o')
        #FIXME this is not right....


        # plot median point
        med_lat = median_ints.loc[t]['Lat']
        med_lon = median_ints.loc[t]['Lon']
        ax.plot(med_lon, med_lat, c='orange', marker='*', 
                   markersize=10, label='Median Intersection')
    
        ax.legend(loc='upper left')
        ax.set_xlim([-116.85, -116.74])
        ax.set_ylim([43.10, 43.18])

        plt.suptitle(t)
        plt.show()



    

    return

test_triangulate()