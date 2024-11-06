#!/usr/bin/python3

import os, datetime, glob, pytz, itertools, xmltodict
import numpy as np
import pandas as pd
import scipy as sci
from obspy.geodetics.base import gps2dist_azimuth
from obspy.signal.util import util_geo_km, util_lon_lat
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

def main(debug=False):
    # TODO change to command line input or something less ugly
    date_list = ["2023-10-7"]#, "2023-10-6"]#, "2023-10-5"]
    freqmin = 24.0
    freqmax = 32.0
    freq_str = f"{freqmin}_{freqmax}"
    array_list = ['TOP', 'JDNA', 'JDNB', 'JDSA', 'JDSB']

    path_harddrive = os.path.join("/", "media", "mad", "LaCie 2 LT", "research", "reynolds-creek")
    path_home = os.path.join("/", "home", "mad", "Documents", "research", "reynolds-creek")
    path_processed = os.path.join(path_harddrive, "data", "processed")
    path_heli = os.path.join(path_harddrive, "data", "helicopter")
    path_station_gps = os.path.join(path_harddrive, "data", "gps")
    path_figures = os.path.join(path_home, "figures")

    
    ##TODO FOR TESTING PURPOSES
    # choose times when helicopter is moving so we can compare points
    t0 = '2023-10-07T17:00:00'
    tf = '2023-10-07T18:30:00'

    # initialize empty dfs
    all_ints = pd.DataFrame()
    if debug == True: 
        all_rays = pd.DataFrame()

    # calculate intersections at each timestep for each pair of arrays
    for i, (arr1_str, arr2_str) in enumerate(itertools.combinations(array_list, 2)):
        # load in processed data and calculate center point of array (lat, lon)
        arr1, p1 = load_ray_data(path_processed, arr1_str, date_list, freq_str, path_station_gps, 
                                 t0, tf)
        arr2, p2 = load_ray_data(path_processed, arr2_str, date_list, freq_str, path_station_gps, 
                                 t0, tf)

        # calculate intersection
        int_pts = pd.DataFrame()
        int_pts['result'] = arr1['Backaz'].combine(arr2['Backaz'], 
                                           (lambda a1, a2: intersection(p1, a1, p2, a2, latlon=True)))
        int_pts['Lat'] = [x[0] for x in int_pts['result']]
        int_pts['Lon'] = [x[1] for x in int_pts['result']]
        int_pts.drop('result', axis=1)
        # save intersection points
        all_ints[f'{arr1_str}_{arr2_str}_Lat'] = int_pts['Lat']#[col[1] for col in int_pts]
        all_ints[f'{arr1_str}_{arr2_str}_Lon'] = int_pts['Lon']#[col[0] for col in int_pts]

        
        
        if debug == True: 
            # save rays
            all_rays[arr1_str] = arr1['Backaz']
            all_rays[arr2_str] = arr2['Backaz']

    #TODO how many beams are crossing
    # this gives LAT AND LON
    #num_nans = all_ints.isna().sum(axis=1)
    # weight result by this??

    # now find median intersection point
    median_ints = pd.DataFrame(index=int_pts.index)
    median_ints['Lat'] = all_ints.filter(regex='Lat').median(axis=1)
    median_ints['Lon'] = all_ints.filter(regex='Lon').median(axis=1)

    # load in true helicopter data
    data_heli = adsb_kml_to_df(path_heli)
    #data_heli = data_heli.set_index('Time')
    # resample heli data
    filt = lambda arr: arr[(arr['Time'] > t0) & (arr['Time'] < tf)]
    data_heli = filt(data_heli ).set_index('Time')
    data_heli = data_heli[~data_heli.index.duplicated(keep='first')]
    data_heli = data_heli.resample('30s').nearest() #TODO make sure nearest is ok -- i think it is

    # now have an array of triangulated points (median_ints)
        # and an array of actual points (data_heli)
        # at the same times (every 30 s during a time when the heli is moving)
    data_heli = data_heli.rename(columns={'Longitude':'Lon True', 'Latitude':'Lat True'})
    #def root_mean_square_error(model, measurements):
    #    #assert len(model) == len(measurements)
    #    N = len(measurements)
    #    rmse = np.sqrt(1/N * np.sum((model - measurements)**2))
    #    return rmse
    #rmse_lat = root_mean_square_error(data_heli['Lat True'], median_ints['Lat'])
    #rmse_lon = root_mean_square_error(data_heli['Lon True'], median_ints['Lon'])

    #diff_lat = data_heli['Lat True'] - median_ints['Lat']
    #print(np.nanmax(diff_lat))
    #diff_lon = data_heli['Lon True'] - median_ints['Lon']
    #print(np.nanmax(diff_lon))

    # calculate error
    error = np.sqrt((data_heli['Lon True'] - median_ints['Lon'])**2 
                    + (data_heli['Lat True'] - median_ints['Lat'])**2)
    # calculate colors based on number of intersections
    int_top = all_ints[all_ints.columns[all_ints.columns.str.contains('TOP')]].isna().sum(axis=1) /2 # div by 2 bc lat AND lon
    int_jd = all_ints[all_ints.columns[~all_ints.columns.str.contains('TOP')]].isna().sum(axis=1) /2
    red = int_top / 4      # max # of intersections is each JD with TOP, or 4 total
    blue = int_jd / 6        # max # is each JD with each other
    green = np.zeros(shape=np.shape(red))
    colors = np.vstack([red, green, blue]).T
    #colors = colors[:len(data_heli)]            #TODO remove these len data helis from everywhere and clean up
    
    fig, ax = plt.subplots(1, 1)
    ax.scatter(data_heli['Lon True'][20:25], data_heli['Lat True'][20:25],
               s=5000* error[:len(data_heli)][20:25], c=colors[:len(data_heli)][20:25])
    # plot some landmarks
    for arr in array_list:
        y, x, _  = station_coords_avg(path_station_gps, arr)
        plt.plot(x, y, 'g^')
    #ax.scatter(data_heli['Lon True'], data_heli['Lat True'], 'ko'
    #           s=)
    
    plt.savefig('temp.png')
    #plt.show()
    plt.close()





    if debug == True:
        # plot backaz rays and intersection points
        debug_ray_plot(path_station_gps, path_figures, array_list,
                       all_ints, all_rays, median_ints, data_heli)
    

    # quantify uncertainty -- difference in true and calculated heli location
    

    # loop through time and plot backaz rays at each time
    # also get waveforms
        ##TODO
        # plot waveforms / pwr for each array (median?)
        # waveform for TOP? 



    return


def debug_ray_plot(path_station_gps, path_figures, array_list, 
                   all_ints, all_rays, median_ints, data_heli):
    for i, row in enumerate(all_rays.iterrows()):
        fig, ax = plt.subplots(1, 1, tight_layout=True)
        colors = cm.winter(np.linspace(0, 1, 5))
        for j, arr_str in enumerate(array_list):
            # get initial point, slope, and y-intercept of ray
            p = station_coords_avg(path_station_gps, arr_str)
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


#def basic_main(path_home):
    #date_list = ["2023-10-7"]#, "2023-10-6"]#, "2023-10-5"]
    #array_str = "TOP"
    #freqmin = 24.0
    #freqmax = 32.0
    #freq_str = f"{freqmin}_{freqmax}"

    ## load in helicopter data
    #path_heli = os.path.join(path_home, "data", "helicopter")
    #data_heli = adsb_kml_to_df(path_heli)

    ## convert heli coords to dist/azimuth from array
    #data_heli = helicoords_to_az(path_home, data_heli, array_str)

    ## load processed infrasound data
    ## path to data on harddrive
    #path_processed = os.path.join("/", "media", "mad", "LaCie 2 LT", "research", 
                                  #"reynolds-creek", "data", "processed")
    ##path_processed = os.path.join(path_home, "data", "processed", "window_60s")
    #output = pd.DataFrame()
    #for date_str in date_list:
        #file = os.path.join(path_processed, f"processed_output_{array_str}_{date_str}_{freq_str}.pkl")
        #output_tmp = pd.read_pickle(file)
        #output = pd.concat([output, output_tmp])

    ## plot processed data
    #fig, ax = plot_utils.plot_backaz(output=output, path_home=path_home, 
                        #subtitle_str=f"{array_str} Array, Filtered {freqmin} to {freqmax} Hz", file_str=None)
    
    ## plot heli data
    #ax.plot(data_heli['Time'], data_heli['Azimuth'], '-', color='green', 
            #alpha=0.6, label='Helicopter Track')
    #ax.legend(loc='upper right')

    #ax.set_xlim([datetime.datetime(2023, 10, 7, 9, 0, 0, tzinfo=pytz.timezone("US/Mountain")), 
                 #datetime.datetime(2023, 10, 7, 16, 0, 0, tzinfo=pytz.timezone("US/Mountain"))])

    ## save figure
    #plt.savefig(os.path.join(path_home, "figures", f"backaz_{array_str}_{freq_str}.png"), dpi=500)
    #plt.close()

    #return

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
    
def station_coords_avg(path_gps, array_str):
    '''
    Find mean location for entire array. 
    INPUTS: 
        path_home : str : Path to main dir.
    RETURNS:
        lat : float : Average latitude for station.
        lon : float : Average longitude for station.
        elv : float : Average elevation for station.
    '''
    # TODO FIXME path
    path_coords = glob.glob(os.path.join(path_gps, "*.csv" ))
    coords = pd.DataFrame()
    for file in path_coords:
        coords = pd.concat([coords, pd.read_csv(file)])
    # filter by array
    coords = coords[coords["Station"].str.contains(array_str)]

    lat = coords['Latitude'].mean()
    lon = coords['Longitude'].mean()
    elv = coords['Elevation'].mean()

    return lat, lon, elv

def helicoords_to_az(path_station_gps, data, array_str):

    # find avg location for entire array specified
    #TODO FIXME path
    coords_top = station_coords_avg(path_station_gps, array_str)

    # use gps2dist to get distance and azimuth between heli and TOP array
    data[['Distance', 'Azimuth', 'az2']] = data.apply(lambda x: 
                                                                gps2dist_azimuth(lat1=coords_top[0], lon1=coords_top[1], 
                                                                                lat2=x["Latitude"], lon2=x["Longitude"]), 
                                                                                axis=1, result_type='expand')
    data = data.drop('az2', axis=1)

    return data

def load_ray_data(path_processed, array_str, date_list, freq_str, path_gps, t0, tf):
    # load processed infrasound data
    output = pd.DataFrame()
    for date_str in date_list:
        file = os.path.join(path_processed, f"processed_output_{array_str}_{date_str}_{freq_str}.pkl")
        output_tmp = pd.read_pickle(file)
        output = pd.concat([output, output_tmp])
    
    #output = output.set_index('Time')
    
    filt = lambda arr: arr[(arr['Time'] > t0) & (arr['Time'] < tf)]
    output = filt(output).set_index('Time')

    # add column for angle (0 deg at x-axis, 90 deg at y-axis)
    #output['Angle'] = (-1 * output['Backaz'] + 90)%360

    # get start points
    lat, lon, elv = station_coords_avg(path_gps, array_str)
    p_start = np.array([lat, lon])

    return output, p_start

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

if __name__ == "__main__":
    
    # path to this file
    path_curr = os.path.dirname(os.path.realpath(__file__))
    path_home = os.path.abspath(os.path.join(path_curr, '..'))

    main(debug=True)
    #basic_main(path_home=path_home)
