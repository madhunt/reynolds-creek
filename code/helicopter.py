#!/usr/bin/python3

# code to import helicopter data and overplot with infrasound data
# copy of triangulate

import plot_utils, utils
import argparse, math, os, datetime, glob, pytz, itertools
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
import matplotlib.animation as animation

def main(path_home):

    date_list = ["2023-10-7"]#, "2023-10-6"]#, "2023-10-5"]
    freqmin = 24.0
    freqmax = 32.0
    freq_str = f"{freqmin}_{freqmax}"

    array_list = ['TOP', 'JDNA', 'JDNB', 'JDSA', 'JDSB']

    # figure with all array backaz in subplots
    #plot_backaz_heli(path_home, date_list, array_list, freq_str)
    

    # calculate all intersections between each 2 arrays
    all_ints = pd.DataFrame()
    for (arr1_str, arr2_str) in itertools.combinations(array_list, 2):
        arr1, p1_latlon = load_ray_data(arr1_str, date_list, freq_str)
        arr2, p2_latlon = load_ray_data(arr2_str, date_list, freq_str)

        # change from lat/lon to x,y in km
        p1_xy = np.array([0, 0])
        p2_xy = util_geo_km(orig_lon=p1_latlon[0], orig_lat=p1_latlon[1], lon=p2_latlon[0], lat=p2_latlon[1])

        # calculate intersection (using angle, not backaz)
        int_pts_km = arr1['Angle'].combine(arr2['Angle'], (lambda a1, a2: intersection(p1_xy, a1, p2_xy, a2)))
        
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

    # load in true helicopter data
    path_heli = os.path.join(path_home, "data", "helicopter")
    data_heli = adsb_kml_to_df(path_heli)
    data_heli = data_heli.set_index('Time')

    
    #############
    filt = lambda arr: arr[(arr.index > '2023-10-07T17:45:00') & (arr.index < '2023-10-07T18:00:00')]
    median_ints = filt(median_ints)
    data_heli = filt(data_heli)
    # resample data heli
    data_heli = data_heli[~data_heli.index.duplicated(keep='first')]
    data_heli = data_heli.resample('30s').nearest()
    # get rid of outliers
    median_ints = median_ints[~((median_ints['Lat'] > median_ints['Lat'].quantile(0.99)) | 
                                (median_ints['Lat'] < median_ints['Lat'].quantile(0.01)) | 
                                (median_ints['Lon'] > median_ints['Lon'].quantile(0.99)) | 
                                (median_ints['Lon'] < median_ints['Lon'].quantile(0.01)) )]


    fig, ax = plt.subplots(3, 1, tight_layout=True, height_ratios=[3,1,1])
    ax[0].set_xlim([min(median_ints['Lon'].min(), data_heli['Longitude'].min()),
                max(median_ints['Lon'].max(), data_heli['Longitude'].max())])
    ax[0].set_ylim([min(median_ints['Lat'].min(), data_heli['Latitude'].min()),
                max(median_ints['Lat'].max(), data_heli['Latitude'].max())])
    # ax1 is timeseries
    ax[1].plot(median_ints.index, median_ints['Lat'], 'ro')
    ax[1].plot(data_heli.index, data_heli['Latitude'], 'k-', label='Actual')
    ax[1].set_title('Latitude')
    ax[2].plot(median_ints.index, median_ints['Lon'], 'ro')
    ax[2].plot(data_heli.index, data_heli['Longitude'], 'k-', label='Actual')
    ax[2].set_title('Longitude')
    ax[1].sharex(ax[2])
    ax[2].set_xlabel("Mountain Time (Local)")
    ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M", tz="US/Mountain"))
    fig.autofmt_xdate()

    ax[0].scatter(data_heli['Longitude'], data_heli['Latitude'], c='k', alpha=0.5, label='Actual')
    ax[0].scatter(median_ints['Lon'], median_ints['Lat'], c='r', alpha=0.5, label='Triangulated')
    ax[0].legend(loc='upper right')
    fig.suptitle(f'{freq_str} Infrasound and Helicopter Coordinates')
    plt.show()

    # scatterplot
    def animate_heli():
        fig, ax = plt.subplots(3, 1, tight_layout=True, height_ratios=[3,1,1])
        ax[0].set_xlim([min(median_ints['Lon'].min(), data_heli['Longitude'].min()),
                    max(median_ints['Lon'].max(), data_heli['Longitude'].max())])
        ax[0].set_ylim([min(median_ints['Lat'].min(), data_heli['Latitude'].min()),
                    max(median_ints['Lat'].max(), data_heli['Latitude'].max())])
        # ax1 is timeseries
        ax[1].plot(median_ints.index, median_ints['Lat'], 'ro')
        ax[1].plot(data_heli.index, data_heli['Latitude'], 'k-', label='Actual')
        ax[1].set_title('Latitude')
        ax[2].plot(median_ints.index, median_ints['Lon'], 'ro')
        ax[2].plot(data_heli.index, data_heli['Longitude'], 'k-', label='Actual')
        ax[2].set_title('Longitude')
        ax[1].sharex(ax[2])
        ax[2].set_xlabel("Mountain Time (Local)")
        ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M", tz="US/Mountain"))
        fig.autofmt_xdate()

        graph1 = ax[0].scatter([], [], c='k', alpha=0.5, label='Actual')
        graph2 = ax[0].scatter([], [], c='r', alpha=0.5, label='Triangulated')
        v1 = ax[1].axvline(data_heli.index[0], ls='-', color='b', lw=1)
        v2 = ax[2].axvline(data_heli.index[0], ls='-', color='b', lw=1)
        ax[0].legend(loc='upper right')
        def animate(i):
            graph1.set_offsets(np.vstack((data_heli['Longitude'][:i+1], data_heli['Latitude'][:i+1])).T)
            graph2.set_offsets(np.vstack((median_ints['Lon'][:i+1], median_ints['Lat'][:i+1])).T)
            v1.set_xdata([data_heli.index[i], data_heli.index[i]])
            v2.set_xdata([data_heli.index[i], data_heli.index[i]])
            return graph1, graph2, v1, v2
        ani = animation.FuncAnimation(fig, animate, repeat=True, interval=50, frames=(len(data_heli)-1))
        #plt.show()
        ani.save(os.path.join(path_home, "figures", f"heli_coords.gif"), dpi=200)#, writer=animation.PillowWriter(fps=30))
        return



    # timeseries plots
    fig, ax = plt.subplots(2, 1, tight_layout=True, sharex=True, figsize=[8,5])

    ax[0].plot(median_ints.index, median_ints['Lat'], 'ro')
    ax[0].plot(data_heli['Time'], data_heli['Latitude'], 'k-', label='Actual')
    ax[1].plot(median_ints.index, median_ints['Lon'], 'ro')
    ax[1].plot(data_heli['Time'], data_heli['Longitude'], 'k-', label='Actual')

    #ax[0].xaxis.set_major_locator(mdates.HourLocator(byhour=range(24), interval=1))

    ax[0].set_ylim([43.05, 43.17])
    ax[1].set_ylim([-116.85,-116.70])
    ax[0].set_xlim([datetime.datetime(2023, 10, 7, 9, 45, 0, tzinfo=pytz.timezone("US/Mountain")), 
                datetime.datetime(2023, 10, 7, 11, 30, 0, tzinfo=pytz.timezone("US/Mountain"))])

    ax[0].set_title("Latitude")
    ax[1].set_title("Longitude")
    ax[0].legend(loc='lower right')
    ax[1].legend(loc='lower right')
    ax[0].grid()
    ax[1].grid()

    # save figure
    plt.savefig(os.path.join(path_home, "figures", f"heli_coords_comparison.png"), dpi=500)
    plt.close()
    return
    


def plot_backaz_heli(path_home, date_list, array_list, freq_str):
    '''
    Plot calculated backazimuths overlaid with helicopter location data for 
    each array (TOP, JDNA, JDNB, JDSA, and JDSB). 
    INPUTS:

    RETURNS:
    '''
    # load in helicopter data
    path_heli = os.path.join(path_home, "data", "helicopter")
    data_heli = adsb_kml_to_df(path_heli)

    fig, ax = plt.subplots(5, 1, sharex=True, tight_layout=True, figsize=[14,9])

    for i, array_str in enumerate(array_list):
        # convert heli coords to dist/azimuth from array
        data_heli = helicoords_to_az(path_home, data_heli, array_str)
        
        # set index as time and remove any duplicate values
        data_heli = data_heli.set_index('Time')
        data_heli = data_heli[~data_heli.index.duplicated(keep='first')]

        # try some reindexing to plot NaNs
        # NOTE does not work do this corret pls
        data_heli = data_heli.reindex(pd.date_range(start=data_heli.index.min(), 
                                                    end=data_heli.index.max(), 
                                                    freq=np.diff(data_heli.index.to_numpy()).min()))



        # load processed infrasound data for all days of interest
        path_processed = os.path.join("/", "media", "mad", "LaCie 2 LT", "research", 
                                    "reynolds-creek", "data", "processed")
        output = pd.DataFrame()
        for date_str in date_list:
            file = os.path.join(path_processed, f"processed_output_{array_str}_{date_str}_{freq_str}.pkl")
            output_tmp = pd.read_pickle(file)
            output = pd.concat([output, output_tmp])

        # plot processed data on array subplot
        fig, axi = plot_utils.plot_backaz(output=output, path_home=path_home, 
                            subtitle_str=f"{array_str}", file_str=None,
                            fig=fig, ax=ax[i])
        # plot heli data on array subplot
        axi.plot(data_heli.index, data_heli['Azimuth'], '-', color='green', 
                alpha=0.6, label='Helicopter Track')
        axi.legend(loc='upper right')
        axi.set_xlim([datetime.datetime(2023, 10, 7, 9, 0, 0, tzinfo=pytz.timezone("US/Mountain")), 
                    datetime.datetime(2023, 10, 7, 16, 0, 0, tzinfo=pytz.timezone("US/Mountain"))])
        # hide x-label for all but last plot
        if i < 4:
            axi.xaxis.label.set_visible(False)

    fig.suptitle(f"Backazimuth, Data Filtered {freq_str}")
    plt.show()
    print('done')
    # save figure
    #plt.savefig(os.path.join(path_home, "figures", f"backaz_ALLARRAYS_{freq_str}.png"), dpi=500)
    #plt.close()

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
    output['Angle'] = (-1 * output['Backaz'] + 90)%360

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
