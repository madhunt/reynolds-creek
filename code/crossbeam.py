#!/usr/bin/python3
'''
Crossbeams to find intersection points between all five arrays (TOP, JD*). 
Finds median intersection point (in UTM), 
and number of total intersections/crossbeams at each timestep.
Re-samples "true" helicopter data during same time period.
Calculates distance between true helicopter point and median intersection.
'''
import os, datetime, glob, pytz, itertools, xmltodict, utm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

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
    data_heli = adsb_kml_to_df(path_heli, latlon=False)
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


def adsb_kml_to_df(path, latlon=True):
    '''
    Loads in aircraft flight track (downloaded from ADS-B Exchange https://globe.adsbexchange.com/) 
    from KML as a pandas dataframe.  
    INPUTS: 
        path    : str   : Path to dir containing all KML files for one aircraft of interest.
        latlon  : bool  : If True, returns data as latitude and longitudes. If False, returns 
            data as UTM coordinates. 
    RETURNS: 
        data_latlon : pandas df : Dataframe containing data from all KML files in specified dir. Columns are 
                    Time        : datetime  : Timestamp of location reading
                    Latitude    : float     : Latitude of aircraft
                    Longitude   : float     : Longitude of aircraft
                    Altitude    : float     : Altitude of aircraft
        OR
        data_utm    : pandas df : Dataframe containing data from all KML files in specified dir. Columns are
                    Time        : datetime  : Timestamp of location reading
                    Easting     : float     : UTM Easting of aircraft
                    Northing    : float     : UTM Northing of aircraft
                    Altitude    : float     : Altitude of aircraft

    '''
    files_kml = glob.glob(os.path.join(path, "*.kml" ))
    data_latlon = pd.DataFrame()

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
        data = data.drop('coord', axis=1)   # clean up temp column
        # store data from multiple files
        data_latlon = pd.concat([data_latlon, data])
    
    # do some cleanup
    data_latlon = data_latlon.sort_values("Time").drop_duplicates()

    if latlon == True:
        return data_latlon
    else:
        # convert coordinates to UTM
        utm_coords = utm.from_latlon(data_latlon['Latitude'].to_numpy(), 
                                     data_latlon['Longitude'].to_numpy())
        data_utm = pd.DataFrame(index=data_latlon.index)
        data_utm['Time'] = data_latlon['Time']
        data_utm['Easting'] = utm_coords[0]
        data_utm['Northing'] = utm_coords[1]
        data_utm['Altitude'] = data_latlon['Altitude']
        return data_utm
    

def station_coords_avg(path_gps, array_str, latlon=True):
    '''
    Find mean location for entire array. 
    INPUTS: 
        path_home : str : Path to main dir.
        latlon  : bool  : If True, returns data as latitude and longitudes. If False, returns 
            data as UTM coordinates. 
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

    if latlon == True:
        return lat, lon
    else:
        # convert coordinates to UTM
        utm_coords = utm.from_latlon(lat, lon)
        easting = utm_coords[0]
        northing = utm_coords[1]
        return easting, northing


def load_ray_data(path_processed, array_str, freq_str, t0, tf):
    # load processed infrasound data between given dates
    output = pd.DataFrame()
    for date_str in create_date_list(t0, tf):
        file = os.path.join(path_processed, f"processed_output_{array_str}_{date_str}_{freq_str}.pkl")
        output_tmp = pd.read_pickle(file)
        output = pd.concat([output, output_tmp])
    
    # now filter data between given times
    filt = lambda arr: arr[(arr['Time'] > t0) & (arr['Time'] < tf)]
    output = filt(output).set_index('Time')
    return output


def create_date_list(t0, tf):
    def date_to_str(datetime_obj):
        return datetime_obj.strftime(format="%Y-%m-%-d")
    # include start and end dates in list
    date_list = [t0.date() + datetime.timedelta(days=i) for i in range((tf-t0).days + 1)]
    # format as str
    date_list = [date_to_str(i) for i in date_list]
    return date_list


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
        arr1 = load_ray_data(path_processed, arr1_str, freq_str, t0, tf)
        arr2 = load_ray_data(path_processed, arr2_str, freq_str, t0, tf)
        # calculate center point of arrays (lat, lon)
        p1 = station_coords_avg(path_station_gps, arr1_str, latlon=True)
        p2 = station_coords_avg(path_station_gps, arr2_str, latlon=True)

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


if __name__ == "__main__":
    # define paths
    path_harddrive = os.path.join("/", "media", "mad", "LaCie 2 LT", "research", "reynolds-creek")
    path_home = os.path.join("/", "home", "mad", "Documents", "research", "reynolds-creek")

    path_processed = os.path.join(path_harddrive, "data", "processed")
    path_heli = os.path.join(path_harddrive, "data", "helicopter")
    path_station_gps = os.path.join(path_harddrive, "data", "gps")
    path_output = os.path.join(path_harddrive, "data", "output")
    
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

    main(path_processed, path_heli, path_station_gps, path_output,
         t0, tf, freqmin, freqmax)
