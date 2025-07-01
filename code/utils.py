#!/usr/bin/python3
'''
Utility functions used across different files.
'''

import glob, obspy, os, glob, xmltodict, utm
import pandas as pd
import numpy as np
import datetime
from obspy.core.util import AttribDict
from obspy.geodetics.base import gps2dist_azimuth
import matplotlib.colors as colors

def load_data(path_data, path_coords, array_str=None, gem_include=None, gem_exclude=None,
              time_start=None, time_stop=None, freqmin=None, freqmax=None):
    '''
    Loads in and pre-processes array data.
        Loads all miniseed files in a specified directory into an obspy stream. 
        Assigns coordinates to all traces. If specified, filters data. If specified, 
        only returns a subset of gems (otherwise, returns full array).
    INPUTS
        path_data : str : Path to data folder. Should contain miniseed files. 
            Coordinates are stored up one directory under gps/{filename}/csv.
        gem_include : list of str : Optional. If specified, should list Gem station
            names to include in processing. Mutually exclusive with gem_exclude.
        gem_exclude : list of str : Optional. If specified, should list Gem station
            names to include in processing. Mutually exclusive with gem_include.
        time_start : 
        time_stop : "YYYY-MM-DD" 

        filter_type : str : Optional. Obspy filter type. Includes 'bandpass', 
            'highpass', and 'lowpass'.
        filter_options : dict : Optional. Obspy filter arguments. For 'bandpass', 
            contains freqmin and freqmax. For low/high pass, contains freq.
    RETURNS
        data : obspy stream : Stream of data traces for full array, or specified 
            Gems. Stats include assigned coordinates.
    '''
    # (1) read in the data
    # paths to mseed and coordinates
    path_mseed = os.path.join(path_data, f"*{array_str}*.mseed")
    #path_coords = glob.glob(os.path.join(path_data, "..", "gps", "*.csv" )) # FIXME

    # import data as obspy stream
    data = obspy.read(path_mseed)
    #data = obspy.Stream()
    #for path in path_mseed:
    #    tmp_data = obspy.read(path)
    #    for tr in tmp_data:
    #        # save path information for each trace
    #        tr.stats['path'] = path
    #    data += tmp_data

    # (2) bandpass filter data if desired
    if freqmin != None and freqmax != None:
        # filter data
        # NOTE this needs to happen before data.merge()
        data = data.filter('bandpass', freqmin=freqmin, freqmax=freqmax)

    # (3) filter by date
    # merge dates (discard overlaps and leave gaps)
    data = data.merge(method=0)
    # only keep data in the time range of interest
    data = data.trim(starttime=time_start, endtime=time_stop, keep_empty_traces=False)
    #if time_start != None:
    #    data = obspy.Stream([trace for trace in data.traces if 
    #            (trace.stats.starttime <= time_start) and (trace.stats.endtime >= time_stop)])
    
    # (4) add coordinates to traces
    # import coordinates
    #FIXME this is a hack so i can feed a pd df into this function instead of a path
    if type(path_coords) == str:
        path_coords = glob.glob(os.path.join(path_coords, "*.csv" ))[0]
        coords = pd.read_csv(path_coords)
    else:
        coords = path_coords
    coords["Name"] = coords["Name"].astype(str) # SN of gem
    
    # get rid of any stations that don't have coordinates
    data = obspy.Stream([trace for trace in data.traces 
                    if trace.stats['station'] in coords["Station"].to_list()])

    # assign coordinates to stations
    for _, row in coords.iterrows():
        sn = row["Station"]
        for trace in data.select(station=sn):
            trace.stats.coordinates = AttribDict({
                'latitude': row["Latitude"],
                'longitude': row["Longitude"],
                'elevation': row["Elevation"] }) 
    
    # (5) filter by gem station ID 
    # only use specified subset of gems
    if gem_include != None:
        data = obspy.Stream([trace for trace in data.traces if trace.stats['station'] in gem_include])
    if gem_exclude != None:
        data = obspy.Stream([trace for trace in data.traces if trace.stats['station'] not in gem_exclude])

    return data


def arg_split_comma(arg):
    if arg != None:
        arg = [s for s in arg.split(",")]
        return arg
    else:
        # return None
        return arg


def load_beamform_results(path_processed, array_str, freq_str, t0, tf):
    # formerly load_backaz_data
    # load processed infrasound data between given dates
    output = pd.DataFrame()
    for date_str in create_date_list(t0, tf):
        file = os.path.join(path_processed, f"processed_output_{array_str}_{date_str}_{freq_str}.pkl")
        try:
            output_tmp = pd.read_pickle(file)
            output = pd.concat([output, output_tmp])
        except FileNotFoundError:
            # HACK to get around files on 10-08
            pass

    # now filter data between given times
    filt = lambda arr: arr[(arr['Time'] > t0) & (arr['Time'] < tf)]
    output = filt(output).set_index('Time')
    return output

def slowness_to_sx_sy(path_processed, array_str, freq_str, t0, tf):
    # load in data
    output = load_beamform_results(path_processed, array_str, freq_str, t0, tf)

    # convert slowness vector and backaz into (sx, sy)
    output["sx"] = output["Slowness"] * np.sin(np.deg2rad(output["Backaz"]))
    output["sy"] = output["Slowness"] * np.cos(np.deg2rad(output["Backaz"]))
    return output


def create_date_list(t0, tf):
    def date_to_str(datetime_obj):
        return datetime_obj.strftime(format="%Y-%m-%-d")
    # include start and end dates in list
    date_list = [t0.date() + datetime.timedelta(days=i) for i in range((tf-t0).days + 2)]
    # format as str
    date_list = [date_to_str(i) for i in date_list]
    return date_list




def adsb_kml_to_df(path, latlon=True):
    '''
    Loads in aircraft flight track (downloaded from ADS-B Exchange https://globe.adsbexchange.com/) 
    from KML as a pandas dataframe.  
    INPUTS 
        path    : str   : Path to dir containing all KML files for one aircraft of interest.
        latlon  : bool  : If True, returns data as latitude and longitudes. If False, returns 
            data as UTM coordinates. 
    RETURNS 
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


def coords_to_az(path_station_gps, data, array_str):
    '''
    Convert coordinates (in lat/lon) to azimuths relative to the center of 
    specified array.
    INPUTS
        path_station_gps    : str       : Path to csv with GPS coordinates of infrasound stations.
        data                : pandas df : Data loaded in as lat/lon coordinates.
        array_str           : str       : Name of array to calculate azimuth from. 
    RETURNS
        data                : pandas df : Original dataframe with 'Distance' and 'Azimuth'
            columns appended on end, representing distance and azimuth from array.
    '''
    # find avg location for entire array specified (in lat/lon)
    coords_top = station_coords_avg(path_station_gps, array_str, latlon=True)
    # use gps2dist to get distance and azimuth between each coordinate and array of interest
    data[['Distance', 'Azimuth', 'az2']] = data.apply(lambda x: 
                                                                gps2dist_azimuth(lat1=coords_top[0], lon1=coords_top[1], 
                                                                                lat2=x["Latitude"], lon2=x["Longitude"]), 
                                                                                axis=1, result_type='expand')
    data = data.drop('az2', axis=1)
    return data



def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    # copied from a helpful stack overflow comment
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
    



