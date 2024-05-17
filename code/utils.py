#!/usr/bin/python3

import glob, obspy, os
import pandas as pd
from obspy.core.util import AttribDict



#TODO add option to specify a date range
def load_data(path_data, gem_include=None, gem_exclude=None,
              filter_type=None, **filter_options):
    '''
    Loads in and pre-processes array data.
        Loads all miniseed files in a specified directory into an obspy stream. 
        Assigns coordinates to all traces. If specified, filters data. If specified, 
        only returns a subset of gems (otherwise, returns full array).
    INPUTS
        path_data : str : Path to data folder. Should contain all miniseed files 
            under 'mseed' dir, and coordinates in .csv file(s).
        gem_include : list of str : Optional. If specified, should list Gem station
            names to include in processing. Mutually exclusive with gem_exclude.
        gem_exclude : list of str : Optional. If specified, should list Gem station
            names to include in processing. Mutually exclusive with gem_include.
        filter_type : str : Optional. Obspy filter type. Includes 'bandpass', 
            'highpass', and 'lowpass'.
        filter_options : dict : Optional. Obspy filter arguments. For 'bandpass', 
            contains freqmin and freqmax. For low/high pass, contains freq.
    RETURNS
        data : obspy stream : Stream of data traces for full array, or specified 
            Gems. Stats include assigned coordinates.
    '''
    # paths to mseed and coordinates
    path_mseed = os.path.join(path_data, "*.mseed")
    #TODO FIXME YIKES
    path_coords = glob.glob(os.path.join(path_data, "..", "gps", "*.csv" ))

    # import data as obspy stream
    data = obspy.read(path_mseed)

    # import coordinates
    coords = pd.DataFrame()
    for file in path_coords:
        coords = pd.concat([coords, pd.read_csv(file)])
    coords["Name"] = coords["Name"].astype(str) # SN of gem
    
    # get rid of any stations that don't have coordinates
    data_list = [trace for trace in data.traces 
                    if trace.stats['station'] in coords["Station"].to_list()]
    # convert list back to obspy stream
    data = obspy.Stream(traces=data_list)

    # assign coordinates to stations
    for _, row in coords.iterrows():
        sn = row["Station"]
        for trace in data.select(station=sn):
            trace.stats.coordinates = AttribDict({
                'latitude': row["Latitude"],
                'longitude': row["Longitude"],
                'elevation': row["Elevation"] }) 
    
    if filter_type != None:
        # filter data
        data = data.filter(filter_type, **filter_options)

    # merge dates (discard overlaps and leave gaps)
    data = data.merge(method=0)
    
    # only use specified subset of gems
    if gem_include != None:
        data_subset = [trace for trace in data.traces if trace.stats['station'] in gem_include]
        data_subset = obspy.Stream(traces=data_subset)
        data = data_subset
    elif gem_exclude != None:
        data_subset = [trace for trace in data.traces if trace.stats['station'] not in gem_exclude]
        data_subset = obspy.Stream(traces=data_subset)
        data = data_subset

    return data



def arg_split_comma(arg):
    if arg != None:
        arg = [s for s in arg.split(",")]
        return arg
    else:
        # return None
        return arg
    