#!/usr/bin/python3

import glob, obspy, os
import pandas as pd
from datetime import datetime, timedelta
from obspy.core.util import AttribDict

def load_data(path_data, path_coords, array_str=None,
              gem_include=None, gem_exclude=None,
              time_start=None, time_stop=None,
              freqmin=None, freqmax=None):
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
    #coords = pd.DataFrame()
    #for file in path_coords:
    #    coords = pd.concat([coords, pd.read_csv(file)])
    coords = pd.read_csv(path_coords)
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
    