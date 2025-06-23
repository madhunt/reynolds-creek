#!/usr/bin/python3
'''
Define paths to use in all files when run from this computer.
'''
import os

path_home = None
path_processed = None
path_mseed = None
path_heli = None
path_station_gps = None
path_output = None
path_figures = None

def set_paths(location):
    global path_home, path_processed, path_mseed, path_heli, path_station_gps, path_output, path_figures

    if location == 'harddrive':
        path_home = os.path.join("/", "media", "mad", "LaCie 2 LT", "research", "reynolds-creek")
    elif location == 'laptop':
        path_home = os.path.join("/", "home", "mad", "Documents", "research", "reynolds-creek")
    elif location == 'borah':
        path_home = os.path.join("/", "bsuhome", "madelinehunt", "reynolds-creek")
    else:
        raise ValueError(f"Unknown location: {location}")

    path_processed = os.path.join(path_home, "data", "processed")
    path_mseed = os.path.join(path_home, "data", "mseed")
    path_heli = os.path.join(path_home, "data", "helicopter")
    path_station_gps = os.path.join(path_home, "data", "gps")
    path_output = os.path.join(path_home, "data", "output")
    path_figures = os.path.join(path_home, "figures")


