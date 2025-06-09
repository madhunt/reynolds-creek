#!/usr/bin/python3
'''
Define paths to use in all files when run from 
this computer.
'''
import os

path_harddrive = os.path.join("/", "media", "mad", "LaCie 2 LT", "research", "reynolds-creek")
path_home = os.path.join("/", "home", "mad", "Documents", "research", "reynolds-creek")

path_processed = os.path.join(path_home, "data", "processed")
path_mseed = os.path.join(path_harddrive, "data", "mseed")
path_heli = os.path.join(path_home, "data", "helicopter")
path_station_gps = os.path.join(path_home, "data", "gps")
path_output = os.path.join(path_home, "data", "output")
path_figures = os.path.join(path_home, "figures")


#path_processed_uncert = os.path.join(path_harddrive, "geoph522")

