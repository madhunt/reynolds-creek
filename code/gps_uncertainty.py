#!/usr/bin/python3
'''
Code to perform uncertainty analysis of beamforming from infrasound sensor GPS locations.
Perturbs GPS coordinates and beamforms infrasound data with Monte-Carlo simulations.
'''
import numpy as np
import pandas as pd
import glob, os, argparse, datetime
from obspy.core.utcdatetime import UTCDateTime
from concurrent.futures import ProcessPoolExecutor

from obspy.core.util import AttribDict
import utils, settings, beamform

def main(data, path_station_gps, path_save,
         array_str, time_start, time_stop,
         freqmin, freqmax,
         gps_perturb_scale, iteration):

    # LOG: record processing start time
    if iteration == 0:
        # create logfile ("w")
        print(f"{datetime.datetime.now()} \t\t {iteration}: Started Processing")
        with open(os.path.join(path_save, f"pylog_{array_str}_{freqmin}-{freqmax}Hz_{gps_perturb_scale}m.txt"), "w") as f:
            print(f"{datetime.datetime.now()} \t\t {iteration}: Started Processing", file=f)
    else:
        # record in existing file
        print(f"{datetime.datetime.now()} \t\t {iteration}: Started Processing")
        with open(os.path.join(path_save, f"pylog_{array_str}_{freqmin}-{freqmax}Hz_{gps_perturb_scale}m.txt"), "a") as f:
            print(f"{datetime.datetime.now()} \t\t {iteration}: Started Processing", file=f)


    # (1) perturb coordinates --------------------- ---------------------------------- -----------------------------------
    # read in coords
    print(iteration, "reading coords")
    path_coords = glob.glob(os.path.join(path_station_gps, "*.csv" ))[0]
    coords = pd.read_csv(path_coords)
    # convert GPS perturbation from m to lat/lon degrees
    lat_scale, lon_scale = meters_to_latlon_deg(dist_m=gps_perturb_scale, lat=coords["Latitude"].mean())
    # choose random numbers to perturb GPS coords
    lat_perturb = np.random.normal(loc=0, scale=lat_scale, size=len(coords))
    lon_perturb = np.random.normal(loc=0, scale=lon_scale, size=len(coords))
    # add perturbations to coordinates
    coords["Latitude"] = coords["Latitude"] + lat_perturb
    coords["Longitude"] = coords["Longitude"] + lon_perturb

    # update coords in data
    # (2) perturb coords in data
    for _, row in coords.iterrows():
        sn = row["Station"]
        for trace in data.select(station=sn):
            trace.stats.coordinates = AttribDict({
                'latitude': row["Latitude"],
                'longitude': row["Longitude"],
                'elevation': row["Elevation"] }) 

    # (3) perform beamforming -----------------------------------

    # test if any sensors stopped recording before specified time_stop
    station_stop = np.array([data.traces[i].stats["endtime"] == time_stop for i in range(len(data.traces))])

    if np.all(station_stop):    # all sensors recorded entire duration
        print(iteration, "all recorded entire time")
        # define path to save resulting output
        path_processed = os.path.join(path_save, f"output_{array_str}_{freqmin}-{freqmax}Hz_{gps_perturb_scale}m_iteration{iteration}.pkl")
        output = beamform.process_data(data, path_processed=path_processed, 
                                        time_start=time_start, time_stop=time_stop,
                                        freqmin=freqmin, freqmax=freqmax)

    else:   # one or more sensors cut out early
        print(iteration, "cut out early")
        idxs = np.where(~station_stop)[0]
        for i in np.where(~station_stop)[0]:
            # run beamforming once with all sensors until early stop time
            time_stop_early = data.traces[i].stats["endtime"]
            path_processed = os.path.join(path_save, f"output_{array_str}_{freqmin}-{freqmax}Hz_{gps_perturb_scale}m_iteration{iteration}_a.pkl")
            output = beamform.process_data(data, path_processed=path_processed, 
                                            time_start=time_start, time_stop=time_stop_early,
                                            freqmin=freqmin, freqmax=freqmax)
            # and run beamforming with fewer sensors from early stop to end
            for tr in data.select(station=data.traces[i].stats['station']): 
                # obspy's ugly way to remove a traces from the stream... yuck
                data.remove(tr)
            path_processed = os.path.join(path_save, f"output_{array_str}_{freqmin}-{freqmax}Hz_{gps_perturb_scale}m_iteration{iteration}_b.pkl")
            output = beamform.process_data(data, path_processed=path_processed, 
                                            time_start=time_stop_early, time_stop=time_stop,
                                            freqmin=freqmin, freqmax=freqmax)

    # LOG: record processing end time
    print(f"{datetime.datetime.now()} \t\t {iteration}: Finished Processing")
    with open(os.path.join(path_save, f"pylog_{array_str}_{freqmin}-{freqmax}Hz_{gps_perturb_scale}m.txt"), "a") as f:
        print(f"{datetime.datetime.now()} \t\t {iteration}: Finished Processing", file=f)

    return


def meters_to_latlon_deg(dist_m, lat):
    """
    Converts a distance in meters to degrees latitude and longitude.
    INPUTS
        dist_m          : float     : Distance to convert, in meters.
        lat             : float     : Latitude of point (approximate).
    RETURNS
        dist_deg_lat    : float     : Distance in degrees latitude.
        dist_deg_lon    : float     : Distance in degrees longitude at point of interest.
    """
    r_earth = 6371e3    # radius of earth in m
    lat_m_conv = r_earth * np.deg2rad(1)    # cm / 1 deg lat
    dist_deg_lat = dist_m / lat_m_conv

    # m in lon depends on lat of point
    lon_m_conv = np.deg2rad(1) * r_earth * np.cos(lat)
    dist_deg_lon = dist_m / lon_m_conv

    return dist_deg_lat, dist_deg_lon


if __name__ == "__main__":
    # parse command line arguments (for running on Borah)
    parser = argparse.ArgumentParser(
        description="Perturb station GPS coordinates in Monte-Carlo simulation to measure uncertainty in beamforming.")
    parser.add_argument("-a", "--array",
                        dest="array_str",
                        type=str,
                        default=None,
                        help="Array String ('-a TOP' for TOP array)")
    parser.add_argument("-f", "--freq", 
                        nargs=2, 
                        dest="freq", 
                        metavar=("freqmin", "freqmax"), 
                        type=float, 
                        help="Min and max corner frequencies for bandpass filter.")
    parser.add_argument("-p", "--paths",
                        dest="paths",
                        type=str,
                        default='borah',
                        help="laptop or borah")
    args = parser.parse_args()

    time_start = UTCDateTime(2023, 10, 5, 0, 0, 0)
    time_stop = UTCDateTime(2023, 10, 8, 0, 0, 0)
    n_iters = 100
    settings.set_paths(location=args.paths)

    data = utils.load_data(settings.path_mseed, 
                           path_coords=settings.path_station_gps, 
                           array_str="TOP", 
                           gem_include=None, 
                           gem_exclude=["TOP32", "TOP07"], 
                           time_start=time_start, time_stop=time_stop, 
                           freqmin=args.freq[0], freqmax=args.freq[1])

    with ProcessPoolExecutor(max_workers=24) as pool:
        # run each iteration within each freq band in parallel
        args_list = [[data, 
                      settings.path_station_gps, 
                      os.path.join(settings.path_processed, "uncert_results"), 
                      args.array_str, 
                      time_start, time_stop, 
                      args.freq[0], args.freq[1], 
                      0.5, i] 
                      for i in range(n_iters)] 
        result = pool.map(main, *zip(*args_list))
    #try running without parallel????
    #for i in range(n_iters):
    #    main(data, settings.path_station_gps, 
    #          os.path.join(settings.path_processed, "uncert_results"), 
    #          args.array_str, time_start, time_stop, 
    #          args.freq[0], args.freq[1], 
    #          0.5, i)

    
