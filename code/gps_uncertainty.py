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

import utils, settings, beamform

def main(path_mseed, path_station_gps, path_save,
         array_str, time_start, time_stop,
         freqmin, freqmax,
         gps_perturb_scale, iter):

    # LOG: record processing start time
    if iter == 0:
        # create logfile ("w")
        with open(os.path.join(path_save, f"pylog_{array_str}_{freqmin}-{freqmax}Hz_{gps_perturb_scale}m.txt"), "w") as f:
            print(f"{datetime.datetime.now()} \t\t {iter}: Started Processing", file=f)
    else:
        # record in existing file
        with open(os.path.join(path_save, f"pylog_{array_str}_{freqmin}-{freqmax}Hz_{gps_perturb_scale}m.txt"), "a") as f:
            print(f"{datetime.datetime.now()} \t\t {iter}: Started Processing", file=f)


    # (1) perturb coordinates --------------------- ---------------------------------- -----------------------------------
    # read in coords
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

    # (2) load in raw data with perturbed coordinates -------------------------------------------------------------------
    data = utils.load_data(path_mseed, path_coords=coords, array_str=array_str,
            gem_include=None, gem_exclude=None,
            time_start=time_start, time_stop=time_stop,
            freqmin=freqmin, freqmax=freqmax)

    # (3) perform beamforming -----------------------------------
    # define path to save resulting output
    path_processed = os.path.join(path_save, f"output_{array_str}_{freqmin}-{freqmax}Hz_{gps_perturb_scale}m_iter{iter}.pkl")
    output = beamform.process_data(data, path_processed=path_processed, 
                                    time_start=time_start, time_stop=time_stop,
                                    freqmin=freqmin, freqmax=freqmax)
    
    # LOG: record processing end time
    with open(os.path.join(path_save, f"pylog_{array_str}_{freqmin}-{freqmax}Hz_{gps_perturb_scale}m.txt"), "a") as f:
        print(f"{datetime.datetime.now()} \t\t {iter}: Finished Processing", file=f)

    ## (4) update mean/standard deviation -----------------------------------
    #if i == 0:
    #    # set initial values for first iteration (mean=0, M2=0)
    #    agg_col_names = ["Semblance Mean", "Abs Power Mean", "Backaz Mean", "Slowness Mean",
    #                        "Semblance M2", "Abs Power M2", "Backaz M2", "Slowness M2", "N"]
    #    data_init = np.zeros(shape=(len(output), len(agg_col_names)))
    #    output_aggregate = pd.DataFrame(data_init, columns=agg_col_names, index=output.index)
    
    #for col in output.columns:
    #    n = i+1     # num of datapoints is 1 greater than iteration number
    #    val = output[col]
    #    mean_old = output_aggregate[col+" Mean"]
    #    M2_old = output_aggregate[col+" M2"]

    #    # update using Welford's Algorithm
    #    mean_new = mean_old + (1/n)*(val - mean_old)
    #    M2_new = M2_old + (val - mean_old)*(val - mean_new)

    #    # update to save new mean and M2
    #    output_aggregate[col+" Mean"] = mean_new
    #    output_aggregate[col+" M2"] = M2_new
    ## save number of data points
    #output_aggregate["N"] = n
    
    ## log iteration number
    #with open(os.path.join(path_save, f"pylog_{array_str}_{freqmin}-{freqmax}Hz_{gps_perturb_scale}m.txt"), "a") as f:
    #    print(f"\t\t Iter: {i}, Num: {n}", file=f)

    ## save updated mean/M2 in file (will be overwritten each loop)
    #output_aggregate.to_pickle(os.path.join(path_save, 
    #                                        f"output_aggregate_{array_str}_{freqmin}-{freqmax}Hz_{gps_perturb_scale}m.pkl"))
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
    parser.add_argument("-g", "--gps-scale",
                        dest="gps_perturb_scale",
                        type=float,
                        help="GPS perturbation scale in meters ('-g 0.2' for 0.2 m perturbations)")

    args = parser.parse_args()

    # arguments that are the same for every run (specific to Borah)
    path_mseed = os.path.join("/", "bsuhome", "madelinehunt", "reynolds-creek", "data", "mseed")
    path_station_gps = os.path.join("/", "bsuhome", "madelinehunt", "reynolds-creek", "data", "gps")
    path_save = os.path.join("/", "bsuhome", "madelinehunt", "reynolds-creek", "data", "processed", "uncert_results")
    
    time_start = UTCDateTime(2023, 10, 5, 0, 0, 0)
    time_stop = UTCDateTime(2023, 10, 8, 0, 0, 0)

    freq_list = [(0.5, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 16.0), (24.0, 32.0)]
    n_iters = 1000

    with ProcessPoolExecutor(max_workers=48) as pool:
        # run each iteration within each freq band in parallel
        args_list = [[path_mseed, path_station_gps, path_save,
                     args.array_str, time_start, time_stop, 
                     freqmin, freqmax,
                     args.gps_perturb_scale, i]
                     for freqmin,freqmax in freq_list for i in range(n_iters)]
        result = pool.map(main, *zip(*args_list))

    #with open(os.path.join(path_save, f"pylog_{args.array_str}_{freqmin}-{freqmin}Hz_{args.gps_perturb_scale}m.txt"), "a") as f:
    #    print(f"{datetime.datetime.now()} \t\t Completed Processing", file=f)

    
