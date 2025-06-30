#!/usr/bin/python3
'''
'''
import os, sys, pickle, re
import pandas as pd
import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import glob

# import files from dir above
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils, settings

def main(path_processed, path_station_gps, path_figures, array_str):

    # (1) set up figure
    fig, ax = plt.subplots(nrows=5, ncols=3, figsize=[12,10])

    freq_list = [(0.5, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 16.0), (24.0, 32.0)]
    gps_perturb_scale = 0.5


    # (2) plot array response
    #TODO
    # plot lines overlaid at 45deg angles


    # (3a) get combined beamforming results
    time = pd.date_range('2023-10-5', '2023-10-07T23:59:00', freq='30s')
    #output_dict = {"Semblance": [0,1], 
    #               "Abs Power": [6, 13], 
    #               "Backaz":    [0, 360], 
    #               "Slowness":  [0, 5]}

    
    for i, (freqmin, freqmax) in enumerate(freq_list):
    
        # load in combined results for freq band
        data_combined = np.load(os.path.join(path_processed, 
                                             f"combined_output_{array_str}_{freqmin}-{freqmax}Hz_{gps_perturb_scale}m.npy"))
        # scale abs power
        data_combined[:,1,:] = np.log10(data_combined[:,1,:])

        # filter out bad slownesses and replace values with nans
        slowness = data_combined[:,3,:]
        mask = (slowness >= 2) & (slowness <= 3.5)
        mask = mask[:, np.newaxis, :]
        mask_broad = np.broadcast_to(mask, data_combined.shape)
        data_combined[~mask_broad] = np.nan

        # (3b) calculate mean/std
        data_mean = np.nanmean(data_combined, axis=0)
        data_std = np.nanstd(data_combined, axis=0)
        # circular statistics for backaz (i=2)
        data_mean[2,:] = sci.stats.circmean(data_combined[:,2,:], 
                                            high=360, low=0, axis=0,
                                            nan_policy='omit')
        data_std[2,:] = sci.stats.circstd(data_combined[:,2,:], 
                                          high=360, low=0, axis=0,
                                            nan_policy='omit')
        
        # plot backaz mean/std
        ax[i,0].scatter(data_mean[2,:], data_std[2,:],
                        c=[(t-dt.datetime(2023,10,4)).total_seconds() for t in time],
                        cmap='viridis')
        ax[i,0].set_xticks([0,90,180,270,360])
        ax[i,0].sharex(ax[i-1,0])
        ax[i,0].sharey(ax[i-1,0])
        ax[i,0].set_xlabel("Mean")
        ax[i,0].set_ylabel("StDev")
        ax[i,0].set_title(f"Backaz {freqmin}-{freqmax}Hz")

        # plot backaz mean/std
        ax[i,1].scatter(data_mean[2,:], data_std[2,:],
                       c=[(t-dt.datetime(2023,10,4)).total_seconds() for t in time],
                       cmap='viridis')
        ax[i,1].set_xticks([0,90,180,270,360])
        ax[i,1].set_ylim([-5, 20])
        ax[i,1].sharex(ax[i-1,1])
        ax[i,1].sharey(ax[i-1,1])
        ax[i,1].set_xlabel("Mean")
        ax[i,1].set_ylabel("StDev")
        ax[i,1].set_title(f"Backaz Zoom")



        # plot slowness mean/std
        #ax[1,i+1].plot(data_mean[3,:], data_std[3,:], 'k.')
        ax[i,2].scatter(data_mean[2,:], data_std[3,:],
                       c=[(t-dt.datetime(2023,10,4)).total_seconds() for t in time],
                       cmap='viridis')
        ax[i,2].sharex(ax[i-1,2])
        ax[i,2].sharey(ax[i-1,2])
        #ax[i,2].set_xlim([0,6])
        ax[i,1].set_xticks([0,90,180,270,360])
        ax[i,2].set_xlabel("Backaz Mean")
        ax[i,2].set_ylabel("Slowness StDev")
        ax[i,2].set_title(f"Slowness StDev vs Backaz Mean {freqmin}-{freqmax}Hz")

        # (3c) plot backaz mean/std









    fig.suptitle(array_str)
    fig.tight_layout()
    fig.savefig(os.path.join(path_figures, "mean_std", f"{array_str}_removed.png"))
    #plt.show()

    plt.close()

    return


if __name__ =="__main__":
    # settings
    #array_list = ["TOP", "JDNA", "JDNB", "JDSA", "JDSB"]
    array_list = ["JDNA", "JDNB", "JDSA", "JDSB"]
    #array_list = ["JDNA"]

    settings.set_paths(location='laptop')
    
    for array_str in array_list:
        main(path_processed=os.path.join(settings.path_processed, "uncert_results", array_str), 
             path_station_gps=settings.path_station_gps,
             path_figures=os.path.join(settings.path_figures, "uncert_results"),
             array_str=array_str)
