#!/usr/bin/python3
'''
'''
import os, datetime, pytz, sys, pickle
import pandas as pd
import numpy as np
import scipy as sci
import glob
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# import files from dir above
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils, settings

def main(path_processed, path_heli, path_station_gps, path_figures,
         freqmin, freqmax, gps_perturb_scale):

    array_str = "JDNB"
    #TODO chancge this to grab max iter # in file list +1
    n_iters = 10 
    
    n_outputs = 4
    

    # (1) load and compile data from all iterations
    path_func = lambda n: os.path.join(path_processed, f"output_{array_str}_{freqmin}-{freqmax}Hz_{gps_perturb_scale}m_iter{n}.pkl")
    output0 = pd.read_pickle(path_func(n=0))
    output_names = output0.columns.to_numpy()
    time = output0.index.to_numpy()             # get time index
    data_combined = load_beamform_data_all_iters(n_iters, time, path_func, n_outputs)

    #TODO fix this?
    # scale abs power
    data_combined[:,1,:] = np.log10(data_combined[:,1,:])

    # (2a) bin the data
    n_bins = 20
    bins = np.array([np.linspace(0, 1, n_bins),   # Semblance
                     np.linspace(7, 14, n_bins),  # Abs Power
                     np.linspace(0, 360, n_bins),   # Backaz
                     np.linspace(0, 5, n_bins)])  # Slowness
    
    data_binned = np.zeros(shape=[n_bins-1, n_outputs, len(time)])
    for t in range(len(time)):
        for col in range(n_outputs):
            hist, b = np.histogram(data_combined[:,col,t], bins=bins[col])
            data_binned[:,col,t] = hist

    # (2b) calculate mean and std
    data_mean = data_combined.mean(axis=0)
    data_std = data_combined.std(axis=0)
    # circular statistics for backaz (i=2)
    data_mean[2,:] = sci.stats.circmean(data_combined[:,2,:], 
                                        high=360, low=0, axis=0)
    data_std[2,:] = sci.stats.circstd(data_combined[:,2,:], 
                                      high=360, low=0, axis=0)

    # (3) plot
    fig_hist, ax_hist = plt.subplots(nrows=n_outputs, ncols=1, sharex=True)
    fig_bars, ax_bars = plt.subplots(nrows=n_outputs, ncols=1, sharex=True)
    for i in range(n_outputs):
        #ax[i].pcolormesh(bins[i,1:], time, data_binned[:,i,:])
        ax_hist[i].imshow(data_binned[:,i,:],
                     cmap='Purples',
                     aspect='auto',
                     extent=[min(time), max(time), 
                             min(bins[i]), max(bins[i])])
        ax_hist[i].set_title(output_names[i])


        ax_bars[i].errorbar(time, data_mean[i,:], yerr=data_std[i,:],
                            fmt='.', color='k', ecolor='r')
        ax_bars[i].set_title(output_names[i])
    

    plt.show()




    #plt.savefig(os.path.join(path_figures, f"backaz_uncertainty_{gps_perturb_scale}m.png"), dpi=500)
    #plt.close()



    return



def load_beamform_data_all_iters(n_iters, time, path_func, n_outputs):
    """
    
    RETURNS
        data_combined : np array [n_iters, 4, len(time)]
    """
    # read in values from all data iterations
    data_combined = np.zeros(shape=[n_iters, n_outputs, len(time)])
    for n in range(n_iters):
        with open(path_func(n), mode='rb') as file:
            output = pickle.load(file)
            data_combined[n,0,:] = output["Semblance"].values
            data_combined[n,1,:] = output["Abs Power"].values
            data_combined[n,2,:] = output["Backaz"].values
            data_combined[n,3,:] = output["Slowness"].values
    return data_combined

if __name__ =="__main__":
    # settings
    freqmin = 4.0
    freqmax = 8.0
    gps_perturb_scale = 0.5 # m

    main(os.path.join(settings.path_processed, "uncert_results"), 
         settings.path_heli,
         settings.path_station_gps,
         settings.path_figures,
         freqmin, freqmax, gps_perturb_scale)
