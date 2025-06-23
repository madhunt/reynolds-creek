#!/usr/bin/python3
'''
'''
import os, sys, pickle, re
import pandas as pd
import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# import files from dir above
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils, settings

def main(path_processed, path_figures, array_str, freqmin, freqmax, gps_perturb_scale):

    #TODO change this to grab max iter # in file list +1
    n_iters = 10 
    n_outputs = 4
    
    # (1) load and compile data from all iterations
    filename = lambda n: f"output_{array_str}_{freqmin}-{freqmax}Hz_{gps_perturb_scale}m_iter{n}.pkl"
    time, output_names, data_combined = load_beamform_data_all_iters(n_iters, n_outputs, path_processed, filename)


    #TODO fix this?
    # scale abs power
    data_combined[:,1,:] = np.log10(data_combined[:,1,:])

    # (2) bin the data
    n_bins = 20
    bins = np.array([np.linspace(0, 1, n_bins),   # Semblance
                     np.linspace(7, 14, n_bins),  # Abs Power
                     np.linspace(0, 360, n_bins), # Backaz
                     np.linspace(0, 5, n_bins)])  # Slowness
    
    data_binned = np.zeros(shape=[n_bins-1, n_outputs, len(time)])
    for t in range(len(time)):
        for col in range(n_outputs):
            hist, b = np.histogram(data_combined[:,col,t], bins=bins[col])
            data_binned[:,col,t] = hist

    # (3) calculate mean and std
    data_mean = data_combined.mean(axis=0)
    data_std = data_combined.std(axis=0)
    # circular statistics for backaz (i=2)
    data_mean[2,:] = sci.stats.circmean(data_combined[:,2,:], 
                                        high=360, low=0, axis=0)
    data_std[2,:] = sci.stats.circstd(data_combined[:,2,:], 
                                      high=360, low=0, axis=0)

    # (4) plot
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

    title = f"{array_str}, {freqmin}-{freqmax}Hz"
    fig_hist.suptitle(title)
    fig_bars.suptitle(title)
    
    # save figures
    fig_hist.savefig(os.path.join(path_figures, f"{array_str}_{freqmin}-{freqmax}Hz_hist.png"))
    fig_bars.savefig(os.path.join(path_figures, f"{array_str}_{freqmin}-{freqmax}Hz_bars.png"))

    return

def load_beamform_data_all_iters(n_iters, n_outputs, path_processed, filename_func):
    """
    Load in all beamforming output results and save in one numpy array.
    INPUTS
        n_iters         : int       : Number of iterations (corresponds with number of files in path_processed).
        n_outputs       : int       : Number of beamforming outputs (generally, should be 4).
        path_processed  : str       : Path to directory containing all beamforming output files (saved as pkls).
        filename_func   : function  : Function/lambda determining filename formatting, which includes iteration number.
    RETURNS
        time            : np array [len(time)]                      : Array of timestamps in numpy datetime64 format.
        output_names    : np array [n_outputs]                      : Array of string column names of beamforming output.
        data_combined   : np array [n_iters, n_outputs, len(time)]  : Array of float64 values of beamforming results.
    """
    # save column names and time index
    output0 = pd.read_pickle(os.path.join(path_processed, filename_func(0)))
    output_names = output0.columns.values
    time = output0.index.values

    # read in values from all data iterations
    data_combined = np.zeros(shape=[n_iters, n_outputs, len(time)])
    for n in range(n_iters):
        with open(os.path.join(path_processed, filename_func(n)), mode='rb') as file:
            output = pickle.load(file)
            data_combined[n,0,:] = output["Semblance"].values
            data_combined[n,1,:] = output["Abs Power"].values
            data_combined[n,2,:] = output["Backaz"].values
            data_combined[n,3,:] = output["Slowness"].values
            
    # save combined data
    file_save = re.sub(r"_iter\d", "", filename_func(0))
    np.save(os.path.join(path_processed, file_save),
            arr=data_combined, allow_pickle=True)

    return time, output_names, data_combined

if __name__ =="__main__":
    # settings
    gps_perturb_scale = 0.5 # m
    array_list = ["TOP", "JDNA", "JDNB", "JDSA", "JDSB"]
    freq_list = [(0.5, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 16.0), (24.0, 32.0)]

    settings.set_paths(location='borah')
    
    for array_str in array_list:
        for freqmin, freqmax in freq_list:
            main(os.path.join(settings.path_processed, "uncert_results", array_str), 
                settings.path_figures,
                array_str, freqmin, freqmax, gps_perturb_scale)
