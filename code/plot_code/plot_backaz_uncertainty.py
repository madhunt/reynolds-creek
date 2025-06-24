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

    #TODO need to edit this to deal with JDNA/JDSA a and b files
    
    # (1) load and compile data from all iterations
    time = pd.date_range('2023-10-5', '2023-10-08', freq='30s')
    output_names = ["Semblance", "Abs Power", "Backaz", "Slowness"]
    data_combined = load_beamform_data_all_iters(n_outputs, path_processed)


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
    fig_hist, ax_hist = plt.subplots(nrows=n_outputs, ncols=1, sharex=True, figsize=[12,9])
    fig_bars, ax_bars = plt.subplots(nrows=n_outputs, ncols=1, sharex=True, figsize=[12,9])
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
    fig_hist.tight_layout()
    fig_bars.tight_layout()
    
    # save figures
    #fig_hist.savefig(os.path.join(path_figures, f"{array_str}_{freqmin}-{freqmax}Hz_hist.png"))
    #fig_bars.savefig(os.path.join(path_figures, f"{array_str}_{freqmin}-{freqmax}Hz_bars.png"))

    plt.show()

    plt.close()

    return

def load_beamform_data_all_iters(n_outputs, path_processed):
    """
    Load in all beamforming output results and save in one numpy array.
    INPUTS
        n_outputs       : int       : Number of beamforming outputs (generally, should be 4).
        path_processed  : str       : Path to directory containing all beamforming output files (saved as pkls).
        filename_func   : function  : Function/lambda determining filename formatting, which includes iteration number.
    RETURNS
        time            : np array [len(time)]                      : Array of timestamps in numpy datetime64 format.
        output_names    : np array [n_outputs]                      : Array of string column names of beamforming output.
        data_combined   : np array [n_iters, n_outputs, len(time)]  : Array of float64 values of beamforming results.
    """
    # check if combined file has already been created
    file_save = f"combined_output_{array_str}_{freqmin}-{freqmax}Hz_{gps_perturb_scale}m.npy"
    if os.path.exists(os.path.join(path_processed, file_save)):
        data_combined = np.load(os.path.join(path_processed, file_save))

    else: # create combined file
        # list of data files
        files_all = [f for f in os.listdir(path_processed) 
                     if os.path.isfile(os.path.join(path_processed, f))                 # check if a file (not dir) 
                     and all(i in f for i in [str(freqmin), str(freqmax), array_str])]  # check if for correct freq range and array
        # get number of iterations
        n_iters = max([int(re.findall(r'\d+', f)[-1]) # get last sequence of digits in filename (iter)
                       for f in files_all]) # then take max iter number
        
        # create empty array
        

        data_combined = np.zeros(shape=[n_iters, n_outputs, 8641])
        for n in range(n_iters):

            files = [f for f in files_all                     # get all files where 
                     if int(re.findall(r'\d+', f)[-1]) == n]  # the iteration number matches n
            
            if len(files) == 1: # if only one result per iteration
                file = files[0]
                with open(os.path.join(path_processed, file), mode='rb') as file:
                    output = pickle.load(file)
                    data_combined[n,0,:] = output["Semblance"].values
                    data_combined[n,1,:] = output["Abs Power"].values
                    data_combined[n,2,:] = output["Backaz"].values
                    data_combined[n,3,:] = output["Slowness"].values
            else:   # if multi results per iteration (eg one sensor cut out early)
                with open(os.path.join(path_processed, files[0]), mode='rb') as file:
                    output = pickle.load(file)
                    data_combined[n,0,:len(output)] = output["Semblance"].values
                    data_combined[n,1,:len(output)] = output["Abs Power"].values
                    data_combined[n,2,:len(output)] = output["Backaz"].values
                    data_combined[n,3,:len(output)] = output["Slowness"].values
                with open(os.path.join(path_processed, files[1]), mode='rb') as file:
                    output = pickle.load(file)
                    data_combined[n,0,-len(output):] = output["Semblance"].values
                    data_combined[n,1,-len(output):] = output["Abs Power"].values
                    data_combined[n,2,-len(output):] = output["Backaz"].values
                    data_combined[n,3,-len(output):] = output["Slowness"].values
        # save combined data
        np.save(os.path.join(path_processed, file_save),
                arr=data_combined, allow_pickle=True)

    return data_combined

if __name__ =="__main__":
    # settings
    gps_perturb_scale = 0.5 # m
    #array_list = ["TOP", "JDNA", "JDNB", "JDSA", "JDSB"]
    #array_list = ["JDNB", "JDSB"]
    array_list = ["JDNA"]
    #freq_list = [(0.5, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 16.0), (24.0, 32.0)]
    freq_list = [(4.0, 8.0)]

    settings.set_paths(location='laptop')
    
    for array_str in array_list:
        for freqmin, freqmax in freq_list:
            main(os.path.join(settings.path_processed, "uncert_results", array_str), 
                os.path.join(settings.path_figures, "uncert_results"),
                array_str, freqmin, freqmax, gps_perturb_scale)
