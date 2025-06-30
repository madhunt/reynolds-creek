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

# import files from dir above
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils, settings

def main(path_processed, path_figures, array_str, freqmin, freqmax, gps_perturb_scale):

    n_outputs = 4

    
    # (1) load and compile data from all iterations
    time = pd.date_range('2023-10-5', '2023-10-07T23:59:00', freq='30s')
    output_dict = {"Semblance": [0,1], 
                   "Abs Power": [6, 13], 
                   "Backaz":    [0, 360], 
                   "Slowness":  [0, 5]}
    data_combined = load_beamform_data_all_iters(path_processed, len(output_dict), len(time))


    # scale abs power
    data_combined[:,1,:] = np.log10(data_combined[:,1,:])
    

    # (3) calculate mean and std
    data_mean = np.nanmean(data_combined, axis=0)
    data_std = np.nanstd(data_combined, axis=0)
    # circular statistics for backaz (i=2)
    data_mean[2,:] = sci.stats.circmean(data_combined[:,2,:], 
                                        high=360, low=0, axis=0)
    data_std[2,:] = sci.stats.circstd(data_combined[:,2,:], 
                                      high=360, low=0, axis=0)
    
    fig_bz, ax_bz = plt.subplots(nrows=len(output_dict), ncols=2)
    for i, output_name in enumerate(output_dict):
        ax_bz[i,0].scatter(data_mean[i,:], data_std[i,:],
                        c=[(t-dt.datetime(2023,10,4)).total_seconds() for t in time],
                        cmap='plasma')
        ax_bz[i,0].set_xlabel("Mean")
        ax_bz[i,0].set_ylabel("StDev")
        ax_bz[i,0].set_xlim(output_dict[output_name])
        ax_bz[i,0].set_title(output_name)

        ax_bz[i,1].scatter(time, data_mean[i,:],
                        c=[(t-dt.datetime(2023,10,4)).total_seconds() for t in time],
                        cmap='plasma')
        ax_bz[i,1].sharex(ax_bz[i-1,1])
        ax_bz[i,1].set_ylim(output_dict[output_name])
        ax_bz[i,1].set_title(output_name)

    fig_bz.suptitle(f"{array_str}, {freqmin}-{freqmax}Hz")
    #plt.tight_layout()


    plt.show()

#    # (2) bin the data
#    n_bins = 20
#    bins = np.array([np.linspace(0, 1, n_bins),   # Semblance
#                     np.linspace(7, 14, n_bins),  # Abs Power
#                     np.linspace(0, 360, n_bins), # Backaz
#                     np.linspace(0, 5, n_bins)])  # Slowness
#    
#    data_binned = np.zeros(shape=[n_bins-1, n_outputs, len(time)])
#    for t in range(len(time)):
#        for col in range(n_outputs):
#            hist, b = np.histogram(data_combined[:,col,t], bins=bins[col])
#            data_binned[:,col,t] = hist
#
#
#
#
#    # (4) plot
#    fig_hist, ax_hist = plt.subplots(nrows=n_outputs, ncols=1, sharex=True, figsize=[12,9])
#    fig_bars, ax_bars = plt.subplots(nrows=n_outputs, ncols=1, sharex=True, figsize=[12,9])
#    for i in range(n_outputs):
#        #ax[i].pcolormesh(bins[i,1:], time, data_binned[:,i,:])
#        ax_hist[i].imshow(data_binned[:,i,:],
#                     cmap='Purples',
#                     aspect='auto',
#                     extent=[min(time), max(time), 
#                             min(bins[i]), max(bins[i])])
#        ax_hist[i].set_title(output_names[i])
#
#        ax_bars[i].errorbar(time, data_mean[i,:], yerr=data_std[i,:],
#                            fmt='.', color='k', ecolor='r')
#        ax_bars[i].set_title(output_names[i])
#
#    title = f"{array_str}, {freqmin}-{freqmax}Hz"
#    fig_hist.suptitle(title)
#    fig_bars.suptitle(title)
#    fig_hist.tight_layout()
#    fig_bars.tight_layout()
#    
#    # save figures
#    #fig_hist.savefig(os.path.join(path_figures, f"{array_str}_{freqmin}-{freqmax}Hz_hist.png"))
#    #fig_bars.savefig(os.path.join(path_figures, f"{array_str}_{freqmin}-{freqmax}Hz_bars.png"))
#
#    plt.show()

    plt.close()

    return

def load_beamform_data_all_iters(path_processed, n_outputs, n_time):
    """
    Load in all beamforming output results and save in one numpy array.
    INPUTS
        path_processed  : str       : Path to directory containing all beamforming output files (saved as pkls).
        n_outputs       : int       : Number of beamforming outputs (generally, should be 4).
        n_time          : int       : Number of timestamps
    RETURNS
        time            : np array [len(time)]                      : Array of timestamps in numpy datetime64 format.
        output_names    : np array [n_outputs]                      : Array of string column names of beamforming output.
        data_combined   : np array [n_iters, n_outputs, len(time)]  : Array of float64 values of beamforming results.
    """
    file_save = f"combined_output_{array_str}_{freqmin}-{freqmax}Hz_{gps_perturb_scale}m.npy"
    # check if combined file has already been created
    if os.path.exists(os.path.join(path_processed, file_save)):
        data_combined = np.load(os.path.join(path_processed, file_save))

    # create combined file
    else:
        # list of all data files
        files_all = [f for f in os.listdir(path_processed) 
                     if os.path.isfile(os.path.join(path_processed, f))             # check if a file (not dir) 
                     and all(i in f for i in [f"{freqmin}-{freqmax}", array_str])]  # check if for correct freq range and array
        iter = lambda f: (int(re.findall(r'\d+', f)[-1]), f[-5])    # get iter number and iter char (if applicable) from filename
        files_all.sort(key=iter)                                    # sort list in order from iter 0 to end
        n_iters = max([iter(f)[0] for f in files_all]) + 1          # get max iter num and add 1 for total num of iterations
        
        # create empty array
        data_combined = np.full([n_iters, n_outputs, n_time], np.nan)
        #for n in range(n_iters):
        if n_iters == len(files_all):
            # open nth file in list
            for n in range(n_iters):
                with open(os.path.join(path_processed, files_all[n]), mode='rb') as file:
                    output = pickle.load(file)
                    # save data in data_combined
                    data_combined = get_output_vals(data_combined, output, n, 
                                                    t_idx_start=0, t_idx_stop=n_time-1)
        elif n_iters * 2 == len(files_all):
            for two_n in np.arange(0, n_iters*2, 2):
                # save first half of beamforming results
                print("----------")
                with open(os.path.join(path_processed, files_all[two_n]), mode='rb') as file:
                    output = pickle.load(file)
                    print(two_n)
                    print(files_all[two_n])
                    print(len(output))
                    data_combined = get_output_vals(data_combined, output, int(two_n/2), 
                                                    t_idx_start=0, t_idx_stop=len(output))
                # save second half of results
                with open(os.path.join(path_processed, files_all[two_n+1]), mode='rb') as file:
                    output = pickle.load(file)
                    print(two_n+1)
                    print(files_all[two_n+1])
                    print(len(output))
                    data_combined = get_output_vals(data_combined, output, int(two_n/2), 
                                                    t_idx_start=-len(output), t_idx_stop=n_time)

        else:
            raise AssertionError("Number of files is not correct for number of iterations.")

        # save combined data
        np.save(os.path.join(path_processed, file_save),
                arr=data_combined, allow_pickle=True)

    return data_combined

def get_output_vals(data_combined, output, n, t_idx_start, t_idx_stop):
    data_combined[n,0,t_idx_start:t_idx_stop] = output["Semblance"].values
    data_combined[n,1,t_idx_start:t_idx_stop] = output["Abs Power"].values
    data_combined[n,2,t_idx_start:t_idx_stop] = output["Backaz"].values
    data_combined[n,3,t_idx_start:t_idx_stop] = output["Slowness"].values
    return data_combined



            #files = [f for f in files_all                     # get all files where 
            #         if int(re.findall(r'\d+', f)[-1]) == n]  # the iteration number matches n
            
            #if len(files) == 1: # if only one result per iteration
            #    file = files[0]
            #    with open(os.path.join(path_processed, file), mode='rb') as file:
            #        output = pickle.load(file)
            #        data_combined[n,0,:] = output["Semblance"].values
            #        data_combined[n,1,:] = output["Abs Power"].values
            #        data_combined[n,2,:] = output["Backaz"].values
            #        data_combined[n,3,:] = output["Slowness"].values
            #else:   # if multi results per iteration (eg one sensor cut out early)
            #    with open(os.path.join(path_processed, files[0]), mode='rb') as file:
            #        output = pickle.load(file)
            #        data_combined[n,0,:len(output)] = output["Semblance"].values
            #        data_combined[n,1,:len(output)] = output["Abs Power"].values
            #        data_combined[n,2,:len(output)] = output["Backaz"].values
            #        data_combined[n,3,:len(output)] = output["Slowness"].values
            #    with open(os.path.join(path_processed, files[1]), mode='rb') as file:
            #        output = pickle.load(file)
            #        data_combined[n,0,-len(output):] = output["Semblance"].values
            #        data_combined[n,1,-len(output):] = output["Abs Power"].values
            #        data_combined[n,2,-len(output):] = output["Backaz"].values
            #        data_combined[n,3,-len(output):] = output["Slowness"].values

if __name__ == "__main__":
    # settings
    gps_perturb_scale = 0.5 # m
    #array_list = ["TOP", "JDNA", "JDNB", "JDSA", "JDSB"]
    #array_list = ["JDNB", "JDSB"]
    array_list = ["JDNA"]
    #freq_list = [(0.5, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 16.0), (24.0, 32.0)]
    freq_list = [(24.0, 32.0)]

    settings.set_paths(location='laptop')
    
    for array_str in array_list:
        for freqmin, freqmax in freq_list:
            main(os.path.join(settings.path_processed, "uncert_results", array_str), 
                os.path.join(settings.path_figures, "uncert_results"),
                array_str, freqmin, freqmax, gps_perturb_scale)
