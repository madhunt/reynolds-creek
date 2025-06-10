#!/usr/bin/python3

import obspy.signal
import obspy.signal.array_analysis
import os, sys, datetime, pytz, glob, utm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.cm as cm
import matplotlib.dates as mdates
import obspy

# import files from dir above
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils, settings, crossbeam

def main(path_station_gps):

    # read in DEM of region
    # USGS DEM of Kilauea https://data.usgs.gov/datacatalog/data/USGS:5eb19f3882cefae35a29c363
    # use dem to get zg (see below)

    # create meshgrid of all locations
    #TODO jeff uses stepsize 25 at line 47
    northing = np.arange(4771000, 4777000, step=25)
    easting = np.arange(515000, 520000, step=25)
    [xg, yg] = np.meshgrid(easting, northing)
    #zg = dem # line 61

    fig_net, ax_net = plt.subplots(1, 1)
    ax_net.set_aspect('equal')
    ax_net.grid(visible=True)
    ax_net.set_xlim(515000, 520000)
    ax_net.set_ylim(4771000, 4777000)
    ax_net.set_xlabel("Easting (m)")
    ax_net.set_ylabel("Northing (m)")

    fig_arr, ax_arr = plt.subplots(2, 5)

    #TODO add contour lines from DEM to fig_net (line 66)

    arr_list = ["TOP", "JDNA", "JDNB", "JDSA", "JDSB"]
    arr_cols = ['red', 'orange', 'yellow', 'green', 'blue']
    # read in station coords
    path_coords = glob.glob(os.path.join(path_station_gps, "*.csv"))[0]
    coords = pd.read_csv(path_coords)
    # convert coordinates to easting/northing
    coords[["Easting", "Northing", "Zone Num", "Zone Lett"]] = coords.apply(lambda x: utm.from_latlon(x["Latitude"], x["Longitude"]), axis=1).to_list()
    # only save coords for arrays we are using
    pattern = '|'.join(arr_list)
    coords = coords[coords["Station"].str.contains(pattern)]
    coords = coords.reset_index()
    
    for i, arr_str in enumerate(arr_list):
        coords_sub = coords[coords["Station"].str.contains(arr_str)]

        # plot all stations
        ax_arr[0,i].plot(coords_sub["Easting"], coords_sub["Northing"], 'k^', markerfacecolor=arr_cols[i])
        ax_net.plot(coords_sub["Easting"], coords_sub["Northing"], 'k^', markerfacecolor=arr_cols[i])

        # add station labels
        #TODO make this neater for all subplots
        ax_net.text(coords_sub["Easting"].max(), coords_sub["Northing"].min(), s=arr_str, verticalalignment='top', horizontalalignment='left')

        center = [coords_sub["Easting"].mean(), coords_sub["Northing"].mean()]
        if "JD" in arr_str:
            ax_arr[0,i].set_xlim(center[0]+[-10,10])
            ax_arr[0,i].set_ylim(center[1]+[-10,10])
        ax_arr[0,i].set_aspect('equal')
        ax_arr[0,i].grid(True)
        ax_arr[0,i].set_title(arr_str)

        # add array element labels
        for _, row in coords_sub.iterrows():
            ax_arr[0,i].text(row["Easting"], row["Northing"],
                             s=row["Station"], 
                             verticalalignment='top', horizontalalignment='left')


        ax_arr[1,i].set_aspect('equal')
        # calculate and plot array response
        #test_coords = coords_sub[["Easting", "Northing"]].to_numpy()
        #test_coords = test_coords - center
        #kxs, kys, K = array_response(test_coords, n_wavenum=101, wavelen_max=15)        
        #TODO add z coord to these
        test_coords = coords_sub[["Longitude", "Latitude"]].to_numpy()
        test_coords = np.insert(test_coords, 2, 0, axis=1)



        # choose wavenum at 24Hz in rad/km as per Jake eq code
        klim = 2 * np.pi * 24 / 0.330   
        kstep = 1
        array_response = obspy.signal.array_analysis.array_transff_wavenumber(test_coords,
                                                                              klim=klim,
                                                                              kstep=kstep,
                                                                              coordsys='lonlat')
        lim = klim #/100
        zlim = [0,1]
        res = ax_arr[1,i].imshow(array_response.transpose(), 
                           extent=[-lim, lim, -lim, lim],
                           origin='lower',
                           vmin=zlim[0], vmax=zlim[1],
                           cmap='plasma')
        cbar = fig_arr.colorbar(res, ax=ax_arr[1,i])
        cbar.set_label("Semblance")

        circle_freq = [4, 8, 24]
        for f in circle_freq:
            r = 2*np.pi*f/0.333
            theta = np.linspace(0, 2*np.pi, 100)
            ax_arr[1,i].plot(r*np.sin(theta), r*np.cos(theta), 'k--')

        ax_arr[1,i].set_xlabel("$k_x$ (rad/km)")
        ax_arr[1,i].set_ylabel("$k_y$ (rad/km)")

        #ax_arr[1,i].imshow(np.abs(K), extent=[np.min(kxs), np.max(kxs), np.min(kys), np.max(kys)])
        #ax_arr[1,i].set_xlim()
        # plot circles on top
        #ax_arr[1,i].plot()
    
    # calculate network response
    # time of flight matricies
    dists = np.empty(shape=[np.shape(xg)[0], np.shape(yg)[1], len(coords)])
    for k in range(len(coords)):
        x = coords["Easting"][k]
        y = coords["Northing"][k]
        #TODO add z coord to this, too
        dists[:,:,k] = np.sqrt((x-xg)**2 + (y-yg)**2)
        #NOTE coords is in order of JDNA, JDNB, JDSA, JDSB, TOP

    #subsample meshgrid
    xgdec = xg[25::25, 25::25]
    ygdec = yg[25::25, 25::25]
    ax_net.plot(xgdec, ygdec, "ko", markerfacecolor=[1,0.5,0.5])
    for k in range(np.size(xgdec)):
        print(f"k={k} of {np.size(xgdec)}")
        # matlab uses linear indexing so need to flatten by columns here
        xtmp = xgdec.ravel(order='F')[k]
        ytmp = ygdec.ravel(order='F')[k]
        #ztmp = dem.ravel(order='F')[k]

        idxy = np.where(np.isin(northing, ytmp))
        idxx = np.where(np.isin(easting, xtmp))
        
        # length of dtmp is 60 bc there are 60 sensors (same len as coords)
        dtmp = np.squeeze(dists[idxy, idxx, :])

        # JDNA idx 0, 1, 2, 3
        dtmp_jdna = np.diff(dtmp[:4])
        dtmp_jdna = np.insert(dtmp_jdna, obj=0, values=(dtmp[3]-dtmp[0])) # handle first row (last minus first)
        restmp_jdna = np.diff(dists[:,:,:4], axis=2)
        restmp_jdna = np.insert(restmp_jdna, obj=0, axis=2, values=(dists[:,:,3]-dists[:,:,0]))
        # JDNB idx 4, 5, 6, 7
        dtmp_jdnb = np.diff(dtmp[4:8])
        dtmp_jdnb = np.insert(dtmp_jdnb, obj=0, values=(dtmp[7]-dtmp[4]))
        restmp_jdnb = np.diff(dists[:,:,4:8], axis=2)
        restmp_jdnb = np.insert(restmp_jdnb, obj=0, axis=2, values=(dists[:,:,7]-dists[:,:,4]))
        # jdsa idx 8, 9, 10, 11
        dtmp_jdsa = np.diff(dtmp[8:12])
        dtmp_jdsa = np.insert(dtmp_jdsa, obj=0, values=(dtmp[11]-dtmp[8]))
        restmp_jdsa = np.diff(dists[:,:,8:12], axis=2)
        restmp_jdsa = np.insert(restmp_jdsa, obj=0, axis=2, values=(dists[:,:,11]-dists[:,:,8]))
        # jdsb idx 12, 13, 14, 15
        dtmp_jdsb = np.diff(dtmp[12:16])
        dtmp_jdsb = np.insert(dtmp_jdsb, obj=0, values=(dtmp[15]-dtmp[12]))
        restmp_jdsb = np.diff(dists[:,:,12:16], axis=2)
        restmp_jdsb = np.insert(restmp_jdsb, obj=0, axis=2, values=(dists[:,:,15]-dists[:,:,12]))
        # top idx 16, 17, ... 59
        dtmp_top = np.diff(dtmp[16:])
        dtmp_top = np.insert(dtmp_top, obj=0, values=(dtmp[-1]-dtmp[16]))
        restmp_top = np.diff(dists[:,:,16:], axis=2)
        restmp_top = np.insert(restmp_top, obj=0, axis=2, values=(dists[:,:,-1]-dists[:,:,16]))

        # combine all residuals into one matrix
        drestmp = np.concatenate((restmp_jdna, restmp_jdnb, restmp_jdsa,
                                           restmp_jdsb, restmp_top), axis=2)
        # distance residual (square and sum)
        drestmp = np.sum(drestmp**2, axis=2)
        trestmp = drestmp / 343


        # add contour lines to plot
    #TODO note originally in for loop -- how to get different lines for different k vals???
    contour = ax_net.contour(easting, northing, trestmp, cmap='viridis')
    cbar_net = fig_net.colorbar(contour)









    
    plt.show()



    return

def array_response(coords, n_wavenum, wavelen_max=15):

    kxs = np.linspace(-2*np.pi/wavelen_max, 2*np.pi/wavelen_max, n_wavenum)
    kys = np.linspace(-2*np.pi/wavelen_max, 2*np.pi/wavelen_max, n_wavenum)
    [kxgs, kygs] = np.meshgrid(kxs, kys)

    channel_num = np.shape(coords)[0]


    KS = np.empty(shape=[len(kxgs), len(kygs), channel_num])
    for k in range(0, channel_num):
        KS[:,:,k] = np.exp(1j * (coords[k][0]*kxgs) + (coords[k][1]*kygs))

        Knum = (np.sum(KS, axis=2))**2
        Kden = KS[:,:,0]

    for k in range(1, channel_num):
        Kden = Kden * KS[:,:,k]

    K = Knum / Kden / channel_num**2


    return kxs, kys, K



if __name__ == "__main__":



    main(settings.path_station_gps)

