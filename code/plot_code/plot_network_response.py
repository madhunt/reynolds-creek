#!/usr/bin/python3

import obspy.signal
import obspy.signal.array_analysis
import rasterio.transform
import rasterio.warp
import rasterio.windows
import os, sys, datetime, pytz, glob, utm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.cm as cm
import matplotlib.dates as mdates
import obspy
from scipy.interpolate import RegularGridInterpolator
import rasterio, pyproj

# import files from dir above
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils, settings, crossbeam

def main(path_station_gps, path_dem, path_figures):
    
    # define RCEW bounds in UTM coords
    bounds_reg = [515000, 520000, 4771000, 4777000] # min_east, max_east, min_north, max_north

    # create figure for network response
    fig_net, ax_net = plt.subplots(1, 1)
    ax_net.set_aspect('equal')
    ax_net.grid(visible=True)
    ax_net.set_xlim(bounds_reg[0], bounds_reg[1])
    ax_net.set_ylim(bounds_reg[2], bounds_reg[3])
    ax_net.set_xlabel("Easting (m)")
    ax_net.set_ylabel("Northing (m)")

    # read in DEM
    easting_dem, northing_dem, elev_dem = read_dem(path_dem, bounds_reg)
    # create 20 m contour intervals
    contour_levels = np.arange(np.floor(np.nanmin(elev_dem)/20)*20,
                               np.ceil(np.nanmax(elev_dem)/20)*20+20, 20)
    # add contours to figure
    ax_net.contour(easting_dem, northing_dem, elev_dem, 
                   levels=contour_levels,
                   colors='grey', linewidths=0.8)
    
    # list of arrays and colors
    arr_list = ["TOP", "JDNA", "JDNB", "JDSA", "JDSB"]
    arr_cols = ['red', 'orange', 'yellow', 'green', 'blue']
    # read in station coordinates
    coords = get_station_coords(path_station_gps, arr_list)

    # plot all stations on figure
    for i, arr_str in enumerate(arr_list):
        coords_sub = coords[coords["Station"].str.contains(arr_str)]
        ax_net.plot(coords_sub["Easting"], coords_sub["Northing"], 'k^', markerfacecolor=arr_cols[i])
        # add array labels
        ax_net.text(coords_sub["Easting"].max(), coords_sub["Northing"].min(), 
                    s=arr_str, verticalalignment='top', horizontalalignment='left')
    
    

    # create meshgrid of all locations for network response
    easting = np.arange(bounds_reg[0], bounds_reg[1], step=25)
    northing = np.arange(bounds_reg[2], bounds_reg[3], step=25)
    [xg, yg] = np.meshgrid(easting, northing)

    # get DEM elevations on same meshgrid
    interpolator = RegularGridInterpolator(points=(northing_dem[:,1], easting_dem[0,:]),
                                           values=elev_dem,
                                           method='linear',
                                           bounds_error=False, fill_value=np.nan)
    zg = interpolator(np.column_stack([yg.ravel(), xg.ravel()]))
    zg = zg.reshape(xg.shape)



    # time of flight matricies    
    dists = np.empty(shape=[np.shape(xg)[0], np.shape(yg)[1], len(coords)])
    for k in range(len(coords)):
        #NOTE coords is in order of JDNA, JDNB, JDSA, JDSB, TOP
        x = coords["Easting"][k]
        y = coords["Northing"][k]
        z = coords["Elevation"][k]
        dists[:,:,k] = np.sqrt((x-xg)**2 + (y-yg)**2 + (z-zg)**2)

    #subsample meshgrid
    sp = 25
    xgdec = xg[25::sp, 25::sp]
    ygdec = yg[25::sp, 25::sp]
    colors_net = plt.get_cmap('rainbow')(np.linspace(0,1,np.size(xgdec)))
    ax_net.plot(xgdec, ygdec, "ko", markerfacecolor=[1,0.5,0.5])   # jeff default color
    #ax_net.scatter(xgdec, ygdec, c=colors_net)

    
    for k in range(np.size(xgdec)):
        print(f"k={k} of {np.size(xgdec)}")
        # matlab uses linear indexing so need to flatten by columns here
        xtmp = xgdec.ravel(order='F')[k]
        ytmp = ygdec.ravel(order='F')[k]

        idxx = np.where(np.isin(easting, xtmp))
        idxy = np.where(np.isin(northing, ytmp))
        
        # length of dtmp is 60 bc there are 60 sensors (same len as coords)
        dtmp = np.squeeze(dists[idxy, idxx, :])

        # JDNA (idx 0, 1, 2, 3)
        def calc_restmp(dtmp, dists):
            dtmp01 = dtmp[0] - dtmp[1]
            restmp01 = np.abs(dtmp01 - (dists[:,:,0] - dists[:,:,1]))
            dtmp12 = dtmp[1] - dtmp[2]
            restmp12 = np.abs(dtmp12 - (dists[:,:,1] - dists[:,:,2]))
            dtmp23 = dtmp[2] - dtmp[3]
            restmp23 = np.abs(dtmp23 - (dists[:,:,2] - dists[:,:,3]))
            dtmp30 = dtmp[3] - dtmp[0]
            restmp30 = np.abs(dtmp30 - (dists[:,:,3] - dists[:,:,0]))
            restmp_array = restmp01**2 + restmp12**2 + restmp23**2 + restmp30**2
            return restmp_array

        restmp_jdna = calc_restmp(dtmp[0:4], dists[:,:,0:4])
        restmp_jdnb = calc_restmp(dtmp[4:8], dists[:,:,4:8])
        restmp_jdsa = calc_restmp(dtmp[8:12], dists[:,:,8:12])
        restmp_jdsb = calc_restmp(dtmp[12:16], dists[:,:,12:16])

        # top idx 16, 17, ... 59
        dtmp_top = -1*np.diff(dtmp[16:])
        dtmp_top = np.append(dtmp_top, values=(dtmp[-1]-dtmp[16]))
        dists_top = dists[:,:,16:]
        restmp_top = np.zeros(shape=[dists_top.shape[0], dists_top.shape[1]])
        for n in range(len(dtmp_top)):
            dtmpnn = dtmp_top[n]
            if n+1 == len(dtmp_top):
                # account for last row
                restmpnn = np.abs(dtmpnn - (dists_top[:,:,n] - dists_top[:,:,0]))
            else:
                restmpnn = np.abs(dtmpnn - (dists_top[:,:,n] - dists_top[:,:,n+1]))
            restmp_top += restmpnn**2
        

        #drestmp = np.sqrt(restmp_jdna/16 + restmp_jdnb/16 +
        #                  restmp_jdsa/16 + restmp_jdsb/16 + 
        #                  restmp_top/44**2)
        drestmp = np.sqrt(restmp_jdna + restmp_jdnb +
                          restmp_jdsa + restmp_jdsb + 
                          restmp_top)
        #drestmp = np.sqrt(restmp_jdna + restmp_jdnb + 
        #                  restmp_jdsa + restmp_jdsb)
        trestmp = drestmp / 343


        print(np.nanmin(trestmp), np.nanmax(trestmp))
        #ax_net.plot(xtmp, ytmp, "ko", markerfacecolor=colors_net[k])
        contour = ax_net.contour(easting, northing, trestmp, 
                                 #levels=np.linspace(trestmp.min(), trestmp.max(), 3),
                                 #NOTE change these (default is jeff's)
                                 levels=np.arange(0.005, 0.015, 0.00125),
                                 #levels=np.arange(0.001, 0.004, 0.001),
                                 #levels=np.arange(0.007, 0.011, 0.001),
                                 #levels=np.linspace(np.nanmin(trestmp), np.nanmin(trestmp)+0.05, 3),
                                 colors=[(1, 0.5, 0.5)])
                                 #colors=[colors_net[k]])
                                 #cmap='viridis')
    #cbar_net = fig_net.colorbar(contour)

    fig_arr = plot_array_response(path_figures, path_station_gps)

    plt.show()



    return

def get_station_coords(path_station_gps, arr_list):
    # read in station coords
    path_coords = glob.glob(os.path.join(path_station_gps, "*.csv"))[0]
    coords = pd.read_csv(path_coords)
    # convert coordinates to easting/northing
    coords[["Easting", "Northing", "Zone Num", "Zone Lett"]] = coords.apply(lambda x: utm.from_latlon(x["Latitude"], x["Longitude"]), axis=1).to_list()
    # only save coords for arrays we are using
    pattern = '|'.join(arr_list)
    coords = coords[coords["Station"].str.contains(pattern)]
    coords = coords.reset_index()
    return coords

def read_dem(path_dem, bounds_reg):
    '''
    
    INPUTS
        path_dem    : str           : Path to DEM .tif file that includes region of interest.
        bounds_reg  : list of float : UTM boundary of region of interest within DEM. 
            In UTM easting/northing as [min_easting, max_easting, min_northing, max_northing]
    RETURNS
        easting     : numpy array   : Meshgrid of easting coordinates in UTM where elev is sampled.
        northing    : numpy array   : Meshgrid of northing coordinates in UTM where elev is sampled.
        elev        : numpy array   : DEM elevation cropped to region of interest and reprojected in 
            UTM coordinate reference system. Same shape as easting and northing arrays.

    '''
    with rasterio.open(path_dem) as src:
        # get coordinate reference system (CRS) in UTM coords
        bounds_dem = src.bounds
        center_lon = (bounds_dem.left + bounds_dem.right) / 2
        center_lat = (bounds_dem.bottom + bounds_dem.top) / 2
        utm_zone = int((center_lon+180)/6)+1
        hemi = 'north' if center_lat >=0 else 'south'
        target_crs = f"EPSG:{32600+utm_zone if hemi=='north' else 32700+utm_zone}"

        # transform bounds of region of interest from UTM to original CRS of DEM
        transformer = pyproj.Transformer.from_crs(target_crs, src.crs, always_xy=True)
        min_lon, max_lat = transformer.transform(bounds_reg[0], bounds_reg[3])
        max_lon, min_lat = transformer.transform(bounds_reg[1], bounds_reg[2])

        # create a window around region of interest in original CRS
        window = rasterio.windows.from_bounds(left=min_lon, bottom=min_lat, 
                                              right=max_lon, top=max_lat, 
                                              transform=src.transform)
        # crop DEM to region of interest and read in
        elev_crop = src.read(1, window=window)
        transform_crop = src.window_transform(window)

        # get bounds of cropped region and calc new reprojection transform
        bounds_crop = rasterio.windows.bounds(window, src.transform)
        transform_utm, width, height = rasterio.warp.calculate_default_transform(src.crs, target_crs, 
                                                                                 elev_crop.shape[1], elev_crop.shape[0], 
                                                                                 *bounds_crop)
        # reproject cropped DEM to UTM CRS and get elevation
        elev = np.empty((height, width), dtype=src.dtypes[0])
        rasterio.warp.reproject(source=elev_crop, 
                                destination=elev, 
                                src_transform=transform_crop, 
                                src_crs=src.crs, 
                                dst_transform=transform_utm, 
                                dst_crs=target_crs, 
                                resampling=rasterio.warp.Resampling.bilinear)
        # get meshgripd of easting and northing in UTM
        x_coords = np.arange(width) * transform_utm.a + transform_utm.c
        y_coords = np.arange(height) * transform_utm.e + transform_utm.f
        easting, northing = np.meshgrid(x_coords, y_coords)
        return easting, northing, elev


        ####dtmp21 = dtmp[1] - dtmp[0]
        ####restmp21 = np.abs(dtmp21 - (dists[:,:,1]-dists[:,:,0]))
        ####dtmp34 = dtmp[2] - dtmp[3]
        ####restmp34 = np.abs(dtmp34 - (dists[:,:,2]-dists[:,:,3]))
        ####dtmp46 = dtmp[3] - dtmp[4]
        ####restmp46 = np.abs(dtmp46 - (dists[:,:,3]-dists[:,:,4]))
        ####dtmp63 = dtmp[4] - dtmp[2]
        ####restmp63 = np.abs(dtmp63 - (dists[:,:,4]-dists[:,:,2]))
        ####dtmp78 = dtmp[5] - dtmp[6]
        ####restmp78 = np.abs(dtmp78 - (dists[:,:,5]-dists[:,:,6]))
        ####dtmp89 = dtmp[6] - dtmp[7]
        ####restmp89 = np.abs(dtmp89 - (dists[:,:,6]-dists[:,:,7]))
        ####dtmp97 = dtmp[7] - dtmp[5]
        ####restmp97 = np.abs(dtmp97 - (dists[:,:,7]-dists[:,:,5]))


        ## JDNA idx 0, 1, 2, 3
        #dtmp_jdna = np.diff(dtmp[:4])
        #dtmp_jdna = np.insert(dtmp_jdna, obj=0, values=(dtmp[3]-dtmp[0])) # handle first row (last minus first)
        #restmp_jdna = np.diff(dists[:,:,:4], axis=2)
        #restmp_jdna = np.insert(restmp_jdna, obj=0, axis=2, values=(dists[:,:,3]-dists[:,:,0]))
        ## JDNB idx 4, 5, 6, 7
        #dtmp_jdnb = np.diff(dtmp[4:8])
        #dtmp_jdnb = np.insert(dtmp_jdnb, obj=0, values=(dtmp[7]-dtmp[4]))
        #restmp_jdnb = np.diff(dists[:,:,4:8], axis=2)
        #restmp_jdnb = np.insert(restmp_jdnb, obj=0, axis=2, values=(dists[:,:,7]-dists[:,:,4]))
        ## jdsa idx 8, 9, 10, 11
        #dtmp_jdsa = np.diff(dtmp[8:12])
        #dtmp_jdsa = np.insert(dtmp_jdsa, obj=0, values=(dtmp[11]-dtmp[8]))
        #restmp_jdsa = np.diff(dists[:,:,8:12], axis=2)
        #restmp_jdsa = np.insert(restmp_jdsa, obj=0, axis=2, values=(dists[:,:,11]-dists[:,:,8]))
        ## jdsb idx 12, 13, 14, 15
        #dtmp_jdsb = np.diff(dtmp[12:16])
        #dtmp_jdsb = np.insert(dtmp_jdsb, obj=0, values=(dtmp[15]-dtmp[12]))
        #restmp_jdsb = np.diff(dists[:,:,12:16], axis=2)
        #restmp_jdsb = np.insert(restmp_jdsb, obj=0, axis=2, values=(dists[:,:,15]-dists[:,:,12]))
        ## top idx 16, 17, ... 59
        #dtmp_top = np.diff(dtmp[16:])
        #dtmp_top = np.insert(dtmp_top, obj=0, values=(dtmp[-1]-dtmp[16]))
        #restmp_top = np.diff(dists[:,:,16:], axis=2)
        #restmp_top = np.insert(restmp_top, obj=0, axis=2, values=(dists[:,:,-1]-dists[:,:,16]))

        ## combine all residuals into one matrix
        #drestmp = np.concatenate((restmp_jdna, restmp_jdnb, restmp_jdsa,
        #                                   restmp_jdsb, restmp_top), axis=2)
        ## distance residual (square, sum, sqrt)
        #drestmp = np.sqrt(np.sum(drestmp**2, axis=2))
        ## normalize by array size
        ## sqrt(sum(top)**2/44 + sum(jdna)**2/4)
        #drestmp = np.sqrt(restmp21**2 + restmp34**2 +
        #                  restmp46**2 + restmp63**2 + 
        #                  restmp78**2 + restmp89**2 + restmp97**2)


        #trestmp = drestmp / 343


        # add contour lines to plot
    #TODO note originally in for loop -- how to get different lines for different k vals???


        ## JDNA
        #dtmp01 = dtmp[0] - dtmp[1]
        #restmp01 = np.abs(dtmp01 - (dists[:,:,0] - dists[:,:,1]))
        #dtmp12 = dtmp[1] - dtmp[2]
        #restmp12 = np.abs(dtmp12 - (dists[:,:,1] - dists[:,:,2]))
        #dtmp23 = dtmp[2] - dtmp[3]
        #restmp23 = np.abs(dtmp23 - (dists[:,:,2] - dists[:,:,3]))
        #dtmp30 = dtmp[3] - dtmp[0]
        #restmp30 = np.abs(dtmp30 - (dists[:,:,3] - dists[:,:,0]))
        ## JDNB
        #dtmp45 = dtmp[4] - dtmp[5]
        #restmp45 = np.abs(dtmp45 - (dists[:,:,4] - dists[:,:,5]))
        #dtmp56 = dtmp[5] - dtmp[6]
        #restmp56 = np.abs(dtmp56 - (dists[:,:,5] - dists[:,:,6]))
        #dtmp67 = dtmp[6] - dtmp[7]
        #restmp67 = np.abs(dtmp67 - (dists[:,:,6] - dists[:,:,7]))
        #dtmp74 = dtmp[7] - dtmp[4]
        #restmp74 = np.abs(dtmp74 - (dists[:,:,7] - dists[:,:,4]))


def plot_array_response(path_figures, path_station_gps):

    fig_arr, ax_arr = plt.subplots(5, 2, figsize=[6,14])

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


        bad_stations = ["JDNA1", "JDNB3", "JDSA4", "TOP19", "TOP24", "TOP07", "TOP32"]
        # plot normal stations normally
        coords_sub_normal = coords_sub[~coords_sub["Station"].str.contains('|'.join(bad_stations))]
        ax_arr[i,0].plot(coords_sub_normal["Easting"], coords_sub_normal["Northing"], 'k^', markerfacecolor=arr_cols[i], markersize=10)
        # plot stations that cut out early or died differently
        coords_sub_bad = coords_sub[coords_sub["Station"].str.contains('|'.join(bad_stations))]
        ax_arr[i,0].plot(coords_sub_bad["Easting"], coords_sub_bad["Northing"], 'k^', markerfacecolor=arr_cols[i], alpha=0.3, markersize=10)

        # add station labels
        #TODO make this neater for all subplots

        center = [coords_sub["Easting"].mean(), coords_sub["Northing"].mean()]
        if "JD" in arr_str:
            # show all JD subfigs at same relative scale
            #ax_arr[i,0].set_xlim(center[0]+[-10,10])
            #ax_arr[i,0].set_ylim(center[1]+[-10,10])
            # add array element labels
            for _, row in coords_sub.iterrows():
                ax_arr[i,0].annotate(text=f'{row["Station"].replace(arr_str, "")}',    # label with just station number
                                    xy=(row["Easting"], row["Northing"]),
                                    xytext=(8,-5),
                                    textcoords='offset points')
            # set tick labels relative to JD array center
            ax_arr[i,0].set_xticks(ticks=np.linspace(center[0]-10, center[0]+10, 5), 
                                   labels=np.linspace(-10, 10, 5, dtype=int))
            ax_arr[i,0].set_yticks(ticks=np.linspace(center[1]-10, center[1]+10, 5), 
                                   labels=np.linspace(-10, 10, 5, dtype=int))
        else: # for TOP
            #ax_arr[i,0].set_xlim(center[0]+[-175,175])
            #ax_arr[i,0].set_ylim(center[1]+[-175,175])
            #ax_arr[i,0].set_xlim((coords_sub["Easting"].min()-50, coords_sub["Easting"].max()+50))
            #ax_arr[i,0].set_ylim((coords_sub["Northing"].min()-50, coords_sub["Northing"].max()+50))
            # add array element labels
            for l, row in coords_sub.reset_index()[::3].iterrows():
                if l in range(0,12): text_loc = (8, -5)
                if l in range(12,20): text_loc = (-5, 10)
                if l in range(20,30): text_loc = (-20, -5)
                if l in range(30,40): text_loc = (5, -10)
                if l in range(41,44): text_loc = (0, 5)
                ax_arr[i,0].annotate(text=f'{row["Station"].replace(arr_str, "")}',    # label with just station number
                                    xy=(row["Easting"], row["Northing"]),
                                    xytext=text_loc,
                                    textcoords='offset points')
            # set tick labels relative to TOP array center
            ax_arr[i,0].set_xticks(ticks=np.linspace(center[0]-180, center[0]+180, 5), 
                                   labels=np.linspace(-180, 180, 5, dtype=int))
            ax_arr[i,0].set_yticks(ticks=np.linspace(center[1]-180, center[1]+180, 5), 
                                   labels=np.linspace(-180, 180, 5, dtype=int))

        ax_arr[i,0].set_aspect('equal')
        ax_arr[i,0].grid(True)
        ax_arr[i,0].set_title(arr_str)
        ax_arr[i,1].set_title(arr_str)
        ax_arr[i,0].set_xlabel('Distance (m)')
        ax_arr[i,0].set_ylabel('Distance (m)')



        ax_arr[i,1].set_aspect('equal')
        # calculate and plot array response
        #test_coords = coords_sub[["Easting", "Northing"]].to_numpy()
        #test_coords = test_coords - center
        #kxs, kys, K = array_response(test_coords, n_wavenum=101, wavelen_max=15)        
        #TODO add z coord to these
        resp_coords = coords_sub_normal[["Longitude", "Latitude"]].to_numpy()
        resp_coords = np.insert(resp_coords, 2, 0, axis=1)



        # choose wavenum at 24Hz in rad/km as per Jake eq code
        klim = 2 * np.pi * 24 / 0.330   
        kstep = 1
        array_response = obspy.signal.array_analysis.array_transff_wavenumber(resp_coords,
                                                                              klim=klim,
                                                                              kstep=kstep,
                                                                              coordsys='lonlat')
        #test_coords = coords_sub[["Easting", "Northing"]].to_numpy()
        #test_coords = test_coords / 1000 #obspy wants km
        #test_coords = np.insert(test_coords, 2, 0, axis=1)
        #test_coords = test_coords.astype(np.float64)
        #array_response = obspy.signal.array_analysis.array_transff_wavenumber(test_coords,
        #                                                                      klim=klim,
        #                                                                      kstep=kstep,
        #                                                                      coordsys='xy')
        lim = klim #/100
        zlim = [0,1]
        res = ax_arr[i,1].imshow(array_response.transpose(), 
                           extent=[-lim, lim, -lim, lim],
                           origin='lower',
                           vmin=zlim[0], vmax=zlim[1],
                           cmap='plasma')
        #lim = 0.4
        #zlim = [0,1]
        #res = ax_arr[1,i].imshow(array_response.transpose(), 
        #                   extent=[-lim, lim, -lim, lim],
        #                   origin='lower',
        #                   vmin=zlim[0], vmax=zlim[1],
        #                   cmap='plasma')

        # plot circles on top
        circle_freq = [4, 8, 24]
        for f in circle_freq:
            r = 2*np.pi*f/0.343
            #r = 2*np.pi*f/0.333/1000
            theta = np.linspace(0, 2*np.pi, 100)
            ax_arr[i,1].plot(r*np.sin(theta), r*np.cos(theta), '--', color='grey')

        ax_arr[i,1].sharex(ax_arr[i-1,1])
        ax_arr[i,1].sharey(ax_arr[i-1,1])
        ax_arr[i,1].set_xticks([-250,0,250])
        ax_arr[i,1].set_yticks([-250,0,250])
        ax_arr[i,1].set_xlabel("$k_x$ (rad/km)")
        ax_arr[i,1].set_ylabel("$k_y$ (rad/km)")
    
        # set colorbars
        cbar = fig_arr.colorbar(res, ax=ax_arr[i,1], fraction=0.046, pad=0.03)
        cbar.set_ticks([0.0, 0.5, 1.0])
        cbar.set_label("Semblance")
    fig_arr.tight_layout()
    fig_arr.savefig(os.path.join(path_figures, "fig2_array_response.png"), dpi=500)

    return fig_arr




if __name__ == "__main__":



    settings.set_paths(location='laptop')
    path_dem = os.path.join(settings.path_home, "gis", "burn_severity", "elevation_contours", "USGS_13_n44w117_20240701.tif")
    main(settings.path_station_gps, path_dem, settings.path_figures)

