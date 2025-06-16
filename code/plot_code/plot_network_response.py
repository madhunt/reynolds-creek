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
import rasterio, pyproj

# import files from dir above
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils, settings, crossbeam

def main(path_station_gps, path_dem):
    
    # define RCEW bounds (east/north)
    min_easting = 515000
    max_easting = 520000
    min_northing = 4771000
    max_northing = 4777000


    # read in DEM of region
    # USGS DEM of Kilauea https://data.usgs.gov/datacatalog/data/USGS:5eb19f3882cefae35a29c363
    # use dem to get zg (see below)

    # create meshgrid of all locations
    easting = np.arange(min_easting, max_easting, step=60)
    northing = np.arange(min_northing, max_northing, step=60)
    #northing = np.arange(4774000, 4775500, 25)
    #easting = np.arange(516000, 518500, 25)
    #northing = np.arange(-5000, 5000, step=25)
    #easting = np.arange(-2500, 7500, step=25)
    [xg, yg] = np.meshgrid(easting, northing)
    #zg = dem # line 61

    # read in DEM
    with rasterio.open(path_dem) as src:
        #determine target UTM CRS
        bounds = src.bounds
        center_lon = (bounds.left + bounds.right) / 2
        center_lat = (bounds.bottom + bounds.top) / 2
        utm_zone = int((center_lon+180)/6)+1
        hemi = 'north' if center_lat >=0 else 'south'
        target_crs = f"EPSG:{32600+utm_zone if hemi=='north' else 32700+utm_zone}"

        # transform RCEW bounds from UTM to DEM original CRS
        transformer = pyproj.Transformer.from_crs(target_crs, src.crs, always_xy=True)
        min_lon, max_lat = transformer.transform(min_easting, max_northing)
        max_lon, min_lat = transformer.transform(max_easting, min_northing)

        # create window (in original CRS)
        window = rasterio.windows.from_bounds(left=min_lon, bottom=min_lat, 
                                              right=max_lon, top=max_lat, 
                                              transform=src.transform)
        # read in cropped DEM
        elev_crop = src.read(1, window=window)
        transform_crop = src.window_transform(window)

        # reproject cropped DEM to UTM CRS
        # bounds of cropped data
        bounds_crop = rasterio.windows.bounds(window, src.transform)
        # calculate new reprojection transform
        transform_utm, width, height = rasterio.warp.calculate_default_transform(src.crs, target_crs, 
                                                                                 elev_crop.shape[1], elev_crop.shape[0], 
                                                                                 *bounds_crop)
        # reproject cropped DEM and get elevation
        elev_proj = np.empty((height, width), dtype=src.dtypes[0])
        rasterio.warp.reproject(source=elev_crop, 
                                destination=elev_proj, 
                                src_transform=transform_crop, 
                                src_crs=src.crs, 
                                dst_transform=transform_utm, 
                                dst_crs=target_crs, 
                                resampling=rasterio.warp.Resampling.bilinear)
        #rows, cols = np.meshgrid(np.arange(height), np.arange(width),
        #                         indexing='ij')
        #east, north = rasterio.transform.xy(transform_utm, rows, cols)
        #east = np.array(east)
        #north = np.array(north)
        # try different method
        x_coords = np.arange(width) * transform_utm.a + transform_utm.c
        y_coords = np.arange(height) * transform_utm.e + transform_utm.f
        east, north = np.meshgrid(x_coords, y_coords)

    # create 20 m contour intervals
    min_elev = np.nanmin(elev_proj)
    max_elev = np.nanmax(elev_proj)
    contour_levels = np.arange(np.floor(min_elev/20)*20,
                               np.ceil(max_elev/20)*20 + 20,
                               20)



    fig_net, ax_net = plt.subplots(1, 1)
    ax_net.set_aspect('equal')
    ax_net.grid(visible=True)
    ax_net.set_xlim(515000, 520000)
    ax_net.set_ylim(4771000, 4777000)
    #ax_net.set_xlim(-2500, 7500)
    #ax_net.set_ylim(-5000, 5000)
    ax_net.set_xlabel("Easting (m)")
    ax_net.set_ylabel("Northing (m)")


    # add contours
    ax_net.contour(east, north, elev_proj, levels=contour_levels,
                   colors='grey',
                   linewidths=0.8)


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

    #coords = pd.DataFrame(columns=['Station', 'Easting', 'Northing'])
    #coords['Station'] = ['KINA1', 'KINA2',
    #                     'KINB3', 'KINB4', 'KINB6',
    #                     'KINC7', 'KINC8', 'KINC9']
    #coords['Easting'] = [2365, 2400, 2580, 2600, 2590, 2105, 2120, 2140]
    #coords['Northing'] = [1815, 1785, 355, 360, 385, -1040, -1060, -1015]

    #arr_list = ['KINA', 'KINB', 'KINC']
    #arr_cols = ['yellow', 'red', 'blue']

    
    for i, arr_str in enumerate(arr_list):
        coords_sub = coords[coords["Station"].str.contains(arr_str)]

        # plot all stations
        ax_arr[0,i].plot(coords_sub["Easting"], coords_sub["Northing"], 'k^', markerfacecolor=arr_cols[i], markersize=10)
        ax_net.plot(coords_sub["Easting"], coords_sub["Northing"], 'k^', markerfacecolor=arr_cols[i])

        # add station labels
        #TODO make this neater for all subplots
        ax_net.text(coords_sub["Easting"].max(), coords_sub["Northing"].min(), s=arr_str, verticalalignment='top', horizontalalignment='left')

        if "JD" in arr_str:
            center = [coords_sub["Easting"].mean(), coords_sub["Northing"].mean()]
            # show all JD subfigs at same relative scale
            ax_arr[0,i].set_xlim(center[0]+[-10,10])
            ax_arr[0,i].set_ylim(center[1]+[-10,10])
            # add array element labels
            for _, row in coords_sub.iterrows():
                ax_arr[0,i].annotate(text=f'{row["Station"].replace(arr_str, "")}',    # label with just station number
                                    xy=(row["Easting"], row["Northing"]),
                                    xytext=(8,-5),
                                    textcoords='offset points')
        else: # for TOP
            ax_arr[0,i].set_xlim((coords_sub["Easting"].min()-50, coords_sub["Easting"].max()+50))
            ax_arr[0,i].set_ylim((coords_sub["Northing"].min()-50, coords_sub["Northing"].max()+50))
            # add array element labels
            for l, row in coords_sub.reset_index()[::3].iterrows():
                if l in range(0,12): text_loc = (8, -5)
                if l in range(12,20): text_loc = (-5, 10)
                if l in range(20,30): text_loc = (-20, -5)
                if l in range(30,40): text_loc = (5, -10)
                if l in range(41,44): text_loc = (0, 5)
                ax_arr[0,i].annotate(text=f'{row["Station"].replace(arr_str, "")}',    # label with just station number
                                    xy=(row["Easting"], row["Northing"]),
                                    xytext=text_loc,
                                    textcoords='offset points')

        ax_arr[0,i].set_aspect('equal')
        ax_arr[0,i].grid(True)
        ax_arr[0,i].set_title(arr_str)



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
        res = ax_arr[1,i].imshow(array_response.transpose(), 
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
        cbar = fig_arr.colorbar(res, ax=ax_arr[1,i])
        cbar.set_label("Semblance")

        # plot circles on top
        circle_freq = [4, 8, 24]
        for f in circle_freq:
            r = 2*np.pi*f/0.343
            #r = 2*np.pi*f/0.333/1000
            theta = np.linspace(0, 2*np.pi, 100)
            ax_arr[1,i].plot(r*np.sin(theta), r*np.cos(theta), 'k--')

        ax_arr[1,i].set_xlabel("$k_x$ (rad/km)")
        ax_arr[1,i].set_ylabel("$k_y$ (rad/km)")
    
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
    sp = 10
    xgdec = xg[10::sp, 10::sp]
    ygdec = yg[10::sp, 10::sp]
    #xgdec = xg[100:350:50, 67:375:50]
    #ygdec = yg[100:350:50, 67:375:50]
    ax_net.plot(xgdec, ygdec, "ko", markerfacecolor=[1,0.5,0.5])
    
    colors_net = plt.get_cmap('rainbow')(np.linspace(0,1,np.size(xgdec)))
    
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

        # top idx 16, 17, ... 59
        dtmp_top = np.diff(dtmp[16:])
        dtmp_top = np.insert(dtmp_top, obj=0, values=(dtmp[-1]-dtmp[16]))
        dists_top = dists[:,:,16:]
        restmp_top = 0
        for n in range(len(dtmp_top)):
            dtmpnn = dtmp_top[n]
            if n+1 == len(dtmp_top):
                # account for last row
                restmpnn = np.abs(dtmpnn - (dists_top[:,:,n] - dists_top[:,:,0]))
            else:
                restmpnn = np.abs(dtmpnn - (dists_top[:,:,n] - dists_top[:,:,n+1]))
            restmp_top += restmpnn**2
        

        drestmp = np.sqrt(restmp_jdna/16 + restmp_jdnb/16 +
                          restmp_jdsa/16 + restmp_jdsb/16 + 
                          restmp_top/44**2)
        #drestmp = np.sqrt(restmp_jdna + restmp_jdnb + 
        #                  restmp_jdsa + restmp_jdsb)
        #drestmp = np.sqrt(restmp01**2 + restmp12**2 + restmp23**2 + restmp30**2 + 
        #                  restmp45**2 + restmp56**2 + restmp67**2 + restmp74**2)
        trestmp = drestmp / 343


        print(trestmp.min(), trestmp.max())
        #ax_net.plot(xtmp, ytmp, "ko", markerfacecolor=colors_net[k])
        contour = ax_net.contour(easting, northing, trestmp, 
                                 #levels=np.linspace(trestmp.min(), trestmp.max(), 3),
                                 #NOTE change these (default is jeff's)
                                 #levels=np.arange(0.005, 0.015, 0.00125),
                                 #levels=np.arange(0.001, 0.004, 0.001),
                                 levels=np.arange(0.010, 0.015, 0.001),
                                 colors=[(1, 0.5, 0.5)])
                                 #colors=[colors_net[k]])
                                 #cmap='viridis')
    cbar_net = fig_net.colorbar(contour)

    plt.close(fig_net)
    plt.show()



    return



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









if __name__ == "__main__":



    path_dem = os.path.join(settings.path_home, "gis", "burn_severity", "elevation_contours", "USGS_13_n44w117_20240701.tif")
    main(settings.path_station_gps, path_dem)

