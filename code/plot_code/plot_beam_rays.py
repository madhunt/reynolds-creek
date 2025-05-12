#!/usr/bin/python3

import os, sys, datetime, pytz
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# import files from dir above
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils, settings, crossbeam


def main(path_processed, path_heli, path_station_gps, path_figures,
         t0, tf, freqmin, freqmax):

    array_list = ['TOP', 'JDNA', 'JDNB', 'JDSA', 'JDSB']

    median_ints, all_ints, all_rays = crossbeam.main(path_processed, path_heli, path_station_gps,
                                                     None, 
                                                     t0, tf, freqmin, freqmax)
    


    for i, row in enumerate(all_rays.iterrows()):
        fig, ax = plt.subplots(1, 1, tight_layout=True)
        colors = cm.winter(np.linspace(0, 1, 5))
        for j, arr_str in enumerate(array_list):
            # get initial point, slope, and y-intercept of ray
            station_east, station_north = utils.station_coords_avg(path_station_gps, arr_str, latlon=False)
            backaz = row[1][arr_str]
            slope = 1 / np.tan(np.deg2rad(backaz))
            intercept = station_north - station_east * slope
            # only plot positive end of the ray (arbitrary length 100)
            if backaz < 180:
                ax.plot([station_east, 1e7], [station_north, 1e7*slope + intercept], 
                        color=colors[j])
            else: 
                ax.plot([station_east, 1e4], [station_north, 1e4*slope + intercept],
                        color=colors[j])
            # plot array centers as triangles
            ax.scatter(station_east, station_north, c='k', marker='^')
            ax.text(station_east, station_north-0.0004, s=arr_str,
                    va='center', ha='center')
        #FIXME for testing purposes plot all intersections
        for k in np.arange(0, 20, 2):
            t = row[0]
            ax.scatter(all_ints.loc[t].iloc[k], all_ints.loc[t].iloc[k+1],
                    c='green', marker='o')

        # plot median point
        med_east = median_ints.loc[t]["Easting"]
        med_north = median_ints.loc[t]["Northing"]
        ax.plot(med_east, med_north, c='orange', marker='*', 
                markersize=10, label='Median Intersection')

        # plot helicopter "true" point
        heli_east = median_ints.loc[t]["True Easting"]
        heli_north = median_ints.loc[t]["True Northing"]
        ax.plot(heli_east, heli_north, c='red', marker='*', 
                markersize=10, label='True Heli Location')

        ax.legend(loc='upper left')
        #ax.set_xlim([median_ints["Easting"].min(), median_ints["Easting"].max()])
        #ax.set_ylim([median_ints["Northing"].min(), median_ints["Northing"].max()])
        ax.set_xlim([515500, 520500])
        ax.set_ylim([4770000, 4780000])

        plt.suptitle(t)
        #plt.show()
        print(os.path.join(path_figures, "backaz_errors", f"{t}_timestep.png"))
        plt.savefig(os.path.join(path_figures, "backaz_errors", f"{t}_timestep.png"), dpi=500)
        plt.close()
    return



if __name__ == "__main__":
    # choose times when helicopter is moving so we can compare points
    # TIMES FOR 07 OCT
    #freqmin = 24.0
    #freqmax = 32.0
    #t0 = datetime.datetime(2023, 10, 7, 19, 0, 0, tzinfo=pytz.UTC)
    #tf = datetime.datetime(2023, 10, 7, 20, 0, 0, tzinfo=pytz.UTC)
    freqmin = 4.0
    freqmax = 8.0
    t0 = datetime.datetime(2023, 10, 6, 20, 0, 0, tzinfo=pytz.UTC)
    tf = datetime.datetime(2023, 10, 6, 21, 0, 0, tzinfo=pytz.UTC)
    main(settings.path_processed, settings.path_heli, 
         settings.path_station_gps, settings.path_figures,
         t0, tf, freqmin, freqmax)