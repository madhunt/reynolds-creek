import numpy as np

import matplotlib.pyplot as plt

import obspy, scipy, importlib

import sys
#sys.path.append('/home/mad/Documents/research/clean')
import cleanbf

try:

    importlib.reload(cleanbf)

    print('reloaded')

except:

    pass

#%% Small array (2x2 square) test with two point sources and noise

s_list = np.arange(-4, 4, 0.25)


num_windows = 20 # the number of windows strongly affects the remaining power ratio

win_length_sec = 60

freq_bin_width = 1

fl = 4

fh = 8

Nt = num_windows * win_length_sec * 100


wind_speed = 10 # m/s
# try other wind speeds? what other results does this give
# get higher slownesses than expected
# what does real wind data give from beamforming?

 

stream = cleanbf.make_synth_stream(Nt = Nt, sx = [-3, 1000/wind_speed], sy = [0, 0], amp = [1,0.7], 

                                   fl = [fl, 0.125], fh = [fh, fl],

                                   Nx = 2, Ny = 2, dx = 0.008, dy = 0.008, uncorrelatedNoiseAmp = 0) 

stream.filter('highpass', freq = fl)

x = obspy.signal.array_analysis.array_processing(stream, win_len = win_length_sec, win_frac = 0.5, 

                                            sll_x = -4, slm_x = 4, sll_y = -4, slm_y = 4, sl_s = 0.1, 

                                            semb_thres = 0, vel_thres = 0, frqlow = fl, frqhigh = fh, 

                                            stime = stream[0].stats.starttime, etime = stream[0].stats.endtime,

                                            prewhiten = False, coordsys = 'xy')
x = x[:,3:]


print(np.quantile(x[:,1], [0.1, 0.25, 0.5, 0.75, 0.9]))

plt.close('all')

plt.plot(x[:,0], x[:,1], 'k.')

plt.plot([90,90], [0,4])

plt.xlabel('Back-azimuth (deg)')

plt.ylabel('Horizontal slowness (s/km)')

plt.show()