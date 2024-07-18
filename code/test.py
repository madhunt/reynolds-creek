import numpy as np
import matplotlib.pyplot as plt

def intersection(coords, angles):

    # create matrix of direction unit vectors
    D = np.array([[np.cos(np.radians(angles[0])), -1*np.cos(np.radians(angles[1]))],
                      [np.sin(np.radians(angles[0])), -1*np.sin(np.radians(angles[1]))]])
    # create vector of difference in start coords (p2x-p1x, p2y-p1y)
    P = np.array([coords[1,0] - coords[0,0],
                  coords[1,1] - coords[0,1]])

    # solve system of equations Dt=P
    t = np.linalg.solve(D, P)

    # see if intersection point is actually along rays
    if t[0]<0 or t[1]<0:
        # if intersection is "behind" rays, return nans
        return np.array([np.nan, np.nan])
        
    # calculate intersection point
    int_pt = np.array([coords[0,0]+D[0,0]*t[0],
                       coords[0,1]+D[1,0]*t[0]])
    return int_pt



coords = np.array([[2, 1],
                  [5, 6]])
angles = np.array([30,
                   45])

O = intersection(coords, angles)


l1_end = np.array([coords[0,0] + 100*np.cos(np.radians(angles[0])), coords[0,1] + 100*np.sin(np.radians(angles[0]))])
l2_end = np.array([coords[1,0] + 100*np.cos(np.radians(angles[1])), coords[1,1] + 100*np.sin(np.radians(angles[1]))])

plt.plot([coords[0,0], l1_end[0]], [coords[0,1], l1_end[1]], 'r-')
plt.plot([coords[1,0], l2_end[0]], [coords[1,1], l2_end[1]], 'b-')
plt.plot(O[0], O[1], 'k*')
plt.show()







