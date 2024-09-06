#!/usr/bin/python3

import pytest
import triangulate

import numpy as np


def test_unit_intersection():

    # make up some points and azimuths
    p1 = np.array([1, 3])
    a1 = 120
    p2 = np.array([5, 7])
    a2 = 200
    # use intersect to find intersections
    int_pt = triangulate.intersection(p1, a1, p2, a2)
    # assert that this intersection is the same as the one calculated
    assert np.round(int_pt[0], 2) == 3.10
    assert np.round(int_pt[1], 2) == 1.79

    # test parallel rays
    # check that it fails in the right way (eg returns nans in this case)
    p1 = np.array([4, 1])
    a1 = 35
    p2 = np.array([-5, -3])
    a2 = 35
    int_pt = triangulate.intersection(p1, a1, p2, a2)
    assert np.isnan(int_pt[0])
    assert np.isnan(int_pt[1])

    p1 = np.array([-2, 5])
    a1 = 110
    p2 = np.array([3, -8])
    a2 = 290
    int_pt = triangulate.intersection(p1, a1, p2, a2)
    assert np.isnan(int_pt[0])
    assert np.isnan(int_pt[1])

    # test intersection pt "behind" rays
    p1 = np.array([-3, 3])
    a1 = 300
    p2 = np.array([-7, 5])
    a2 = 190
    int_pt = triangulate.intersection(p1, a1, p2, a2)
    assert np.isnan(int_pt[0])
    assert np.isnan(int_pt[1])


    # maybe one more angle test with diff angles 
    p1 = np.array([5, -3])
    a1 = 280
    p2 = np.array([3, 4])
    a2 = 230
    int_pt = triangulate.intersection(p1, a1, p2, a2)
    assert np.round(int_pt[0], 2) == -3.55
    assert np.round(int_pt[1], 2) == -1.49



    return




test_unit_intersection()