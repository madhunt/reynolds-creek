#!/usr/bin/python3
import pytest, sys, os
import numpy as np
# import from personal scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import triangulate

def test_intersection():
    '''
    Runs 5 test cases for intersection() function. Tests correct intersection of points and special 
        cases that should return an intersection point of (nan, nan), like parallel lines and an intersection 
        "behind" the two rays.
    '''
    def call_intersection(test_dict):
        '''
        Calls intersection() function with provided test case.
        INPUTS
            test_dict   : dict  : Contains p1, a1, p2, a2 of test rays, and manually calculated 
                intersection (int_x, int_y).
        '''
        int_pt = triangulate.intersection(test_dict['p1'], test_dict['a1'], test_dict['p2'], test_dict['a2'])

        if np.isnan(test_dict['int_x']) and np.isnan(test_dict['int_y']):
            assert np.isnan(int_pt[0])
            assert np.isnan(int_pt[1])
        else: 
            assert np.round(int_pt[0], 2) == test_dict['int_x']
            assert np.round(int_pt[1], 2) == test_dict['int_y']
        return
    
    # normal points in Q1 intersecting in Q1
    test1 = {'p1': np.array([1, 3]),
             'a1': 120,
             'p2': np.array([5, 7]),
             'a2': 200,
             'int_x': 3.10,
             'int_y': 1.79}
    # normal pointsin Q1, Q2 intersecting in Q3
    test2 = {'p1': np.array([5, -3]),
             'a1': 280,
             'p2': np.array([3, 4]),
             'a2': 230,
             'int_x': -3.55,
             'int_y': -1.49}
    # parallel rays with same angle
    test3 = {'p1': np.array([4, 1]),
             'a1': 35,
             'p2': np.array([-5, -3]),
             'a2': 35,
             'int_x': np.nan,
             'int_y': np.nan}
    # parallel rays with angle 180 deg off
    test4 = {'p1': np.array([-2, 5]),
             'a1': 110,
             'p2': np.array([3, -8]),
             'a2': 290,
             'int_x': np.nan,
             'int_y': np.nan}
    # intersection point "behind" rays
    test5 = {'p1': np.array([-3, 3]),
             'a1': 300,
             'p2': np.array([-7, 5]),
             'a2': 190,
             'int_x': np.nan,
             'int_y': np.nan}

    # run tests through intersection function
    call_intersection(test1)
    call_intersection(test2)
    call_intersection(test3)
    call_intersection(test4)
    call_intersection(test5)
    return

test_intersection()