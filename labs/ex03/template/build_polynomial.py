# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    """""
    Args:
        x: numpy array of shape (N,), N is the number of samples.
        degree: integer.
        
    Returns:
        poly: numpy array of shape (N,d+1)
        
    >>> build_poly(np.array([0.0, 1.5]), 2)
    array([[1.  , 0.  , 0.  ],
           [1.  , 1.5 , 2.25]])
    """
    # *************************************************** #first we need to ad n columns raised to the power of n
    num_samples = x.shape[0]
    tx = np.c_[np.ones(num_samples), x]
    degree = degree + 1
    
    for j in range(degree):
        if j > 1:
            new_column = np.array(np.power(tx[:,1],j) )
            new_column = np.transpose(new_column)
            new_column = new_column[:, np.newaxis]
            tx = np.append(tx, new_column, axis=1)
    return tx 