# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np




def compute_loss(y, tx, w, loss):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    # ***************************************************
    e = (y-np.dot(tx,w))
    
    match loss: 
        case 'mse':
            mse = (1/2)*np.mean(e**2)
            return mse , e
        case 'mae':
            mae = np.mean(np.abs(e))
            return mae, e
        case other:
            print("incorrect loss type, use 'mae' or 'mse' " )
    # ***************************************************
    #raise NotImplementedError