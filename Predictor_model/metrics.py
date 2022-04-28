# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 10:11:15 2022

@author: filipa
"""
from tensorflow.keras import backend as K

def rmse(y_true, y_pred):
    """
    This function implements the root mean squared error measure
    ----------
    y_true: True label   
    y_pred: Model predictions 
    -------
    Returns the rmse metric to evaluate regressions
    """
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def mse(y_true, y_pred):
    """
    This function implements the mean squared error measure
    ----------
    y_true: True label   
    y_pred: Model predictions 
    -------
    Returns the mse metric to evaluate regressions
    """
    return K.mean(K.square(y_pred - y_true), axis=-1)

def r_square(y_true, y_pred):
    """
    This function implements the coefficient of determination (R^2) measure
    ----------
    y_true: True label   
    y_pred: Model predictions 
    -------
    Returns the R^2 metric to evaluate regressions
    """
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/(SS_tot + K.epsilon()))
