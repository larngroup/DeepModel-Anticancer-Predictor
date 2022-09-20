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


# #concordance correlation coeﬃcient (CCC)
# def ccc(y_true,y_pred):
#     """
#     This function implements the concordance correlation coefficient (ccc)
#     ----------
#     y_true: True label   
#     y_pred: Model predictions 
#     -------
#     Returns the ccc measure that is more suitable to evaluate regressions.
#     """
#     num = 2*K.sum((y_true-K.mean(y_true))*(y_pred-K.mean(y_pred)))
#     den = K.sum(K.square(y_true-K.mean(y_true))) + K.sum(K.square(y_pred-K.mean(y_pred))) + K.int_shape(y_pred)[-1]*K.square(K.mean(y_true)-K.mean(y_pred))
#     return num/den

def mae(y_true, y_pred):
    """
    This function implements the coefficient of determination (R^2) measure
    ----------
    y_true: True label   
    y_pred: Model predictions 
    -------
    Returns the mae metric to evaluate regressions
    """
    return K.mean(K.abs(y_pred - y_true), axis=-1)