 # -*- coding: utf-8 -*-
"""
Created on Mon Jan 4 10:43:07 2022

@author: filipa
"""

import json
from bunch import Bunch

def load_cfg(path):
    """ Loads configuration file
    Args:
        path (str): The path of the configuration file

    Returns:
        config (json): The configuration file
    """
    with open(path, 'r') as config_file:
        config_dict = json.load(config_file)
        config = Bunch(config_dict)
    return config
    
def save_splitted_dataset(train,test, data_type):
    if data_type == 'exp':
        train.to_pickle("data/gepae_training_subset.pkl")
        test.to_pickle("data/gepae_testing_subset.pkl")
    elif data_type == 'mut':
        train.to_pickle("data/mutae_training_subset.pkl")
        test.to_pickle("data/mutae_testing_subset.pkl")
    elif data_type == 'exp_ccle':
        train.to_pickle("data/exp_training_subset_ccle.pkl")
        test.to_pickle("data/exp_testing_subset_ccle.pkl")
    elif data_type == 'mut_ccle':
        train.to_pickle("data/mut_training_subset_ccle.pkl")
        test.to_pickle("data/mut_testing_subset_ccle.pkl")
    else:
        train.to_pickle("data/ic50_training_subset_ccle.pkl")
        test.to_pickle("data/ic50_testing_subset_ccle.pkl")


def r_square(y_true, y_pred):
    """
    This function implements the coefficient of determination (R^2) measure
    ----------
    y_true: True label   
    y_pred: Model predictions 
    -------
    Returns the R^2 metric to evaluate regressions
    """
    import tensorflow.keras.backend as K
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/(SS_tot + K.epsilon()))

def rmse(y_true, y_pred):
    """
    This function implements the root mean squared error measure
    ----------
    y_true: True label   
    y_pred: Model predictions 
    -------
    Returns the rmse metric to evaluate regressions
    """
    from tensorflow.keras import backend as K
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))