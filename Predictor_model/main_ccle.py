# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 10:23:54 2022

@author: filipa
"""

import warnings
warnings.filterwarnings('ignore')
from model_ccle import exp_mut, split_data
from utils import load_cfg
import numpy as np
import pandas as pd
import os

config_file = 'config_model.json' # Configuration file 

def run():
    """Loads data, builds model, trains and evaluates it"""
    # load configuration file
    config = load_cfg(config_file)
    
    if config.exp_type == 'rsem':
        config.inp_dimension = 1954
    else:
        config.inp_dimension = 1428    
    
    file_path = 'grid_results.csv'
    
    if os.path.exists(file_path):
        results = pd.read_csv(file_path)
        results = results.drop(columns = "Unnamed: 0")
    else:
        results = pd.DataFrame(columns=['Data_type', 'Layers encoder','layers decoder',
                                        'Batch_size', 'Units', 'Dropout', 'RNN', 'Remove outliers', 
                                        'Normalize','Norm_type','Epochs', 'Runs',
                                        'mse', 'rmse', 'r^2', 'mse_denorm', 'rmse_denorm','r^2_denorm','train_time'])
    
    
    model = exp_mut(config)
    model.load_data()
    model.load_autoencoders()
    
    n_runs = 1
    metrics = []
    total_train_time = []
    
    for i in range(n_runs):
        model.pre_process_data()
        if config.cross_val == "False":
            model.build()
            model.train()
            model.evaluate() 
        else:
            exp_metrics = []
            for i,split in enumerate(model.data_cv):
                print('\nCross validation, fold number ' + str(i) + ' in progress...')
                train_id, val_id = split
                model.train_mut, model.val_mut, model.test_mut, model.train_exp, model.val_exp, model.test_exp, model.train_drugs, model.val_drugs, model.test_drugs = split_data(model.dataset_mut, model.dataset_exp, model.drugs, train_id, val_id, model.test_id)
                model.train_ic50, model.val_ic50, model.test_ic50 = model.train_id.iloc[train_id]['IC50'].values, model.train_id.iloc[val_id]['IC50'].values, model.test_id['IC50'].values
                model.build()
                model.train()
                model.evaluate()
                exp_metrics.append(model.metric)
            model.metric = np.mean(exp_metrics, axis = 0)
        
        metric = model.metric
        train_time = model.time_elapsed
        
        metrics.append(metric)
        total_train_time.append(train_time)
        
    mean_metrics = np.mean(metrics, axis = 0)
    mean_time = np.mean(total_train_time)
        
        
    results = results.append({'Data_type': config.exp_type, 'Layers encoder': str(model.layers_enc_units),
                              'layers decoder': str(config.layers_dec_units), 'Batch_size': config.batch_size,
                              'Units': config.n_units , 'Dropout': config.dropout ,'RNN': config.rnn, 'Remove outliers': config.remove_outliers, 
                              'Normalize': config.normalize, 'Norm_type': config.norm_type,
                              'Epochs': config.n_epochs, 'Runs': n_runs ,'mse': round(mean_metrics[0],4), 'rmse': round(mean_metrics[1],4),
                              'r^2': round(mean_metrics[2],4), 'mse_denorm': round(mean_metrics[3],4), 'rmse_denorm': round(mean_metrics[4],4),
                              'r^2_denorm': round(mean_metrics[5],4),'train_time': round(mean_time,2)}, ignore_index=True)
    
    results.to_csv(file_path)
if __name__ == '__main__':
    run()
