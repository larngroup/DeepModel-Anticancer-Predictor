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
import itertools

config_file = 'config_model.json' # Configuration file 

def run():
    """Loads data, builds model, trains and evaluates it"""
    # load configuration file
    config = load_cfg(config_file)
    
    if config.exp_type == 'rsem':
        config.inp_dimension = 1954
    else:
        config.inp_dimension = 1428    
    
    file_path = 'grid_results_final.csv'
    
    if grid_search == True:
        
        autoencoders = ['True']
        n_enc_layers = [4, 5]
        layers_enc_units = [2046, 1024, 512, 256, 128]
        cnn_filters = [32, 64, 128]
        cnn_filters_size = [4, 5, 7] 
        batch_size = [ 128]
        learning_rate = [0.01, 0.001]
        dropout = [0.05, 0.1]
        global_type = ['Max', 'Average']
        activation = ['relu', 'leakyrelu']
    
    else:
        autoencoders = [config.autoencoders]
        n_enc_layers = [5]
        layers_enc_units = [config.layers_enc_units]
        cnn_filters = config.cnn_filters
        cnn_filters_size = config.cnn_filters_size
        batch_size = [config.batch_size]
        learning_rate = [config.learning_rate]
        dropout = [config.dropout]
        global_type = [config.global_type]
        activation = [config.activation]
    
    if os.path.exists(file_path):
        results = pd.read_csv(file_path)
        results = results.drop(columns = "Unnamed: 0")
    else:
        results = pd.DataFrame(columns=['Data_type', 'Autoencoders', 'Layers_encoder', 'Layers_decoder',
                                        'cnn_filters', 'cnn_filters_size',
                                        'Batch_size', 'LR', 'Dropout', 'Epochs', 'Runs',
                                        'Global_type', 'Activation', 
                                        'mse_train', 'mse', 'rmse', 'r^2','train_time'])
    
    for ae in autoencoders:
        if ae == 'True':
            n_enc_layers = [5]
            layers_enc_units = [4096, 2048, 1024, 512, 128]
        for L in n_enc_layers:
            for subset in itertools.combinations(layers_enc_units, L):
                config.layers_enc_units = list(subset)
                
                for subset_2 in itertools.combinations(cnn_filters, 2):
                    config.cnn_filters = list(subset_2)
                    
                    for subset_3 in itertools.combinations(cnn_filters_size, 2):
                        config.cnn_filters_size = list(subset_3)
                        
                        for bs in batch_size:
                            for lr in learning_rate:
                                for dr in dropout:
                                    for glob in global_type:
                                        for act in activation:
                                            
                                            if ((results['Data_type']== config.exp_type) & (results['Autoencoders'].astype(str) == ae) 
                                                & (results['Layers_encoder'] == str(config.layers_enc_units)) & (results['Layers_decoder'] == str(config.layers_dec_units))
                                                & (results['cnn_filters'] == str(config.cnn_filters)) & (results['cnn_filters_size'] == str(config.cnn_filters_size))
                                                & (results['Batch_size']== bs) & (results['LR']== lr) 
                                                & (results['Dropout'] == dr) & (results['Epochs'] == config.n_epochs)
                                                & (results['Global_type']== glob) & (results['Activation']== act)).any() == False:
  
                                                config.autoencoders = ae
                                                config.batch_size = bs 
                                                config.learning_rate = lr
                                                config.dropout = dr
                                                config.global_type = glob
                                                config.activation = act
                                                 
                                                model = exp_mut(config)
                                                model.load_data()
                                                if ae == 'True':
                                                    model.load_autoencoders()
                                
                                                n_runs = 1
                                                metrics = []
                                                metrics_train = []
                                                total_train_time = []
                                                
                                                for i in range(n_runs):
                                                    model.pre_process_data()
                                                    model.build()
                                                    model.train()
                                                    model.evaluate() 
                                                    metric = model.metric
                                                    train_time = model.time_elapsed
                                                    
                                                    metrics.append(metric)
                                                    metrics_train.append(model.metric_train)
                                                    total_train_time.append(train_time)
                                                    
                                                mean_metrics = np.mean(metrics, axis = 0)
                                                mean_metrics_train = np.mean(metrics_train, axis = 0)
                                                mean_time = np.mean(total_train_time)
                        
            
                                                results = results.append({'Data_type': config.exp_type, 'Autoencoders': config.autoencoders,'Layers_encoder': str(config.layers_enc_units),
                                                                          'Layers_decoder': str(config.layers_dec_units),'cnn_filters': str(config.cnn_filters),
                                                                          'cnn_filters_size': str(config.cnn_filters_size) ,'Batch_size': config.batch_size,
                                                                          'LR': config.learning_rate , 'Dropout': config.dropout ,'Epochs': config.n_epochs,
                                                                          'Runs': n_runs , 'Global_type': config.global_type , 'Activation': config.activation,
                                                                          'mse_train': round(mean_metrics_train[0],4),
                                                                          'mse': round(mean_metrics[0],4), 'rmse': round(mean_metrics[1],4),
                                                                          'r^2': round(mean_metrics[2],4), 'train_time': round(mean_time,2)}, ignore_index=True)
                                            
                                                results.to_csv(file_path)
                                                del model
                                            else:
                                                print('next parameter')
    
if __name__ == '__main__':
    grid_search = False
    run()
