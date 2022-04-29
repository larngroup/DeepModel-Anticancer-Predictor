 # -*- coding: utf-8 -*-
"""
Created on Mon Jan 4 10:43:07 2022

@author: filipa
"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import itertools
import os
import numpy as np
import time

from autoencoders import autoencoder
from utils import load_cfg


config_file = 'config_autoencoder.json' # Configuration file 

def run():
    """Loads data, builds model, trains and evaluates it"""
    
    # load configuration file
    cfg_file = load_cfg(config_file)
     
    # Implementation of GEP VAE
    gep_model = autoencoder(cfg_file)
    gep_model.load_data('exp')
    gep_model.pre_process_data('exp')
    gep_model.build() 
    gep_model.train()
    rmse_exp, mse_exp, mae_exp, r2_exp = gep_model.evaluate()
    gep_model.save('exp')
    
    mut_model = autoencoder(cfg_file)
    mut_model.load_data('mut') 
    mut_model.pre_process_data('mut')
    mut_model.build() 
    mut_model.train()
    rmse_mut, mse_mut, mae_mut, r2_mut = mut_model.evaluate()
    mut_model.save('mut')  
    
def grid_search(): 
    
    file_code = 'land_rsem' 
    if file_code.endswith('rsem'):
        data_dir = '../data_TCGA/rsem'
    else:
        data_dir = '../data_TCGA/tpm'
        
    cfg_file = load_cfg(config_file)
    cfg_file.exp_file = data_dir + "_data/E_TCGA_"+ file_code +".pkl"
    cfg_file.mut_file = data_dir + "_data/M_TCGA_"+ file_code +".pkl"
    cfg_file.inp_dimension = 1954
    
    n_layers = [5, 4]
    layers_units = [8192, 4096, 2048, 1024, 512, 256]
    dropout = [0.1, 0.2, 0.3]
    batch_size = [64, 128, 256]
    cfg_file.n_epochs = 1
    if os.path.exists('../results/exp_results.csv'):
        results = pd.read_csv('../results/exp_results.csv')
        results = results.drop(columns = "Unnamed: 0")
    else:
        results = pd.DataFrame(columns=['File', 'Layers', 'Batch_size', 'Dropout', 'Cross_val', 'Epochs', 'Runs',
                                        'RMSE_exp', 'MSE_exp', 'MAE_exp', 'r2_exp',
                                        'RMSE_mut', 'MSE_mut', 'MAE_mut', 'r2_mut',
                                        'Train_time_exp', 'Train_time_mut', 'Total_time'])
    
    for L in n_layers:
        for subset in itertools.combinations(layers_units, L):
            layers_enc = list(subset)
            layers_dec = sorted(layers_enc, reverse = False)
            if layers_enc[0] > 3*cfg_file.inp_dimension:
                continue
            else:
                for batch in batch_size:
                    for drop in dropout:
                        
                        if ((results['File']== file_code) & (results['Layers'] == str(layers_enc)) 
                            & (results['Dropout'] == drop) & (results['Cross_val'].astype(str) == cfg_file.cross_validation) 
                            & (results['Epochs'] == cfg_file.n_epochs)
                            & (results['Batch_size']== batch)).any() == False:
                            
                            cfg_file.dropout = drop
                            cfg_file.batch_size = batch
                            cfg_file.layers_enc_units = layers_enc
                            cfg_file.layers_dec_units = layers_dec
                            
                            metrics = []
                            train_time = []
                            n_runs = 10
                            for i in range(n_runs):
                                start_total = time.time()
                                
                                gep_model = autoencoder(cfg_file)
                                gep_model.load_data('exp')
                                gep_model.pre_process_data('exp')
                                gep_model.build() 
                                start = time.time()
                                
                                exp_metrics = []
                                if cfg_file.cross_validation == 'True':
                                    for i,split in enumerate(gep_model.data_cv):
                                        print('\nCross validation, fold number ' + str(i) + ' in progress...')
                                        idx_train, idx_val = split
                                        gep_model.data_train = gep_model.dataset.iloc[idx_train,:]
                                        gep_model.data_val = gep_model.dataset.iloc[idx_val,:]
                                        gep_model.train()
                                        end = time.time()
                                        train_time_exp = end-start
                                        
                                        rmse, mse, mae, r2 = gep_model.evaluate()
                                        exp_metrics.append([rmse, mse, mae, r2])
                                    rmse_exp, mse_exp, mae_exp, r2_exp = np.mean(exp_metrics, axis = 0)
                                else:
                                    gep_model.train()
                                    end = time.time()
                                    train_time_exp = end-start
                                    rmse_exp, mse_exp, mae_exp, r2_exp = gep_model.evaluate()
                                
                                mut_model = autoencoder(cfg_file)
                                mut_model.load_data('mut')
                                mut_model.pre_process_data('mut')
                                mut_model.build() 
                                start = time.time()
                                
                                mut_metrics = []
                                if cfg_file.cross_validation == 'True':
                                    for i,split in enumerate(mut_model.data_cv):
                                        print('\nCross validation, fold number ' + str(i) + ' in progress...')
                                        idx_train, idx_val = split
                                        mut_model.data_train = mut_model.dataset.iloc[idx_train,:]
                                        mut_model.data_val = mut_model.dataset.iloc[idx_val,:]
                                        mut_model.train()
                                        end = time.time()
                                        train_time_exp = end-start
                                        
                                        rmse, mse, mae, r2 = mut_model.evaluate()
                                        mut_metrics.append([rmse, mse, mae, r2])
                                    rmse_mut, mse_mut, mae_mut, r2_mut = np.mean(mut_metrics, axis = 0)
                                else:
                                    mut_model.train()
                                    end = time.time()
                                    train_time_exp = end-start
                                    rmse_mut, mse_mut, mae_mut, r2_mut = mut_model.evaluate()
                                
                                mut_model.train()
                                end = time.time()
                                train_time_mut = end-start
                                mut_model.evaluate()
                                
                                end_total = time.time()
                                total_time = end_total-start_total
                                
                                metrics.append([rmse_exp, mse_exp, mae_exp, r2_exp, rmse_mut, mse_mut, mae_mut, r2_mut])
                                train_time.append([train_time_exp, train_time_mut, total_time])
                            
                            mean_metrics = np.mean(metrics, axis = 0)
                            mean_time = np.mean(train_time, axis = 0)
                            
                            results = results.append({'File': file_code, 'Layers': str(layers_enc), 'Batch_size': batch, 
                                                      'Dropout':drop, 'Cross_val': cfg_file.cross_validation, 'Epochs': cfg_file.n_epochs, 'Runs': n_runs,
                                                      'RMSE_exp': round(mean_metrics[0],4),'MSE_exp': round(mean_metrics[1],4), 
                                                      'MAE_exp': round(mean_metrics[2],4) , 'r2_exp': round(mean_metrics[3],4),
                                                      'RMSE_mut': round(mean_metrics[4],4), 'MSE_mut': round(mean_metrics[5],4),
                                                      'MAE_mut': round(mean_metrics[6],4), 'r2_mut': round(mean_metrics[7],4), 
                                                     'Train_time_exp': round(mean_time[0],2),'Train_time_mut': round(mean_time[1],2),
                                                     'Total_time': round(mean_time[2],2)},ignore_index=True)
                            results.to_csv('../results/exp_results.csv')
                        else:
                            continue
                                         
                           
if __name__ == '__main__':
    
    search = False
    if search == True:
        grid_search()
    else:
        run()

 