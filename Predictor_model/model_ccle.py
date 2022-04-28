# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 10:35:01 2022

@author: filipa
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import math
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
tf.config.run_functions_eagerly(True)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from tokens import tokens_table
from utils import normalize, pred_scatter_plot, denormalization, padding
from metrics import rmse, r_square

def split_data(dataset_mut, dataset_exp, drugs, train_id, val_id, test_id):
    train_mut = dataset_mut.loc[train_id, :]
    val_mut = dataset_mut.loc[val_id, :]
    test_mut = dataset_mut.loc[test_id.index, :]
    
    train_exp = dataset_exp.loc[train_id, :]
    val_exp = dataset_exp.loc[val_id, :]
    test_exp = dataset_exp.loc[test_id.index, :]
    
    train_drugs = drugs.loc[train_id, :].smiles_int
    val_drugs = drugs.loc[val_id, :].smiles_int
    test_drugs = drugs.loc[test_id.index, :].smiles_int
    
    return train_mut, val_mut, test_mut, train_exp, val_exp, test_exp, train_drugs, val_drugs, test_drugs
    
class exp_mut:
    
    """Anticancer drug response model class"""
    def __init__(self, config):
        self.exp_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
        self.config = config
        
        self.token_table = tokens_table().table
        self.n_table = tokens_table().table_len

    def load_data(self):
        """ Loads and pre-processes the dataset"""
        
        print("Loading data...")
        self.dataset_exp = pd.read_pickle(self.config.ccle_exp_file)
        self.dataset_mut = pd.read_pickle(self.config.ccle_mut_file)
        
        if self.config.ic50_imputed == 'False':
            self.dataset_ic50 = pd.read_csv(self.config.cell_drug_id).drop(columns = "Unnamed: 0")
            self.drugs = pd.read_csv(self.config.drugs_file).drop(columns = "Unnamed: 0")
            self.drugs.index = self.drugs['drugs']
            self.drugs = self.drugs.drop(columns = "drugs")
            self.drugs = self.drugs.loc[self.dataset_ic50.drug_id.unique()]
            
            tokenDict = dict((token, i) for i, token in enumerate(self.token_table))

            molecules = padding(self.drugs.smiles, 90)
            smiles_int = []
            for mol in molecules:
                smiles_int.append([tokenDict[k] for k in mol])
            self.drugs['smiles_int'] = smiles_int
                
            ic_cell_lines = np.intersect1d(self.dataset_exp.index,self.dataset_ic50['cell_id'].unique(), return_indices=True)[1]
            self.dataset_exp = self.dataset_exp.iloc[ic_cell_lines, :] 
            self.dataset_mut = self.dataset_mut.iloc[ic_cell_lines, :]
            
            ic_cell_lines = np.intersect1d(self.dataset_exp.index,self.dataset_ic50['cell_id'].unique(), return_indices=True)[0]
            self.dataset_ic50 = self.dataset_ic50.loc[self.dataset_ic50['cell_id'].isin(ic_cell_lines)]
            
            #Remove outliers
            if self.config.remove_outliers == "True":
                q3, q1 = np.percentile(self.dataset_ic50['IC50'], [75 ,25])
                iqr = q3 - q1
                
                lower = q1 - (1.5 * iqr)
                upper = q3 + (1.5 * iqr)
                
                self.dataset_ic50 = self.dataset_ic50.loc[(self.dataset_ic50['IC50'] >= lower) & (self.dataset_ic50['IC50'] <= upper)]
        
            self.dataset_exp = self.dataset_exp.loc[self.dataset_ic50['cell_id'].values]
            self.dataset_mut = self.dataset_mut.loc[self.dataset_ic50['cell_id'].values]
            self.drugs = self.drugs.loc[self.dataset_ic50['drug_id'].values]
            
            self.dataset_exp.index = [i for i in range(self.dataset_exp.shape[0])]
            self.dataset_mut.index = [i for i in range(self.dataset_mut.shape[0])]
            self.dataset_ic50.index = [i for i in range(self.dataset_ic50.shape[0])]
            self.dataset_ic50.IC50 = self.dataset_ic50.IC50.astype(np.float32)
            self.drugs.index = [i for i in range(self.drugs.shape[0])]

        else:
            self.dataset_ic50 = pd.read_csv(self.config.ic50_imputed_file).drop(columns = "Unnamed: 0")
            self.dataset_ic50.index = self.dataset_ic50.X
            self.dataset_ic50.drop(columns='X', inplace=True)
            self.dataset_ic50 = self.dataset_ic50.transpose()

            ic_cell_lines = np.intersect1d(self.dataset_exp.index,self.dataset_ic50.index, return_indices=True)[1]
            self.dataset_exp = self.dataset_exp.iloc[ic_cell_lines, :] 
            self.dataset_mut = self.dataset_mut.iloc[ic_cell_lines, :]
            
            ic_cell_lines = np.intersect1d(self.dataset_exp.index,self.dataset_ic50.index, return_indices=True)[2]
            self.dataset_ic50 = self.dataset_ic50.iloc[ic_cell_lines, :]
            
        
    def load_autoencoders(self):
        """Loads autoencoders and saves the units of each dense layer and dropout rate"""
        
        print("Loading autoencoders...")
        
        self.autoencoder_exp = tf.keras.models.load_model(self.config.autoencoder_exp_file)
        self.autoencoder_mut = tf.keras.models.load_model(self.config.autoencoder_mut_file)
        
        self.layers_enc_units = []
        for layer in self.autoencoder_exp.layers:
            if layer.name.split('_')[0] == 'dense':
                self.layers_enc_units.append(layer.units)
            if layer.name == 'dropout':
                self.dropout = layer.rate
        
    def pre_process_data(self):    
        """ Pre-processes the datasets"""
        
        if self.config.ic50_imputed == 'True':
            """Split data in training and testing sets (model without drugs' information)"""
            self.train_exp, self.test_exp = train_test_split(self.dataset_exp, test_size=0.20, random_state=55)
            self.train_mut = self.dataset_mut.loc[self.train_exp.index, :]
            self.test_mut = self.dataset_mut.loc[self.test_exp.index, :]
            
            self.train_ic50 = self.dataset_ic50.loc[self.train_exp.index, :]
            self.test_ic50 = self.dataset_ic50.loc[self.test_exp.index, :]
            
        else:
            """Split data in training, validating and testing sets (model with drugs' information)"""
            if self.config.cross_val == "False":
                train_id, test_id = train_test_split(self.dataset_ic50, test_size=0.30, random_state=55)
                test_id, val_id = train_test_split(test_id, test_size=0.50, random_state=55)
                
                self.train_mut = self.dataset_mut.loc[train_id.index, :]
                self.val_mut = self.dataset_mut.loc[val_id.index, :]
                self.test_mut = self.dataset_mut.loc[test_id.index, :]
                
                self.train_exp = self.dataset_exp.loc[train_id.index, :]
                self.val_exp = self.dataset_exp.loc[val_id.index, :]
                self.test_exp = self.dataset_exp.loc[test_id.index, :]
                
                self.train_drugs = self.drugs.loc[train_id.index, :].smiles_int
                self.val_drugs = self.drugs.loc[val_id.index, :].smiles_int
                self.test_drugs = self.drugs.loc[test_id.index, :].smiles_int
                
                my_dict = {'Labels_train': np.reshape(train_id['IC50'].values, -1) , 
                           'Labels_val': np.reshape(val_id['IC50'].values, -1),
                           'Labels_test': np.reshape(test_id['IC50'].values, -1)}
                fig, ax = plt.subplots()
                ax.boxplot(my_dict.values())
                ax.set_xticklabels(my_dict.keys())
                
                if self.config.normalize == "False":
                    self.train_ic50, self.val_ic50, self.test_ic50 = train_id['IC50'].values, val_id['IC50'].values, test_id['IC50'].values
                else:
                    self.aux, self.train_ic50, self.val_ic50, self.test_ic50 = normalize(train_id['IC50'].values,val_id['IC50'].values, test_id['IC50'].values, self.config.norm_type)
                            
            else:
                
                self.train_id, self.test_id = train_test_split(self.dataset_ic50, test_size=0.15, random_state=55)

                cross_validation_split = KFold(n_splits = 5, shuffle=True, random_state=42)
                self.data_cv = list(cross_validation_split.split(self.train_id))
        
    def build(self):
        """ Builds the architecture of the model"""
        
        print("Building model...")     
             
        """ Mutation encoder"""
        au_input = tf.keras.layers.Input(shape=(self.config.inp_dimension,))
        for i,layer in enumerate(self.layers_enc_units[:-1]):
            encoder = tf.keras.layers.Dense(layer, activation="relu")(au_input if i == 0 else encoder)
            encoder = tf.keras.layers.BatchNormalization()(encoder)
            encoder = tf.keras.layers.Dropout(self.dropout)(encoder) 
        encoder = tf.keras.layers.Dense(self.layers_enc_units[-1], activation="relu")(encoder)
        
        model_mut = tf.keras.Model(inputs=au_input, outputs=encoder)  
        layer_names=[layer.name for layer in model_mut.layers]
        for i,layer_name in enumerate(layer_names):
            if i!=0:
                model_mut.get_layer(layer_name).set_weights(self.autoencoder_mut.layers[i].get_weights())
    
        """Gene expression encoder"""
        au_input = tf.keras.layers.Input(shape=(self.config.inp_dimension,))
        for i,layer in enumerate(self.layers_enc_units[:-1]):
            encoder = tf.keras.layers.Dense(layer, activation="relu")(au_input if i == 0 else encoder)
            encoder = tf.keras.layers.BatchNormalization()(encoder)
            encoder = tf.keras.layers.Dropout(self.dropout)(encoder) 
        encoder = tf.keras.layers.Dense(self.layers_enc_units[-1], activation="relu")(encoder) 
            
        model_exp = tf.keras.Model(inputs=au_input, outputs=encoder)   
        layer_names=[layer.name for layer in model_exp.layers]
        for i,layer_name in enumerate(layer_names):
            if i!=0:
                model_exp.get_layer(layer_name).set_weights(self.autoencoder_exp.layers[i].get_weights())
        
        if self.config.ic50_imputed == 'True':
            """ Model without drugs"""
            x = tf.keras.layers.Concatenate()([model_exp.output, model_mut.output])
            for i,layer in enumerate(self.config.layers_dec_units):
                l = tf.keras.layers.Dense(layer, activation='relu')(x if i == 0 else l)
                l = tf.keras.layers.Dropout(self.config.dropout)(l)
            out = tf.keras.layers.Dense(self.config.n_drugs, activation='linear')(l)
            
            self.ic50_predictor = tf.keras.Model(inputs=[model_exp.input, model_mut.input], outputs=out)
            
        else:
            """ Model with drugs"""
            drug_input = tf.keras.layers.Input(shape=(self.config.input_length,))
            d = tf.keras.layers.Embedding(self.n_table, self.config.n_units, input_length=self.config.input_length)(drug_input)
            
            if self.config.rnn == 'LSTM':
                d = tf.keras.layers.LSTM(self.config.n_units, return_sequences=True, input_shape=(None,self.config.n_units,self.config.input_length),dropout = self.config.dropout)(d)
                d = tf.keras.layers.LSTM(self.config.n_units, dropout = self.config.dropout)(d)
            
            else:
                d = tf.keras.layers.GRU(self.config.n_units, return_sequences=True, input_shape=(None,self.config.n_units,self.config.input_length), dropout = self.config.dropout)(d)
                d = tf.keras.layers.GRU(self.config.n_units, dropout = self.config.dropout)(d)
                
            d = tf.keras.layers.Dense(self.config.n_units, activation="relu")(d)
            model_drugs = tf.keras.Model(inputs=drug_input, outputs=d) 
            
            x = tf.keras.layers.Concatenate()([model_exp.output, model_mut.output, model_drugs.output])
            for i,layer in enumerate(self.config.layers_dec_units):
                l = tf.keras.layers.Dense(layer, activation='relu')(x if i == 0 else l)
                l = tf.keras.layers.Dropout(self.config.dropout)(l)
            out = tf.keras.layers.Dense(1, activation='linear')(l)
            
            self.ic50_predictor = tf.keras.Model(inputs=[model_exp.input, model_mut.input, model_drugs.input], outputs=out)

    
    
    def train(self):
        
        """ Trains the model""" 
        since = time.time()
        
        es = EarlyStopping(verbose=1, patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(factor = 0.1, patience=5, min_lr = 1e-6)
        callbacks_list = [es,reduce_lr]
        opt = Adam(lr=self.config.learning_rate, amsgrad=False)

        self.ic50_predictor.compile(loss="mean_squared_error", optimizer = opt, metrics = [rmse, r_square])
    
        if self.config.ic50_imputed == 'False':

            drug = np.array([np.array(xi) for xi in self.train_drugs])
            drug_val = np.array([np.array(xi) for xi in self.val_drugs])
            self.ic50_predictor.fit([self.train_exp,self.train_mut, drug], self.train_ic50,
                                    epochs=self.config.n_epochs, 
                                    validation_data =([self.val_exp,self.val_mut, drug_val], self.val_ic50),
                                    shuffle = True,
                                    batch_size=self.config.batch_size, 
                                    callbacks=callbacks_list)
            
        else:
            
            self.ic50_predictor.fit([self.train_exp, self.train_mut], self.train_ic50,
                                    epochs=self.config.n_epochs, 
                                    validation_data =([self.test_exp, self.test_mut], self.test_ic50),
                                    batch_size=self.config.batch_size, 
                                    callbacks=callbacks_list)

        
        self.time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(self.time_elapsed // 60, self.time_elapsed % 60))

    def evaluate(self):
        
        """ Evaluates the model""" 
        if self.config.ic50_imputed == 'True':
            
            self.pred = self.ic50_predictor.predict([self.test_exp, self.test_mut])
            self.metric = self.ic50_predictor.evaluate([self.test_exp, self.test_mut], self.test_ic50, verbose=0)
            self.metric.insert(0,0)
            self.metric.insert(0,0)
            self.metric.insert(0,0)  
            print(self.metric)
            
        else:
            drug_test = np.array([np.array(xi) for xi in self.test_drugs])
            
            self.pred = self.ic50_predictor.predict([self.test_exp, self.test_mut, drug_test])
            self.pred = self.pred.reshape(-1)
            self.metric = self.ic50_predictor.evaluate([self.test_exp, self.test_mut, drug_test], self.test_ic50, verbose=0)
            self.metric[1] = math.sqrt(self.metric[0])
        
            if self.config.normalize == "False":
                pred_scatter_plot(self.test_ic50, self.pred, 'Predictions vs True Values', 'True values', 'Predictions', 'denorm')
                self.metric.insert(0,0)
                self.metric.insert(0,0)
                self.metric.insert(0,0)   
                print(self.metric) 
    
                drug = np.array([np.array(xi) for xi in self.train_drugs])
                metric_train = self.ic50_predictor.evaluate([self.train_exp, self.train_mut, drug], self.train_ic50, verbose=0)
                print('Train mse: ' + str(metric_train[0]) )  
            else:
                
                pred_scatter_plot(self.test_ic50, self.pred, 'Predictions vs True Values(Normalized)', 'True values', 'Predictions', 'norm')
                test_ic50_denorm = denormalization(self.test_ic50, self.aux, self.config.norm_type)
    
                pred_denorm = denormalization(self.pred, self.aux, self.config.norm_type)
    
                mse_denorm = sklearn.metrics.mean_squared_error(test_ic50_denorm, pred_denorm)
                rmse_denorm = math.sqrt(mse_denorm)
                r_squared_denorm = (np.corrcoef(test_ic50_denorm, pred_denorm)[0, 1])**2
                self.metric.append(mse_denorm)
                self.metric.append(rmse_denorm)
                self.metric.append(r_squared_denorm)
                print(self.metric)
    
                pred_scatter_plot(test_ic50_denorm, pred_denorm, 'Predictions vs True Values (Denormalized)', 'True values', 'Predictions', 'denorm')
        
    def save(self):
        
        """ Save the model""" 
        
        dirs = os.path.join('experiments',self.exp_time+'\\')
            
       	try:
            if not os.path.exists(dirs):
                os.makedirs(dirs)

       	except Exception as err:
       		print('Creating directories error: {}'.format(err))
       		exit(-1)
               
        model_name = 'IC50_Predictor'
        self.IC50_Predictor.save(dirs+model_name+".h5")
        print("\n" + model_name + " successfully saved to disk")
