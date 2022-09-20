# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 10:35:01 2022

@author: filipa
"""

# external
import tensorflow as tf
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.keras.optimizers import Adam
from tokens import tokens_table
from utils import tokenize, smiles2idx, normalize, pred_scatter_plot, denormalization, padding
tf.config.run_functions_eagerly(True)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from metrics import r_square, rmse, mse
import matplotlib.pyplot as plt
import math
import sklearn
from tensorflow.keras.losses import mean_squared_error
from sklearn.model_selection import KFold
import seaborn as sns

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
    
    """Gene Expression Profile (GEP) Variational Autoencoder Model Class"""
    def __init__(self, config):
        self.exp_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
        self.config = config
        
        self.token_table = tokens_table().table
        self.n_table = tokens_table().table_len

    def load_data(self):
        """ Loads and pre-processes the dataset
        
        Returns:
            dataset (list): The list with two elements - training and testing 
                            arrays.
        """
        
        print("Loading data...")
        self.dataset_exp = pd.read_csv(self.config.ccle_exp_file, index_col=0)
        self.dataset_mut = pd.read_csv(self.config.ccle_mut_file, index_col=0)
        
        if self.config.ic50_imputed == 'True':
            self.dataset_ic50 = pd.read_csv(self.config.ic50_imputed_file).drop(columns = "Unnamed: 0")
            self.dataset_ic50.index = self.dataset_ic50.X
            self.dataset_ic50.drop(columns='X', inplace=True)
            self.dataset_ic50 = self.dataset_ic50.transpose()

            ic_cell_lines = np.intersect1d(self.dataset_exp.index,self.dataset_ic50.index, return_indices=True)[1]
            self.dataset_exp = self.dataset_exp.iloc[ic_cell_lines, :] 
            self.dataset_mut = self.dataset_mut.iloc[ic_cell_lines, :]
            
            ic_cell_lines = np.intersect1d(self.dataset_exp.index,self.dataset_ic50.index, return_indices=True)[2]
            self.dataset_ic50 = self.dataset_ic50.iloc[ic_cell_lines, :]
            
        else:
            self.dataset_ic50 = pd.read_csv(self.config.cell_drug_id).drop(columns = "Unnamed: 0")
            self.drugs = pd.read_csv(self.config.drugs_file).drop(columns = "Unnamed: 0")
            self.drugs.index = self.drugs['drugs']
            self.drugs = self.drugs.drop(columns = "drugs")
            self.drugs = self.drugs.loc[self.dataset_ic50.drug_id.unique()]
            
            tokenDict = dict((token, i) for i, token in enumerate(self.token_table))
            if self.config.tokens_final == "False":
                # Padding and tokenize
                tokens = tokenize (self.drugs.smiles, self.token_table, 90)
                # Transforms each token to the respective integer
                smiles_int = smiles2idx(tokens, tokenDict)
                self.drugs['smiles_int'] = smiles_int.tolist()  
            else:
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
            
            if self.config.pic50 == "True":
                self.dataset_ic50.IC50 = -np.log10(10**self.dataset_ic50.IC50/10**6)
                self.dataset_ic50 = self.dataset_ic50.loc[(self.dataset_ic50['IC50'] > 0) & (self.dataset_ic50['IC50'] < 10)]
            
            # self.dataset_ic50 = self.dataset_ic50.drop(self.dataset_ic50[self.dataset_ic50.drug_id == 'Cisplatin'].index)
            #remove outliers
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
            
        
    def load_autoencoders(self):
        
        print("Loading autoencoders...")
        
        self.autoencoder_exp = tf.keras.models.load_model(self.config.autoencoder_exp_file)
        self.autoencoder_mut = tf.keras.models.load_model(self.config.autoencoder_mut_file)
        
        self.layers_enc_units = []
        for layer in self.autoencoder_exp.layers:
            if layer.name.split('_')[0] == 'dense':
                self.layers_enc_units.append(layer.units)
            if layer.name == 'dropout':
                self.dropout = layer.rate
        self.config.layers_enc_units = self.layers_enc_units
        
    def pre_process_data(self):    
        
        if self.config.ic50_imputed == 'True':
            # split the data in training and testing sets
            self.train_exp, self.test_exp = train_test_split(self.dataset_exp, test_size=0.20, random_state=55)
            self.test_exp, self.val_exp = train_test_split(self.test_exp, test_size=0.50, random_state=55)
            
            self.train_mut = self.dataset_mut.loc[self.train_exp.index, :]
            self.val_mut = self.dataset_mut.loc[self.val_exp.index, :]
            self.test_mut = self.dataset_mut.loc[self.test_exp.index, :]
            
            self.train_ic50 = self.dataset_ic50.loc[self.train_exp.index, :]
            self.val_ic50 = self.dataset_ic50.loc[self.val_exp.index, :]
            self.test_ic50 = self.dataset_ic50.loc[self.test_exp.index, :]
            
        else:
            train_id, test_id = train_test_split(self.dataset_ic50, test_size=0.30, random_state=55)
            test_id, val_id = train_test_split(test_id, test_size=0.50, random_state=55)
            
            # self.train_mut, self.val_mut, self.test_mut, self.train_exp, self.val_exp, self.test_exp, self.train_drugs, self.val_drugs, self.test_drugs = split_data(self.dataset_mut, self.dataset_exp, self.drugs, train_id, val_id, test_id)
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
                            
        
    def build(self):
        """ Builds the architecture of the model"""
        
        print("Building model...")     
             
        """Mutation encoder"""
        au_input = tf.keras.layers.Input(shape=(self.config.inp_dimension,))
        for i,layer in enumerate(self.config.layers_enc_units[:-1]):
            encoder = tf.keras.layers.Dense(layer, activation="relu")(au_input if i == 0 else encoder)
            encoder = tf.keras.layers.BatchNormalization()(encoder)
            encoder = tf.keras.layers.Dropout(self.dropout)(encoder) 
        encoder = tf.keras.layers.Dense(self.config.layers_enc_units[-1], activation="relu")(encoder)
        
        model_mut = tf.keras.Model(inputs=au_input, outputs=encoder)  
        """Load mutation encoder weights"""
        if self.config.autoencoders == 'True':
            layer_names=[layer.name for layer in model_mut.layers]
            for i,layer_name in enumerate(layer_names):
                if i!=0:
                    model_mut.get_layer(layer_name).set_weights(self.autoencoder_mut.layers[i].get_weights())
    
        """Expression encoder"""
        au_input = tf.keras.layers.Input(shape=(self.config.inp_dimension,))
        for i,layer in enumerate(self.config.layers_enc_units[:-1]):
            encoder = tf.keras.layers.Dense(layer, activation="relu")(au_input if i == 0 else encoder)
            encoder = tf.keras.layers.BatchNormalization()(encoder)
            encoder = tf.keras.layers.Dropout(self.dropout)(encoder) 
        encoder = tf.keras.layers.Dense(self.config.layers_enc_units[-1], activation="relu")(encoder) 
            
        model_exp = tf.keras.Model(inputs=au_input, outputs=encoder)   
        """Load expression encoder weights"""
        if self.config.autoencoders == 'True':
            layer_names=[layer.name for layer in model_exp.layers]
            for i,layer_name in enumerate(layer_names):
                if i!=0:
                    model_exp.get_layer(layer_name).set_weights(self.autoencoder_exp.layers[i].get_weights())
        
        exp_out = tf.expand_dims(model_exp.output, axis=2)     
        mut_out = tf.expand_dims(model_mut.output, axis=1)
        """Matrix nxn with mutation and expression information"""
        mul = tf.matmul(exp_out, mut_out)
        
        if self.config.ic50_imputed == 'True':
            
            """Predicting IC50 without drugs' information"""
            for i in range(2):
                d = tf.keras.layers.Conv1D(filters = self.config.cnn_filters[i],
                                          kernel_size = self.config.cnn_filters_size[i],
                                          strides = 1, padding = 'same',
                                          activation = 'relu') (mul if i == 0 else d)
                d = tf.keras.layers.BatchNormalization()(d)
            if self.config.global_type == 'Max':
                global_pool = tf.keras.layers.GlobalMaxPool1D()(d)
            else:
                global_pool = tf.keras.layers.GlobalAveragePooling1D()(d)
            out = tf.keras.layers.Dense(self.config.n_drugs, activation='linear')(global_pool)
            # print(out.shape)
            self.ic50_predictor = tf.keras.Model(inputs=[model_exp.input, model_mut.input], outputs=out) 
            
        elif self.config.ic50_imputed == 'False':
            """Predicting IC50 with drugs' information"""
            for i in range(2):
                g = tf.keras.layers.Conv1D(filters = self.config.cnn_filters[i],
                                            kernel_size = self.config.cnn_filters_size[i],
                                            strides = 1, padding = 'same') (mul if i == 0 else g)
                if self.config.activation == 'relu':
                    g = tf.keras.activations.relu(g)
                elif self.config.activation == 'leakyrelu':
                    g = tf.keras.layers.LeakyReLU()(g)
                g = tf.keras.layers.BatchNormalization()(g)
            if self.config.global_type == 'Max':
                global_pool_gene = tf.keras.layers.GlobalMaxPool1D()(g)
            else:
                global_pool_gene = tf.keras.layers.GlobalAveragePooling1D()(g)

            drug_input = tf.keras.layers.Input(shape=(self.config.input_length,))
            d = tf.keras.layers.Embedding(self.n_table, self.config.n_units, input_length=self.config.input_length)(drug_input)
            
            for i in range(len(self.config.cnn_filters)):
                d = tf.keras.layers.Conv1D(filters = self.config.cnn_filters[i],
                                          kernel_size = self.config.cnn_filters_size[i],
                                          strides = 1, padding = 'same') (d)
                if self.config.activation == 'relu':
                    d = tf.keras.activations.relu(d)
                elif self.config.activation == 'leakyrelu':
                    d = tf.keras.layers.LeakyReLU()(d)
                d = tf.keras.layers.BatchNormalization()(d)
            
            if self.config.attention == "Self" or self.config.attention == "False":
                if self.config.attention == "Self":
                    d = tf.keras.layers.Attention()([d,d])
                if self.config.global_type == 'Max':
                    global_pool_drug = tf.keras.layers.GlobalMaxPool1D()(d)
                else:
                    global_pool_drug = tf.keras.layers.GlobalAveragePooling1D()(d)
                conc = tf.keras.layers.Concatenate()([global_pool_gene, global_pool_drug])
            else:
                conc = tf.keras.layers.Attention()([g,d])
                if self.config.global_type == 'Max':
                    conc = tf.keras.layers.GlobalMaxPool1D()(conc)
                else:
                    conc = tf.keras.layers.GlobalAveragePooling1D()(conc)
                    
            for i in range(len(self.config.layers_dec_units)):
                f = tf.keras.layers.Dense(self.config.layers_dec_units[i])(conc if i==0 else f)
                
                if self.config.activation == 'relu':
                    f = tf.keras.activations.relu(f)
                elif self.config.activation == 'leakyrelu':
                    f = tf.keras.layers.LeakyReLU()(f)
                    
                f = tf.keras.layers.Dropout(self.config.dropout)(f)

            out = tf.keras.layers.Dense(1, activation='linear')(f)
            
            self.ic50_predictor = tf.keras.Model(inputs=[model_exp.input, model_mut.input, drug_input], outputs=out)

    
    def train(self):
        since = time.time()
        
        es = EarlyStopping(verbose=1, patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(factor = 0.1, patience=5, min_lr = 1e-6)
        # mc = ModelCheckpoint('best_model.h5', mode='min', verbose=1, save_best_only=True)
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
                                    validation_data =([self.val_exp, self.val_mut], self.val_ic50),
                                    batch_size=self.config.batch_size, 
                                    callbacks=callbacks_list)

        
        self.time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(self.time_elapsed // 60, self.time_elapsed % 60))

    def evaluate(self):
        
        if self.config.ic50_imputed == 'True':
            
            self.pred = self.ic50_predictor.predict(([self.test_exp, self.test_mut]))
            self.metric = self.ic50_predictor.evaluate([self.test_exp, self.test_mut], self.test_ic50, verbose=0)
            
            print(self.metric)
            
            self.metric_train = self.ic50_predictor.evaluate([self.train_exp, self.train_mut], self.train_ic50, verbose=0)
            print('train: ' + str(self.metric_train[0]) )  
            
            test = self.test_ic50.to_numpy()
            s =  [np.corrcoef(i,j) for i,j in zip(self.pred, test) ]
            pearsonc = [i[0,1] for i in s]
            len(pearsonc)
            np.mean(pearsonc)
            
            from matplotlib.ticker import PercentFormatter
            plt.figure()
            ax = sns.distplot(pearsonc, bins = 30, hist_kws=dict(edgecolor="black", linewidth=0.5), color = 'orangered')
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=len(pearsonc)))
            ax.set_ylabel('Density of cell lines')
            ax.set_xlabel('Sample-wise Pearson correlation in log IC50 for CCLE test sample')
            
        else:
            drug = np.array([np.array(xi) for xi in self.train_drugs])
            drug_test = np.array([np.array(xi) for xi in self.test_drugs])
            
            self.pred = self.ic50_predictor.predict([self.test_exp, self.test_mut, drug_test])
            self.pred = self.pred.reshape(-1)
            self.metric = self.ic50_predictor.evaluate([self.test_exp, self.test_mut, drug_test], self.test_ic50, verbose=0)
            self.metric[1] = math.sqrt(self.metric[0])

            print(self.metric)
            
            self.metric_train = self.ic50_predictor.evaluate([self.train_exp, self.train_mut, drug], self.train_ic50, verbose=0)
            print('train: ' + str(self.metric_train[0]) )  

    def save(self):
        
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
            
            # model_name.save(dirs+model)
