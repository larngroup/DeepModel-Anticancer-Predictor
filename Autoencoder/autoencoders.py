 # -*- coding: utf-8 -*-
"""
Created on Mon Jan 4 10:43:07 2022

@author: filipa
"""

import math
import tensorflow 
import os
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
tensorflow.compat.v1.disable_eager_execution()
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils import rmse, r_square

class autoencoder:
    
    """Gene Expression Profile and Mutation Autoencoder Model Class"""
    def __init__(self, config):
        self.exp_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())

        self.dataset = None
        self.config = config
        self.data_train = None
        self.data_test = None

    def load_data(self, data_type):
        """ Loads gene expression and mutation data
        """
        
        print("Loading data...")
        if data_type == 'exp':
            self.dataset = pd.read_pickle(self.config.exp_file)
        elif data_type == 'mut':
            self.dataset = pd.read_pickle(self.config.mut_file)

    def pre_process_data(self, data_type):
        """ Split datasets
        """
        
        if self.config.cross_validation == 'False':
            self.data_train, self.data_test = train_test_split(self.dataset, test_size=0.20, random_state=55)
            self.data_test, self.data_val = train_test_split(self.data_test, test_size=0.50, random_state=55)
            
        else:
            data_train, self.data_test = train_test_split(self.dataset, test_size=0.15, random_state=55)
            
            cross_validation_split = KFold(n_splits = 5, shuffle=True, random_state=42)
            self.data_cv = list(cross_validation_split.split(data_train))
            
            # Save splitted dataset
            # Utils.save_splitted_dataset(self.data_train,self.data_test, data_type)
        
    def build(self):
        """ Builds the autoencoder architecture"""
        
        print("Building model...")     
               
        input_data = tensorflow.keras.layers.Input(shape=(self.config.inp_dimension,))
    
        for i, layer in enumerate(self.config.layers_enc_units):
            encoder = tensorflow.keras.layers.Dense(layer, activation="relu")(input_data if i == 0 else encoder)
            encoder = tensorflow.keras.layers.BatchNormalization()(encoder)
            encoder = tensorflow.keras.layers.Dropout(self.config.dropout)(encoder) 

        latent_encoding = tensorflow.keras.layers.Dense(self.config.latent_dim)(encoder)

        self.encoder_model = tensorflow.keras.Model(input_data, latent_encoding)
        # self.encoder_model.summary()
        
        decoder_input = tensorflow.keras.layers.Input(shape=(self.config.latent_dim))
        for i, layer in enumerate(self.config.layers_dec_units):
            decoder = tensorflow.keras.layers.Dense(layer, activation="relu")(decoder_input if i == 0 else decoder)
            decoder = tensorflow.keras.layers.BatchNormalization()(decoder)
            decoder = tensorflow.keras.layers.Dropout(self.config.dropout)(decoder) 
        
        decoder_output = tensorflow.keras.layers.Dense(self.config.inp_dimension, activation = 'linear') (decoder)
        
        self.decoder_model = tensorflow.keras.Model(decoder_input, decoder_output)
        # self.decoder_model.summary()
        
        encoded = self.encoder_model(input_data)
        decoded = self.decoder_model(encoded)
        
        self.autoencoder = tensorflow.keras.models.Model(input_data, decoded)
    
       
    def train(self):
        """Compiles and trains the model"""
        pte = 200
        if self.config.optimizer == 'adam':
            opt = tensorflow.keras.optimizers.Adam(learning_rate = 0.001)
            
        es = EarlyStopping(verbose=1, patience=pte, restore_best_weights=True)
        #Reduces the learning rate by a factor of 10 when no improvement has been see in the validation set for 2 epochs
        reduce_lr = tensorflow.keras.callbacks.ReduceLROnPlateau(factor = 0.1, patience=10, min_lr = 1e-6)
        callbacks_list = [es,reduce_lr]
        
        self.autoencoder.compile(loss="mse", optimizer=opt, experimental_run_tf_function=False, metrics=[rmse, r_square])
        self.autoencoder.summary()
            
            
        self.autoencoder.fit(self.data_train, self.data_train, epochs=self.config.n_epochs, 
                            batch_size=self.config.batch_size, validation_data=(self.data_val, self.data_val), 
                            callbacks= callbacks_list)
        
    def evaluate(self):
        """Predicts resuts for the test dataset"""

        test_set = self.data_test
        preds_test = self.autoencoder.predict(test_set)
        
        self.mse = round(mean_squared_error(test_set, preds_test, squared = True),4)
        self.rmse = math.sqrt(self.mse)
        self.mae = round(mean_absolute_error(test_set, preds_test),4)
        self.r2 = round(r2_score(test_set, preds_test),4)

        print('\nRMSE test: ', self.rmse)
        print('\nMSE test: ', self.mse)
        
        return self.rmse, self.mse, self.mae, self.r2
        

    def save(self, data_type):
        
        models = ['Encoder','Decoder', 'Autoencoder']
        
        dirs = os.path.join('../experiments',self.exp_time+'//')
            
       	try:
            if not os.path.exists(dirs):
                os.makedirs(dirs)

       	except Exception as err:
       		print('Creating directories error: {}'.format(err))
       		exit(-1)
               
        for model in models:
            if model == 'Encoder':
                model_name = self.encoder_model
            elif model == 'Decoder':
                model_name = self.decoder_model
            elif model == 'Autoencoder':
                model_name = self.autoencoder
                
 
            model_name.save(dirs+model+"_"+data_type+".h5")
            print("\n" + model + " successfully saved to disk")
            

                        