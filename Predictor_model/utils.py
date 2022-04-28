 # -*- coding: utf-8 -*-
"""
Created on Mon Jan 4 14:38:10 2022

@author: filipa
"""

import csv
import numpy as np
import json
import time
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from bunch import Bunch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem
from rdkit import DataStructs

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


def reading_csv(config, dataset_identifier):
    """
    config: configuration file
    dataset_identifier: dataset to use

    Returns
    -------
    smiles, labels: Lists with the loaded data 
    """
    if dataset_identifier == 'ups7':
        filepath = config.datapath_ups7
    elif dataset_identifier == 'jak2':
        filepath = config.datapath_jak2
    elif dataset_identifier == 'a2a':
        filepath = config.datapath_a2a
    elif dataset_identifier == 'kop':
        filepath = config.datapath_kop
    else:
        filepath = config.datapath_var
        
    raw_smiles = []
    raw_labels = []
    
    with open(filepath, 'r') as csvFile:
        reader = csv.reader(csvFile)
        it = iter(reader)
        next(it, None)  # skip first item.    
        for row in it:
            try:
                raw_smiles.append(row[0])
                raw_labels.append(float(row[1]))
            except:
                pass
    
    smiles = []
    labels = []
    
    # SMILES with length under a certain threshold defined in the configuration file
    for i in range(len(raw_smiles)):
        if len(raw_smiles[i]) <= config.smiles_len_threshold  and 'a' not in raw_smiles[i] and 'Z' not in raw_smiles[i] and 'K' not in raw_smiles[i]:
            smiles.append(raw_smiles[i])
            labels.append(raw_labels[i])

    return smiles, labels
           

def tokenize(smiles, tokens, len_threshold):
    """
    Transforms the SMILES strings into a list of tokens
    ----------
    smiles: Set of SMILES strings with different sizes;
    tokens: Set of characters;
    len_threshold: Integer that specifies the maximum size of the padding    
    
    Returns
    -------
    tokenized: returns the list with tokenized SMILES

    """  
    tokenized = []  
    for smile in smiles:
        i = 0
        token = []
        while (i < len(smile)):
            for j in range(len(tokens)):
                symbol = tokens[j]
                if symbol == smile[i:i + len(symbol)]:
                    token.append(symbol)
                    i += len(symbol)
                    break
                
        while len(token) < len_threshold:
            token.append(' ')
        tokenized.append(token)
        
    return tokenized

def token(mol):
    """ returns the list of tokens in a molecule """    
    tokens=[]
    iao='1'
    i=0
    while i < len(mol):
        x=mol[i]
        if x=='[':
            iao=x
        elif iao[-1]!=']' and iao[0]=='[':
            iao=iao+x     
            if iao[-1]==']':
                tokens.append(iao)
                iao='1'
        elif i<=len(mol)-2 and x=='B' and mol[i+1]=='r':
            tokens.append(x+mol[i+1])
            i+=1
        elif i<=len(mol)-2 and x=='C' and mol[i+1]=='l':
            tokens.append(x+mol[i+1])
            i+=1
        else:
            tokens.append(x)
        i+=1
    return tokens
        


def padding(molecules,lenfeatures):  
    """ Adds padding to all molecules, returns a list of all molecules with padding """      
    padMolecules = []
    for mol in molecules:
        molecule=token(mol)
        if len(molecule) <= lenfeatures:
            dif = lenfeatures-len(molecule)   
            for i in range(dif):
                molecule.append(' ')
            padMolecules.append(molecule)
    return padMolecules  
  
             
def smiles2idx(smiles,tokenDict):
    """
    Transforms each SMILES token to the correspondent integer, according the token-integer dictionary.
    ----------
    smiles: Set of SMILES strings with different sizes;
    tokenDict: Dictionary that maps the characters to integers;    
    -------
    newSmiles: Returns the transformed smiles, with the characters replaced by 
    the numbers. 
    """           
    newSmiles =  np.zeros((len(smiles), len(smiles[0])))
    for i in range(0,len(smiles)):
        for j in range(0,len(smiles[i])):
            
            newSmiles[i,j] = tokenDict[smiles[i][j]]
            
    return newSmiles
     

def smiles2ecfp(smiles, radius, bit_len=4096, index=None):
    """
    Transforms a list of SMILES strings into a list of ECFP.
    ----------
    smiles: List of SMILES strings to transform
    -------
    This function returns the SMILES strings transformed into a vector of 4096 elements
    """
    fps = np.zeros((len(smiles), bit_len))
    
    for i, smile in enumerate(smiles):
        mol = Chem.MolFromSmiles(smile)
        arr = np.zeros((1,))
        try:
            mol = MurckoScaffold.GetScaffoldForMol(mol)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=bit_len)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps[i, :] = arr
        except:
            print(smile)
            fps[i, :] = [0] * bit_len
            
    return pd.DataFrame(fps, index=(smiles if index is None else index))
    

def smiles2rdkit(smiles):
    """
    Transforms a list of SMILES strings into a list of rdkit fp.
    ----------
    smiles: List of SMILES strings to transform
    -------
    This function returns the SMILES strings transformed
    """
    
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    
    rdk_fp = [Chem.RDKFingerprint(mol) for mol in mols]

    return rdk_fp


def cv_split(data, config):
    """
    Data spliting into folds. Each fold is then used once as a test set 
    while the remaining folds form the training set.
    ----------
    config: configuration file;
    data: List with the list of SMILES strings set and a list with the label;
    -------
    data_cv: object that contains the indexes for training and testing for all folds
    """
    train_val_smiles = data[0]
    train_val_labels = data[1]
    
    cross_validation_split = KFold(n_splits=config.n_splits, shuffle=True, random_state=42)
    data_cv = list(cross_validation_split.split(train_val_smiles, train_val_labels))
    return data_cv


def normalize(y_train, y_val, y_test, norm_type):
    """
    Percentile normalization step (to avoid the interference of outliers).
    ----------
    data: List of label lists. It contains the y_train, y_test
    -------
    Returns data with normalized values. 
    """
    print(y_test)

    if norm_type == 'percentile':
        q1_train = np.percentile(y_train, 5)
        q3_train = np.percentile(y_train, 95)
        
        aux = [q1_train, q3_train]

        y_train = (y_train - q1_train) / (q3_train - q1_train)
        y_val = (y_val - q1_train) / (q3_train - q1_train)
        y_test  = (y_test - q1_train) / (q3_train- q1_train)

    elif norm_type == 'min_max':
        y_min = np.min(y_train)
        y_max = np.max(y_train)
        
        aux = [y_min, y_max]

        y_train = (y_train - y_min) / (y_max - y_min)
        y_val = (y_val - y_min) / (y_max - y_min)
        y_test  = (y_test - y_min) / (y_max - y_min)

    elif norm_type == 'mean_std': 
        y_mean= np.mean(y_train)
        y_std = np.std(y_train)

        aux = [y_mean, y_std]

        y_train = (y_train - y_mean) / y_std
        y_val = (y_val - y_mean) / y_std
        y_test  = (y_test - y_mean) / y_std
    
    print(y_test)
    # my_dict = {'Labels_train': np.reshape(y_train, -1) , 'Labels_val': np.reshape(y_val, -1), 'Labels_test': np.reshape(y_test, -1)}
    # fig, ax = plt.subplots()
    # ax.boxplot(my_dict.values())
    # ax.set_xticklabels(my_dict.keys())
   
    return aux, y_train, y_val, y_test

def denormalization(predictions, aux, norm_type):
    """
    This function implements the denormalization step.
    ----------
    predictions: Output from the model
    data: q3 and q1 values to perform the denormalization
    Returns
    -------
    Returns the denormalized predictions.
    """
    print(predictions)
    if norm_type == 'percentile':
        q1_train = aux[0]
        q3_train = aux[1]

        predictions = (q3_train - q1_train) * predictions + q1_train
    
    elif norm_type == 'min_max':
        y_min = aux[0]
        y_max = aux[1]

        predictions = (y_max - y_min) * predictions + y_min
    
    elif norm_type == 'mean_std':
        y_mean = aux[0]
        y_std = aux[1]

        predictions = (y_std) * predictions + y_mean

    print(predictions)

  
    return predictions

def pred_scatter_plot(real_values, pred_values, title, xlabel, ylabel, data_type):
    fig, ax = plt.subplots()
    ax.scatter(real_values, pred_values, c='tab:blue',
               alpha=0.6, edgecolors='black')
    ax.plot(real_values, real_values, 'k--', lw=4)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    plt.title(title)
    
    if data_type == 'norm':
        plt.xlim([-1, 2])
        plt.ylim([-1, 2])
    else:
        plt.xlim([-10, 12])
        plt.ylim([-10, 12])
    plt.show()
