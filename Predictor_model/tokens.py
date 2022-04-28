# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 11:08:18 2021

@author: filipa
"""

class tokens_table(object):
    """
    This class has a list with all the necessary tokens to build the SMILES strings
    Note that a token is not necessarily a character. It can be a two characters like Br.
    ----------
    tokens: List with symbols and characters used in SMILES notation.
    """
    def __init__(self):
        
        tokens = ['C', '1', '=', 'N', '(', 'S', ')', '2', 'O', '3', '4', 'F',
                  '[C@@H]', '#', 'Cl', '5', '6', '7', '8', '[C@H]', 'Br', '/', '\\',
                  '[C@@]', '[N+]', '[O-]', '.', 'P', '[Br-]', '9', 'I', '[C@]', '[Pt]', 'B', ' ']
        self.table = tokens
        self.table_len = len(self.table)