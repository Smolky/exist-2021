import csv
import sys
import string
import numpy as np
import pandas as pd
import os

from tqdm import tqdm
from .Dataset import Dataset
from scipy import stats


class DatasetExist (Dataset):
    """
    DatasetExist
    
    The Oxford English Dictionary defines sexism as “prejudice, stereotyping or discrimination, 
    typically against women, on the basis of sex”. Inequality and discrimination against 
    women that remain embedded in society is increasingly being replicated online.
    

    TASK 1: Sexism Identification
    The first subtask is a binary classification. The systems have to decide whether or not a given
    text (tweet or gab) is sexist (i.e., it is sexist itself, describes a sexist situation or criticizes a sexist
    behaviour), and classifies it according to two categories: “sexist” and “non-sexist”.

    TASK 2: Sexism Categorization
    Once a message has been classified as sexist, the second task aims to categorize the message
    according to the type of sexism. In particular, we propose a five-classification task:
    “ideological-inequality”, “stereotyping-dominance”, “objectification”, “sexual-violence” and
    “misogyny-non-sexual-violence”.
    
    @link http://nlp.uned.es/exist2021/
    
    @extends Dataset
    """

    def __init__ (self, dataset, options, corpus = '', task = '', refresh = False):
        """
        @inherit
        """
        Dataset.__init__ (self, dataset, options, corpus, task, refresh)
    
    def compile (self):
        
        # Load dataframes
        dfs = []
        for index, dataframe in enumerate (['EXIST2021_training.tsv', 'EXIST2021_test.tsv']):
            
            # Open file
            df_split = pd.read_csv (self.get_working_dir ('dataset', dataframe), sep = '\t')
            
            
            # Remove data that it is not from my language
            df_split = df_split.drop (df_split[df_split['language'] != self.options['language']].index)
            
            
            # Determine split
            df_split = df_split.assign (__split = 'train' if index == 0 else 'test')
            
            
            # Merge
            dfs.append (df_split)
        
        
        # Concat and assign
        df = pd.concat (dfs, ignore_index = True)
        
        
        # Change class names
        df = df.rename (columns = {
            'text': 'tweet', 
            'task1': 'label',
            'task2': 'category',
            'id': 'twitter_id', 
        })
        
        
        # @var training_indexes Sample validation test
        training_indexes = df[df['__split'] == 'train'].sample (frac = 0.2)
        df.loc[training_indexes.index, '__split'] = 'val'
        
        
        # Drop non-needed columns
        df = df.drop (['test_case'], axis = 1)
        
        
        # Store this data on disk
        self.save_on_disk (df)
        
        
        # Return
        return df
        