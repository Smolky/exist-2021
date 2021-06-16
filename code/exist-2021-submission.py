"""
    EXIST-2021 results
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import sys

import regex as re
import os.path
import glob
import pandas as pd

from pathlib import Path

from tqdm import tqdm


def main ():
    
    # @var directory_path String
    directory_path = '../assets/exist/*.csv'
    
    
    # @var ensemble Dict
    ensemble = {}
    
    
    # Iterate over filenames
    for filename in glob.glob (directory_path):
    
        # @var task String
        task = 'task-1' if 'task-1' in filename else 'task-2'
        
        
        # @var features String
        features = os.path.basename (filename)
        features = re.sub ('output-task-[0-9]-(deep-learning|transformers)-', '', features)
        features = features.replace ('.csv', '')
        
        
        # @var 
        df = pd.read_csv (filename)
        df = df.assign (test_case = 'EXIST2021')
        df = df[['test_case', 'id', 'label']]
        df = df.rename ({'label': task + '-' + features}, axis = 1)
        df['id'] = df['id'].astype (str)
        df['id'] = df['id'].apply (lambda x: '{0:0>6}'.format (x))
        
        
        # Discard
        if df.shape[0] != 4368:
            continue
        
        
        ensemble[task + '-' + features] = df 


    # Generate ensemble
    ensemble_df = pd.concat (ensemble.values (), axis = 1)
    ensemble_df = ensemble_df.loc[:,~ensemble_df.columns.duplicated ()]
    
    
    # Filter by task
    filter_col_task_1 = [col for col in ensemble_df if col.startswith ('task-1-')]
    filter_col_task_2 = [col for col in ensemble_df if col.startswith ('task-2-')]


    def save_run (df, task, run):

        # @var output_path String
        output_path = '../assets/exist/exist2021_UMUTEAM/'


        # @var filename String
        filename = output_path + "task" + str (task) + "_UMU_" + str (run)


        # Save to disk
        df.to_csv (filename, sep = "\t", index = False, header = False)
        
        
        print (df)
        print (df.iloc[:,-1:].value_counts (normalize = True))


    print ("lf")
    print ("----------")
    save_run (ensemble['task-1-lf'], 1, 1)
    save_run (ensemble['task-2-lf'], 2, 1)
    
    
    print ("bert+lf")
    print ("----------")
    save_run (ensemble['task-1-lf-bert-lf'], 1, 2)
    save_run (ensemble['task-2-lf-bert-lf'], 2, 2)
    
    
    
    task_1_df = ensemble_df[filter_col_task_1]
    task_2_df = ensemble_df[filter_col_task_2]
    
    
    # Assign new label
    task_1_df['label'] = task_1_df.mode (axis = 'columns')[0]
    task_2_df['label'] = task_2_df.mode (axis = 'columns')[0]
    task_1_df['test_case'] = ensemble_df['test_case']
    task_2_df['test_case'] = ensemble_df['test_case']
    task_1_df['id'] = ensemble_df['id']
    task_2_df['id'] = ensemble_df['id']
    
    
    print ("ensemble")
    print ("--------")
    
    save_run (task_1_df[['test_case', 'id', 'label']], 1, 3)
    save_run (task_2_df[['test_case', 'id', 'label']], 2, 3)

if __name__ == "__main__":
    main ()