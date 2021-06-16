"""
    EXIST-2021 results
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import sys

import pandas as pd

from pathlib import Path

from tqdm import tqdm

from dlsdatasets.DatasetResolver import DatasetResolver
from dlsmodels.ModelResolver import ModelResolver
from features.FeatureResolver import FeatureResolver
from utils.Parser import DefaultParser


def main ():
    
    # var parser
    parser = DefaultParser (description = 'Exist-2021')


    # @var args Get arguments
    args = parser.parse_args ()
    
    
    # @var resolver Resolver
    resolver = DatasetResolver ()
    
    
    # @var model_resolver ModelResolver
    model_resolver = ModelResolver ()


    # @var results Dict
    results = {}
    
    
    # @var corpora List
    corpora = ['2021-es', '2021-en']
    
    
    # @var models List
    models = ['deep-learning', 'transformers', 'transformers-lf']
    
    
    # @var tasks List
    tasks = ['', 'task-2']


    # Iterate over corpora
    for task in tasks:
        for corpus in corpora:
            for model_key in models:
                def callback (feature_key, y_pred, y_real, models_per_architecture):
                    """
                    This callback is for storing the results individually
                    """
                    
                    print (task)
                    print (corpus)
                    print (feature_key)
                    print (model_key)

                    if task not in results:
                        results[task] = {}

                    if model_key not in results[task]:
                        results[task][model_key] = {}
                    
                    if feature_key not in results[task][model_key]:
                        results[task][model_key][feature_key] = []
                    
                    
                    # attach
                    results[task][model_key][feature_key].append (pd.DataFrame ({
                        'id': dataset.df['twitter_id'],
                        'label': y_pred,
                        'message': dataset.df['tweet']
                    }))

    
    
                # @var dataset Dataset
                dataset = resolver.get (args.dataset, corpus, task, False)
                
                
                # Determine if we need to use the merged dataset or not
                dataset.filename = dataset.get_working_dir (task, 'dataset.csv')
            
        
                # @var df DataFrame
                df = dataset.get ()
                
                
                # Get the labels
                dataset.set_true_labels (dataset.get_true_labels ())
                dataset.set_num_labels (dataset.get_num_labels ())    
                dataset.set_available_labels (dataset.get_available_labels ())
            
            
                # @var model Model
                model = model_resolver.get (model_key)
                model.set_dataset (dataset)
                model.is_merged (False)
            
            
                # Replace the dataset to contain only the test or val-set
                dataset.df = dataset.get_split (df, 'test')


                # @var feature_resolver FeatureResolver
                feature_resolver = FeatureResolver (dataset)
                
                
                # @var available_features List
                available_features = model.get_available_features ()
                    
                
                # Iterate over all available features
                for feature_set in available_features:
                
                    # @var features_cache String The file where the features are stored
                    if feature_set == 'lf':
                        features_cache = dataset.get_working_dir (args.task, feature_set + '_minmax_ig.csv')
                    else:
                        features_cache = dataset.get_working_dir (args.task, feature_set + '_ig.csv')


                    if not Path (features_cache).is_file():
                        features_cache = dataset.get_working_dir (args.task, feature_set + '.csv')

                
                    # @var transformer
                    transformer = feature_resolver.get (feature_set, cache_file = features_cache)


                    # Set the features in the model
                    model.set_features (feature_set, transformer)


                # Perform the prediction
                model.predict (using_official_test = True, callback = callback)

    
    # Create the final predictions
    for task, models in results.items ():
        for model_key, features in models.items ():
            for feature_key, dataframes in features.items ():
        
                # @var concat_df Dataframe
                concat_df = pd.concat (dataframes)
                
                
                # @var filename String
                filename = dataset.get_working_dir ('..', 'output-' + (task if task else 'task-1') + '-' + model_key + "-" + feature_key + '.csv')
                
                
                # Save on disk
                concat_df.to_csv (filename, index = False)

                print (task)
                print (model_key)
                print (feature_key)
                print (concat_df)
        
    

if __name__ == "__main__":
    main ()