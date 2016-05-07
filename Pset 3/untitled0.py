# -*- coding: utf-8 -*-
"""
Created on Fri May  6 23:08:49 2016

@author: Abhi
"""


import pandas as pd
import numpy as np
import matplotlib
import statsmodels.api as sm
import os
from matplotlib import pyplot as plt
import seaborn as sns
import pickle

import pipeline as pipe


if __name__ == '__main__':

    train = pipe.read_data('cs-training.csv')  
    train = pipe.fill_missing(train)
    
    original_features = train.columns[1:]
    response_var = train.columns[0]

    binned_features = ['age','number_of_open_credit_lines_and_loans'] 
    train,bins = pipe.discretize(train,binned_features)  
    
    new_train = train.drop(binned_features,1)
    new_train = train.drop([x+'_binned' for x in binned_features],1)
    
    new_features = new_train.columns
    
    n_models = len(pipe.modelNames)
    big_results = [] 

    for i in range(2):
        model_name = pipe.modelNames[i]
        results = pipe.clf_loop(new_train[new_features],new_train[response_var],3,[pipe.modelList[i]])
        big_results += results        
        #with open('raw_results_'+model_name,'wb') as f:
        #    pickle.dump(results,f)
        pipe.write_results_to_file('results2_'+model_name+'.csv',results)
    
    pipe.write_results_to_file('all_results.csv',big_results)
    

    all_results = pd.read_csv('all_results.csv')
    criteria = pipe.criteriaHeader
    if 'Function called' in criteria:
        criteria.remove('Function called')
    all_results = pipe.clean_results(all_results,criteria)
    best_clfs = pipe.best_by_each_metric(all_results)
    best_clfs.to_csv('best_clfs.csv')
    
    