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
    
    heavy_tail_club = ['revolving_utilization_of_unsecured_lines',
                                   'debt_ratio','monthly_income']
    train = pipe.trim_tails(train,heavy_tail_club,95)    
    
    original_features = train.columns[1:]
    response_var = train.columns[0]
    
    binned_features = ['age','number_of_open_credit_lines_and_loans'] 
    train,bins = pipe.discretize(train,binned_features)  
        
    train = pipe.create_dummies(train,[x+'_binned' for x in binned_features])
    new_train = train.drop(binned_features,1)
    new_train = train.drop([x+'_binned' for x in binned_features],1)
    
    new_features = list(new_train.columns)
    new_features.remove(response_var)
    
    n_models = len(pipe.modelNames)
    big_results = [] 

    for i in [8]:#range(n_models):
        model_name = pipe.modelNames[i]
        results = pipe.clf_loop(new_train[new_features],new_train[response_var],4,[pipe.modelList[i]])
        big_results += results        
        #with open('raw_results_'+model_name,'wb') as f:
        #    pickle.dump(results,f)
        pipe.write_results_to_file('results_'+model_name+'.csv',results)
    
    results_list = []
    for model_name in pipe.modelNames:
        results_list.append(pd.read_csv('results_'+model_name+'.csv'))
        
    all_results = pd.concat(results_list)
    all_results.to_csv('all_results.csv')
    
    #all_results = pd.read_csv('all_results.csv')
    criteria = pipe.criteriaHeader
    if 'Function called' in criteria:
        criteria.remove('Function called')
    all_results = pipe.clean_results(all_results,criteria)
    best_clfs = pipe.best_by_each_metric(all_results)
    best_clfs.to_csv('best_clfs.csv')
    
    comparison = pipe.compare_clf_acoss_metric(all_results,'AUC')
    comparison.to_csv('comparison_of_clfs.csv')
    