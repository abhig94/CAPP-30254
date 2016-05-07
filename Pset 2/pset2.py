# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 17:21:29 2016

@author: Abhi
"""

import pandas as pd
import numpy as np
import matplotlib
import statsmodels.api as sm
import os
from matplotlib import pyplot as plt
import seaborn as sns

import pipeline as pipe



if __name__ == '__main__':
    
    # import data
    train = pipe.read_data('cs-training.csv')
    test = pipe.read_data('cs-test.csv')
    
    # explore data
    if 'plots' not in os.listdir():
        os.mkdir('plots')
        
    os.chdir('plots')
    pipe.explore_data(train,False,'train')  
    os.chdir('..')
    
    # process data
    train = pipe.process_data(train)
    test.ix[:,1:] = pipe.process_data(test.ix[:,1:])
    
    heavy_tail_club = ['revolving_utilization_of_unsecured_lines',
                                   'debt_ratio','monthly_income']
    train = pipe.trim_tails(train,heavy_tail_club,90)    
    test = pipe.trim_tails(test,heavy_tail_club,90)
    
    original_features = train.columns[1:]
    response_var = train.columns[0]
    
    binned_features = ['age','number_of_open_credit_lines_and_loans'] 
    train,bins = pipe.discretize(train,binned_features)  
    test = pipe.discretize_given_bins(test,binned_features,bins)
        
    train = pipe.create_dummies(train,[x+'_binned' for x in binned_features])
    test = pipe.create_dummies(test,[x+'_binned' for x in binned_features])

    
    new_train = train.drop(binned_features,1)
    new_train = train.drop([x+'_binned' for x in binned_features],1)
    new_test = test.drop(binned_features,1)
    new_test = test.drop([x+'_binned' for x in binned_features],1)
    
    new_features = new_train.columns
    
    # identify features of interest
    os.chdir('plots')
    pipe.identify_important_features(train[original_features],train[response_var],
                                     True,'train')
    pipe.x_vs_y_plots(train[original_features],train[response_var],
                      True,'train')
    os.chdir('..')
     
    # fit and test accuracy of a logistic regression 
    logit_clf = pipe.build_classifier(train[original_features],train[response_var],
                                          'LR')
    logit_acc = pipe.evaluate_classifier(train[original_features],train[response_var],
                                         logit_clf)
    print('logit model accuracy: ', '%.4f' % logit_acc)
    
    # fit and test a linear SVM 
    svm_clf = pipe.build_classifier(train[original_features],train[response_var],
                                        'SVM',{'penalty':'l2',})
    svm_acc = pipe.evaluate_classifier(train[original_features],train[response_var],
                                       svm_clf)
    print('Linear SVM model accuracy: ', '%.4f' % svm_acc)
    
    # make use of binned data
    binned_logit_clf =  pipe.build_classifier(new_train[new_features],new_train[response_var],
                                          'LR')
    binned_logit_acc = pipe.evaluate_classifier(new_train[new_features],new_train[response_var],
                                         binned_logit_clf)
    print('logit model accuracy: ', '%.4f' % binned_logit_acc)
    
    predictions = pipe.predict_values(test[original_features],logit_clf)
    np.savetxt("predictions.csv", predictions, delimiter=",")
    