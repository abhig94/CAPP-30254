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
    pipe.explore_data(train,True,'train')  
    os.chdir('..')
    
    # process data
    train = pipe.process_data(train)
    test.ix[:,1:] = pipe.process_data(test.ix[:,1:])
    
    # split into X and y
    feature_vars = train.columns[1:]
    response_var = train.columns[0]
    
    X = train[feature_vars]
    y = train[response_var]
    
    # identify features of interest
    os.chdir('plots')
    pipe.identify_important_features(X,y,True,'train')
    pipe.x_vs_y_plots(X,y,True,'train')
    os.chdir('..')
     
    # fit and test accuracy of a logistic regression 
    logit_clf = pipe.build_classifier(X,y,'logistic_reg')
    logit_acc = pipe.evaluate_classifier(X,y,logit_clf)
    print('logit model accuracy: ', '%.4f' % logit_acc)
    
    # fit and test a K nearest neighbors model, with K=10
    KNN_clf = pipe.build_classifier(X,y,'KNN',{'n_neighbors': 10})
    KNN_acc = pipe.evaluate_classifier(X,y,logit_clf)
    print('KNN model accuracy: ', '%.4f' % KNN_acc)
    
    # fit and test a linear SVM 
    svm_clf = pipe.build_classifier(X,y,'linear_SVM',{'penalty':'l2'})
    svm_acc = pipe.evaluate_classifier(X,y,svm_clf)
    print('Linear SVM model accuracy: ', '%.4f' % svm_acc)
    
        
            
    
    
    
    
    