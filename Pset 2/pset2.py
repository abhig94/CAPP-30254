# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 17:21:29 2016

@author: Abhi
"""

import pandas as pd
import numpy as np
import matplotlib
import statsmodels.api as sm

from matplotlib import pyplot as plt
import seaborn as sns

import pipeline as pipe



if __name__ == '__main__':
    
    # import data
    df = pipe.read_data('cs-training.csv')
    
    # plots and summary tables
    pipe.explore_data(df,save_toggle=True)
    
    # process data
    df = pipe.process_data(df)
    
    # split into X and y
    feature_vars = df.columns[1:]
    response_var = df.columns[0]
    
    X = df[feature_vars]
    y = df[response_var]
    
    # identify features of interest
    pipe.identify_important_features(X,y)
    pipe.x_vs_y_plots(X,y)
     
    # fit and test accuracy of a logistic regression 
    logit_clf = pipe.build_classifier(X,y,'logistic_reg')
    logit_acc = pipe.evaluate_classifier(X,y,logit_clf)
    print('logit model accuracy: ', '%.4f' % logit_acc)
    
    # fit and test a K nearest neighbors model, with K=10
    KNN_clf = pipe.build_classifier(X,y,'KNN',{'n_neighbors': 10})
    KNN_acc = pipe.evaluate_classifier(X,y,logit_clf)
    print('KNN model accuracy: ', '%.4f' % KNN_acc)
    
    # fit and test a linear SVM 
    svm_clf = pipe.build_classifier(X,y,'linear_SVM')
    svm_acc = pipe.evaluate_classifier(X,y,svm_clf)
    print('Linear SVM model accuracy: ', '%.4f' % svm_acc)
            
    
    
    
    
    