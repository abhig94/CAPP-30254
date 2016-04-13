# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 14:22:00 2016

@author: Abhi


A set of ML pipeline functions
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import re
import sklearn
from sklearn import linear_model, neighbors, ensemble, svm
import scipy.stats as stat
from matplotlib import pyplot as plt
import seaborn as sns

import pdb

"""
Read data
"""
def camel_to_snake(column_name):
    """
    converts from camel case to snake case

    Taken from  http://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-camel-case
    via the DataGotham2013 GitHub repo    
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', column_name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def read_data(filename):
    """
    Takes the name of a file to be read, returns a DataFrame object.
    filename should be a string. 
    """
    assert(type(filename)==str and filename.endswith('.csv'))
    data = pd.read_csv(filename,index_col=0)
    data.columns = [camel_to_snake(col) for col in data.columns]
    return data
    

"""
Explore Data
"""

def explore_data(data,save_toggle=False,file_prefix=''):
    """
    Takes a DataFrame as input and produces summary plots.
    save_toggle controls whether plots are saved.
    """
    if len(file_prefix) > 0:
        file_prefix += '_'
    numeric_fields = data.select_dtypes([np.number])
    categorical_fields = data.select_dtypes(['object','category'])
    
    if len(categorical_fields.columns) > 0:
        summary_stats = pd.DataFrame(index=['mode','num_missing'],
                                  columns=categorical_fields.columns)
        for col in data.columns:
            summary_stats.ix['num_missing',col] = (len(data.index)-data.count())[col]
            try:
                summary_stats.ix['mode',col] = data[col].mode()[0]
            except:
                continue  
            
        print(summary_stats)   
        if save_toggle:
            summary_stats.to_csv(file_prefix+'summary_stats_numeric.csv')        
        
        for col in categorical_fields.columns:             
            fig = sns.distplot(numeric_fields[col].dropna())
            #categorical_fields[col].value_counts().plot(kind = 'bar')
            fig.set_title(col)
            if save_toggle:
                plt.savefig(file_prefix+col+'_hist.png')
            else:
                plt.show()
    
    if len(numeric_fields.columns) > 0:
        summary_stats = numeric_fields.describe()
        print(summary_stats)
        if save_toggle:
            summary_stats.to_csv(file_prefix+'summary_stats_numeric.csv')
        
        for col in numeric_fields.columns:
            fig = sns.distplot(numeric_fields[col].dropna()) 
            #numeric_fields[col].hist(bins=100)
            fig.set_title(col)
            if save_toggle:
                plt.savefig(file_prefix+col+'_hist.png')
            else:
                plt.show()
                
    return
    
    
def identify_important_features(X,y,save_toggle=False,file_prefix=''):
    """
    takes a response series and a matrix of features, and uses a random
    forest to rank the relative importance of the features for predicting
    the response.
    
    Based on code from the DataGotham2013 GitHub repo and scikit learn docs
    """    
    if len(file_prefix) > 0:
        file_prefix += '_'

    forest = ensemble.RandomForestClassifier()
    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    sorted_indices = np.argsort(importances)[::-1]

    padding = np.arange(len(X.columns)) + 0.5
    plt.barh(padding, importances[sorted_indices],color='r', align='center',xerr=std[sorted_indices])
    plt.yticks(padding, X.columns[sorted_indices])
    plt.xlabel("Relative Importance")
    plt.title("Variable Importance")
    plt.tight_layout()
    if save_toggle:
        plt.savefig(file_prefix+'important_features.png')
    plt.show()
        
def x_vs_y_plots(X,y,save_toggle=False,file_prefix=''):
    """
    Plot x vs y for each x in X
    """
    if len(file_prefix) > 0:
        file_prefix += '_'    
    
    df = pd.concat([X, pd.DataFrame(y, index=X.index)], axis=1)
    for x in X.columns:
        df[[x,y.name]].groupby(x).mean().plot()
        if save_toggle:
            plt.savefig(file_prefix+x+'_vs_'+y.name+'.png')
        else:
            plt.show()
    return
    
    
"""
Process Data
"""

def process_data(data):
    """
    fills in missing values using unconditional mode/median as appropriate
    """
    numeric_fields = data.select_dtypes([np.number])
    categorical_fields = data.select_dtypes(['object','category'])
    
    if len(categorical_fields.columns) > 0:
        for col in categorical_fields.columns:
            ind = pd.isnull(data[col])
            fill_val = data[col].mode()[0]
            data.ix[ind,col] = fill_val
        
    if len(numeric_fields.columns) > 0:    
        for col in numeric_fields.columns:
            ind = pd.isnull(data[col])
            fill_val = data[col].median()
            data.ix[ind,col] = fill_val
            
    return data
    
    
def trim_tails(data,target_cols,threshold = 95):
    """
    trims excessively heavy tails
    """
    for col in target_cols:
        cap = np.percentile(data[col],threshold)
        if threshold >=.5:
            data[col] = data[col].where(data[col]<=cap,cap)
        else:
            data[col] = data[col].where(data[col]>=cap)
    return data
    
"""
Generate Features
"""

def discretize(data,target_cols,n_bins=10):
    """
    discretizes and returns target columns
    """
    for col in target_cols:
        data[col+'_binned'] = pd.cut(data[col],n_bins)  
    return data


def create_dummies(data,target_cols):
    """
    creates dummy variables from categorical ones and return resulting DataFrame
    """
    for col in target_cols:
        temp = pd.get_dummies(data[col],prefix=col)
        data = pd.concat([data, pd.DataFrame(temp, index=data.index)], axis=1)
        #data.drop(col, axis=1, inplace=True)
    return data


"""
Build classifier
"""
def build_classifier(X,y,method='logistic_reg',params={}):
    """
    takes a dataset and a method, returns the fitted model
    """
    
    clfs = {'logistic_reg':linear_model.LogisticRegression,
            'KNN':neighbors.KNeighborsClassifier,
            'random_forest':ensemble.RandomForestClassifier,
            'linear_SVM':svm.LinearSVC}
            
    assert(method in clfs.keys())
    clf = clfs[method](**params)
    clf.fit(X,y)
    print(clf)
    return clf
    
    
    
"""
Evaluate classifier
"""
def evaluate_classifier(X,y,clf):
    """
    Evaluates the accuracy of the fitted classifier
    and prints a small table to illustrate results
    """
    predicted = predict_values(X,clf)
    accuracy = np.round(np.sum(predicted==y)/len(y),6)
    print(pd.crosstab(y, clf.predict(X), rownames=["Actual"], colnames=["Predicted"]))
    return accuracy
    
def predict_values(X,clf):
    """
    takes a classifier and a set of features, and returns predicted values
    """    
    return clf.predict(X)
    
    
    