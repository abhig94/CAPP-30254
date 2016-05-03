# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 14:22:00 2016

@author: Abhi


A set of ML pipeline functions
"""

import pandas as pd
import numpy as np
import matplotlib
import requests
import json
import statsmodels.api as sm
import re
import sklearn
from sklearn import linear_model, neighbors, ensemble, svm

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn import preprocessing, cross_validation, svm, metrics, tree, decomposition, svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
import random

"""
Read data
"""
def camel_to_snake(column_name):
    """
    converts from camel case to snake case

    Taken from:  http://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-camel-case
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
    numeric_fields = data.select_dtypes([np.number])
    categorical_fields = data.select_dtypes(['object','category'])
    if len(file_prefix)>0:
        file_prefix += '_'
    
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
            summary_stats.to_csv(file_prefix+'categorical_summary.csv')
        
        for col in categorical_fields.columns:             
            fig = categorical_fields[col].value_counts().plot(kind = 'bar')
            fig.set_title(col)
            if save_toggle:
                plt.savefig(file_prefix+col+'_hist.png')
            plt.show()
    
    if len(numeric_fields.columns) > 0:
        summary_stats = numeric_fields.describe()
        print(summary_stats)
        if save_toggle:
            summary_stats.to_csv(file_prefix+'numeric_summary.csv')
        
        for col in numeric_fields.columns:
            fig = numeric_fields[col].hist(bins=100)
            fig.set_title(col)
            if save_toggle:
                plt.savefig(file_prefix+col+'_hist.png')
            plt.show()
    
    return
    
    
def identify_important_features(X,y,save_toggle=False,file_prefix=''):
    """
    takes a response series and a matrix of features, and uses a random
    forest to rank the relative importance of the features for predicting
    the response.
    
    Based on code from the DataGotham2013 GitHub repo and scikit learn docs
    """    
    if len(file_prefix)>0:
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
    if len(file_prefix)>0:
        file_prefix += '_'
        
    df = pd.concat([X, pd.DataFrame(y, index=X.index)], axis=1)
    for x in X.columns:
        df[[x,y.name]].groupby(x).mean().plot()
        if save_toggle:
            plt.savefig(file_prefix+x+'_vs_'+y.name+'.png')
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
        if threshold >=50:
            data[col] = data[col].where(data[col]<=cap,cap)
        else:
            data[col] = data[col].where(data[col]>=cap,cap)
    return data
    
"""
Generate Features
"""

def discretize(data,target_cols,bins=10):
    """
    discretizes and returns target columns
        
    """
    if type(bins) is int:
        bins_mat = pd.DataFrame()
        for col in target_cols:
            data[col+'_binned'],temp = pd.cut(data[col],bins,retbins=True) 
            bins_mat[col+'_binned'] = temp
        return data,bins_mat
    else:
        raise TypeError('invalid arguments given')
        return

def discretize_given_bins(data,target_cols,bin_mat):
    """
    discretizes and returns target cols using bins in a corresponding
    data frame
    """
    if type(bin_mat) in [pd.DataFrame,pd.Series]: 
        for col in target_cols:
            data[col+'_binned'] = pd.cut(data[col],bin_mat[col+'_binned'])
        return data
    else:
        raise TypeError('invalid arguments given')
        return


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
def build_classifier(X,y,method='LR',params={}):
    """
    takes a dataset and a method, returns the fitted model
    """
    
    clfs, param_grid = define_clfs_params()
            
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
    
def predict_probs(X,clf):
    """
    takes a classifier and a set of features, and returns predicted probabilities
    """
    return clf.predict_proba(X)

"""
Ghani Code
"""
def define_clfs_params():

    clfs = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
        'ET': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
        'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
        'LR': LogisticRegression(penalty='l1', C=1e5),
        'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
        'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
        'NB': GaussianNB(),
        'DT': DecisionTreeClassifier(),
        'SGD': SGDClassifier(loss="hinge", penalty="l2"),
        'KNN': KNeighborsClassifier(n_neighbors=3) 
            }

    grid = { 
    'RF':{'n_estimators': [1,10,100,1000,10000], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
    'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
    'ET': { 'n_estimators': [1,10,100,1000,10000], 'criterion' : ['gini', 'entropy'] ,'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
    'GB': {'n_estimators': [1,10,100,1000,10000], 'learning_rate' : [0.001,0.01,0.05,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20,50,100]},
    'NB' : {},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
           }
    return clfs, grid

def clf_loop(models_to_run,grid,X,y,clfs):
    """
    loops over different classifiers and various parameter choices
    models_to_run: list of strings that name classifiers
    grid: dictionary that maps clfs to dictionaries of paramters and their values
    clfs: dictionary of classifiers
    """
    
    for n in range(1, 2):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        for index,clf in enumerate([clfs[x] for x in models_to_run]):
            print(models_to_run[index])
            parameter_values = grid[models_to_run[index]]
            for p in ParameterGrid(parameter_values):
                try:
                    clf.set_params(**p)
                    print(clf)
                    y_pred_probs = clf.fit(X_train, y_train).predict_proba(X_test)[:,1]
                        
                    # add evaluation stuff in here
                    
                    #threshold = np.sort(y_pred_probs)[::-1][int(.05*len(y_pred_probs))]
                    #print threshold
                    print(precision_at_k(y_test,y_pred_probs,.05))
                    #plot_precision_recall_n(y_test,y_pred_probs,clf)
                except IndexError as e:
                    print('Error:',e)
                    continue


def plot_precision_recall_n(y_true, y_prob, model_name):
    from sklearn.metrics import precision_recall_curve
    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    
    name = model_name
    plt.title(name)
    #plt.savefig(name)
    plt.show()

def precision_at_k(y_true, y_scores, k):
    threshold = np.sort(y_scores)[::-1][int(k*len(y_scores))]
    y_pred = np.asarray([1 if i >= threshold else 0 for i in y_scores])
    return metrics.precision_score(y_true, y_pred)

    
