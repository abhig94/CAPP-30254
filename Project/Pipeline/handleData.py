#Michael Fosco, Abhi Gupta, and Daniel Roberts


import numpy as np 
import pandas as pd 
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
import os, timeit, sys, itertools, re, time, requests, random, functools, logging
#from sklearn import linear_model, neighbors, ensemble, svm, preprocessing
from numba.decorators import jit, autojit
from numba import double #(nopython = True, cache = True, nogil = True [to run concurrently])
from sklearn import preprocessing, cross_validation, svm, metrics, tree, decomposition, svm
from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
#from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
#from sklearn.neighbors.nearest_centroid import NearestCentroid
#from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.grid_search import ParameterGrid
#from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler

###############################################################
'''
Read in the data functions
'''

'''
converts from camel case to snake case
Taken from:  http://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-camel-case
'''
def camel_to_snake(column_name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', column_name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

'''
Read data from a csv
'''
def readcsv(filename,index_col=None):
    assert(type(filename) == str and filename.endswith('.csv'))
    #assert(type(index_col) is int or type(index_col) is None)
    try:
        if index_col is not None:
            data = pd.read_csv(filename,index_col=index_col)
        else:
            data = pd.read_csv(filename)
    except:
        if index_col is not None:
            data = pd.read_csv(filename,index_col=index_col,engine='python')
        else:
            data = pd.read_csv(filename,engine='python')
    #data.columns = [camel_to_snake(col) for col in data.columns]
    return data




"""
Explore Data
"""

def explore_data(smata,col_list,save_toggle=False,file_prefix=''):
    """
    Takes a DataFrame as input and produces summary plots.
    save_toggle controls whether plots are saved.
    """
    data = smata[col_list]
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
            #temp = categorical_fields[col].value_counts()  
            #fig, ax = plt.subplots(1,1)
            #ax.hist(temp,bins=15)  
            fig = numeric_fields[col].hist(bins=20)
            fig.set_title(col)
            if save_toggle:
                plt.savefig(file_prefix+col+'_hist.png')
            plt.show()
    
    return
    
    
def identify_important_features(X,y,max_plot_feats,save_toggle=False,file_prefix='',show = False):
    """
    takes a response series and a matrix of features, and uses a random
    forest to rank the relative importance of the features for predicting
    the response.
    
    Based on code from the DataGotham2013 GitHub repo and scikit learn docs
    """    
    if len(file_prefix)>0:
        file_prefix += '_'

    forest = RandomForestClassifier()
    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    sorted_indices = np.argsort(importances)[::-1]
    sorted_indices = sorted_indices[0:max_plot_feats]
    print("Feature ranking:")

    for f in range(max_plot_feats):
        print("%d. feature %d %s (%f)" % (f + 1, sorted_indices[f], X.columns[sorted_indices[f]], importances[sorted_indices[f]]))

    padding = np.arange(len(sorted_indices)) + 0.5
    plt.barh(padding, importances[sorted_indices],color='r', align='center',xerr=std[sorted_indices])
    plt.yticks(padding, X.columns[sorted_indices])
    plt.xlabel("Relative Importance")
    plt.title("Variable Importance")
    plt.tight_layout()
    if save_toggle:
        plt.savefig(file_prefix+'important_features.png')
    if show == True:
        plt.show()
    return X.columns[sorted_indices]
        
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

def get_q_24(data):
    '''
    @return: process data to turn q_24 into binary with only its valid rows
    @input: data            The data to process
    This function depends on replace_value.
    '''
    df = data[data.q24 != 5]
    df = df[df.q24 != 6]

    df = replace_value(df, ['q24'], 1, 0)
    df = replace_value(df, ['q24'], 2, 0)
    df = replace_value(df, ['q24'], 3, 1)
    return replace_value(df, ['q24'], 4, 1)    
    
"""
Process Data
""" 
def replace_value(data,target_cols,value,replacement):
    """
    replaces the target value with the replacement value
    target_cols is a list of strings
    """
    for col in target_cols:
#       if value is np.NaN:
#           data[col] = data[col].fillna(replacement)
#       else:
            #data[col] = data[col].where(data[col]==value,replacement,inplace=True)
        data[col] = data[col].replace(value,replacement)
    return data

def fill_missing(data,target_cols=None,replacement=None):
    """
    fills in missing values using unconditional mode/median as appropriate
    can do the following:
        1) replace each column's missing vals with the mode/median as appropriate
        2) replace the missing values in all cols with a single value
        3) replace each column's missing vals with the corresponding entry in replacement

    target_cols: either None (fills all cols) or a list of columns names
    replacement: either a single replacement val, a list of vals the same length as target_cols/
    the number of cols in the data, or None (uses mean/median)
    """
    if target_cols is None:
        target_cols == list(data.columns)

    if replacement == None:
        numeric_fields = data.select_dtypes([np.number])
        categorical_fields = data.select_dtypes(['object','category'])
        
        if len(categorical_fields.columns) > 0:
            for col in [x for x in target_cols if x in categorical_fields.columns]:
                ind = pd.isnull(data[col])
                fill_val = data[col].mode()[0]
                data.ix[ind,col] = fill_val
            
        if len(numeric_fields.columns) > 0:    
            for col in [x for x in target_cols if x in numeric_fields.columns]:
                ind = pd.isnull(data[col])
                fill_val = data[col].median()
                data.ix[ind,col] = fill_val
    else:
        try:
            if len(replacement)==len(target_cols):
                for i,col in enumerate(target_cols):
                    data = replace_value(data,col,np.NaN,replacement[i])
            else:
                data = replace_value(data,target_cols,np.NaN,replacement)
        except: 
            data = replace_value(data,target_cols,np.NaN,replacement)
            
    return data
    
    
def trim_tails(data,target_cols,threshold = 95):
    """
    trims excessively heavy tails
    """
    if type(threshold) in [int,float] or len(threshold) == 1:
        threshold = [threshold]*len(target_cols)
    for i,col in enumerate(target_cols):
        cap = np.percentile(data[col],threshold[i])
        if threshold[i] >=50:
            data[col] = data[col].where(data[col]<=cap,cap)
        else:
            data[col] = data[col].where(data[col]>=cap,cap)
    return data
"""
def transform_data(data,transform,target_cols,name):
    """
    applies an arbitrary transform to the data
    """
    for col in target_cols:
        data[col+'_'+name] = transform(data[col])
    return data

def normalize_data(data,target_cols):
    for col in target_cols:
        data[col+'_normalized'] = normalize(data[col])
    return data

def robust_scale_data(data,target_cols):
    #scaler = RobustScaler().fit(data[target_cols])
    #data[target_cols] = scaler.tranform(data[target_cols])
    data[target_cols] = robust_scale(data[target_cols])
    return data

def scale_data(data,target_cols):
    #scaler =  StandardScaler().fit(data[target_cols])
    #data[target_cols] = scaler.tranform(data[target_cols])
    data[target_cols] = scale(data[target_cols])
    return data
"""
def macaroni(train_data, test_data, target_cols, method):
    train_based_meth = preprocessing.method.fit(train_data[target_cols])
    train_data[target_cols] = train_based_meth.transform(train_data[target_cols])
    test_data[target_cols] = train_based_meth.transform(test_data[target_cols])
    return train_data, test_data
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
            #data[col+'_binned'],temp = pd.cut(data[col],bins,retbins=True)
            data[col],temp = pd.cut(data[col],bins,retbins=True) 
            #bins_mat[col+'_binned'] = temp
            bins_mat[col] = temp
            #data.drop(col, axis=1, inplace=True)
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
            #data[col+'_binned'] = pd.cut(data[col],bin_mat[col+'_binned'])
            data[col] = pd.cut(data[col],bin_mat[col])
            #data.drop(col, axis=1, inplace=True)
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
        data.drop(col, axis=1, inplace=True)
    return data
