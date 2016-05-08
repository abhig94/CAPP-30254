# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 14:22:00 2016

@author: Abhi


A set of ML pipeline functions
"""
import pdb
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

from sklearn import preprocessing, cross_validation, metrics, tree, decomposition, grid_search
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler, normalize, RobustScaler
import random


import numpy as np 
import pandas as pd 
from scipy import optimize
import statsmodels.api as sm
import matplotlib.pyplot as plt
from patsy import dmatrices
import multiprocessing as mp
from multiprocessing import Process, Queue
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
import os, timeit, sys, itertools, re, time, requests, random, functools, logging, csv, datetime
import seaborn as sns
from sklearn import linear_model, neighbors, ensemble, svm, preprocessing
from numba.decorators import jit, autojit
from numba import double #(nopython = True, cache = True, nogil = True [to run concurrently])
from sklearn import preprocessing, cross_validation, svm, metrics, tree, decomposition, svm
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler, RobustScaler, robust_scale, scale
from time import time 


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
            #temp = categorical_fields[col].value_counts()  
            #fig, ax = plt.subplots(1,1)
            #ax.hist(temp,bins=15)  
            fig = numeric_fields[col].hist(bins=20)
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
def replace_value(data,target_cols,value,replacement):
    """
    replaces the target value with the replacement value
    """
    for col in target_cols:
        data[col] = data[col].where(data[col]==value,replacement)
    return data

def fill_missing(data):
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


'''

Older pipeline code

'''


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
    
    

def find_accuracy(X,y,clf):
    """
    Evaluates the accuracy of the fitted classifier
    """
    predicted = predict_values(X,clf)
    accuracy = np.round(np.sum(predicted==y)/len(y),6)
    #print(pd.crosstab(y, clf.predict(X), rownames=["Actual"], colnames=["Predicted"]))
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


'''
Model Dictionaries
'''

criteriaHeader = ['AUC', 'Accuracy', 'Function called', 'Precision at .05',
                  'Precision at .10', 'Precision at .2', 'Precision at .25', 'Precision at .5',
                  'Precision at .75','Precision at .85','Recall at .05','Recall at .10',
                  'Recall at .20','Recall at .25','Recall at .5','Recall at .75',
                  'Recall at .85','f1 at 0.05','f1 at 0.1','f1 at 0.2','f1 at 0.25',
                  'f1 at 0.5','f1 at 0.75','f1 at 0.85','test_time (sec)','train_time (sec)']

# no adaboost for now.
modelNames = ['LogisticRegression', 'KNeighborsClassifier', 'RandomForestClassifier', 'ExtraTreesClassifier',
              'AdaBoostClassifier', 'SVC', 'GaussianNB', 'DecisionTreeClassifier',
              'SGDClassifier']
n_estimMatrix = [5, 10, 25, 50, 100, 200]
depth = [10, 20, 50]
cpus = mp.cpu_count()
cores = cpus-1
modelLR = {'model': LogisticRegression, 'solver': ['liblinear'], 'C' : [.01, .1, .5, 1],#, 5, 10, 25],
          'class_weight': ['balanced', None], 'n_jobs' : [cores],
          'tol' : [1e-5, 1e-3, 1], 'penalty': ['l1', 'l2']} #tol also had 1e-7, 1e-4, 1e-1
modelKNN = {'model': neighbors.KNeighborsClassifier, 'weights': ['uniform', 'distance'], 'n_neighbors' : [100, 500, 1000],#2,5, 10, 50, 10000],
            'leaf_size': [60, 120]}#, 'n_jobs': [cpus/4]} #leaf size also had 15, 30
modelRF  = {'model': RandomForestClassifier, 'n_estimators': [25, 50, 100], 'criterion': ['gini', 'entropy'],
            'max_features': ['sqrt', 'log2'], 'max_depth': depth, 'min_samples_split': [20, 50], #min sample split also had 2, 5, 10
            'bootstrap': [True], 'n_jobs':[cores]} #bootstrap also had False

modelET  = {'model': ExtraTreesClassifier, 'n_estimators': [25, 50, 100], 'criterion': ['gini', 'entropy'],
            'max_features': ['sqrt', 'log2'], 'max_depth': depth,
            'bootstrap': [True, False], 'n_jobs':[cores]}

modelAB  = {'model': AdaBoostClassifier, 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [5, 10, 25, 50, 100]}#, 200]}
modelSVM = {'model': svm.SVC, 'C':[0.1,1], 'max_iter': [1000, 2000], 'probability': [True], 
            'kernel': ['rbf', 'poly', 'sigmoid', 'linear']} #C was: [0.00001,0.0001,0.001,0.01,0.1,1,10]
modelNB  = {'model': GaussianNB}
modelDT  = {'model': DecisionTreeClassifier, 'criterion': ['gini', 'entropy'], 'max_depth': [10,20,50], #had 100, 1, 5
            'max_features': ['sqrt','log2'],'min_samples_split': [10, 20, 50]} #had a 2,5
modelSGD = {'model': SGDClassifier, 'loss': ['modified_huber'], 'penalty': ['l1', 'l2', 'elasticnet'], 
            'n_jobs': [cores]}

modelList = [modelLR, modelKNN, modelRF, modelET, 
             modelAB, modelSVM, modelNB, modelDT,
             modelSGD] 



'''

main looping funcs

based on code from my group project repository

'''


def evaluate_model(y, pred_probs, train_times, test_times, accuracies, classifier):
    """
    Takes in y values, the associated probabilities, times, accuracies, and the
    name of the classifier. Returns a dictionary with AUC, acc, etc. statistics
    """

    levels = ['Precision at .05', 'Precision at .10', 'Precision at .2', 'Precision at .25', 'Precision at .5', 'Precision at .75', 'Precision at .85']
    recalls = ['Recall at .05', 'Recall at .10', 'Recall at .20', 'Recall at .25', 'Recall at .5', 'Recall at .75', 'Recall at .85']
    amts= [.05, .1, .2, .25, .5, .75, .85]
    res = {}
    y_range = range(0, len(y))
    res['classifier'] = classifier
    for x in range(0, len(amts)):
        #print('check 1')
        thresh = amts[x]
        #pdb.set_trace()
        preds = [np.asarray([1 if j >= thresh else 0 for j in z]) for z in pred_probs]
        prec = [metrics.precision_score(y[j], preds[j]) for j in y_range]
        rec = [metrics.recall_score(y[j], preds[j]) for j in y_range]
        prec_std = np.std(prec)
        rec_std = np.std(rec)
        #print('check 2')
        f1_score = [2*(prec[j]*rec[j])/(prec[j]+rec[j]) for j in y_range]
        f1_std = np.std(f1_score)

        prec_m = np.mean(prec)
        rec_m = np.mean(rec)
        f1_m = np.mean(f1_score)
        res[levels[x]] = str(prec_m) + ' (' + str(prec_std) + ')'
        res[recalls[x]] = str(rec_m) + ' (' + str(rec_std) + ')'
        res['f1 at ' + str(thresh)] = str(f1_m) + ' (' + str(f1_std) + ')'

    auc = [metrics.roc_auc_score(y[j], pred_probs[j]) for j in y_range]
    auc_std = np.std(auc)
    auc_m = np.mean(auc)
    train_m = np.mean(train_times)
    train_std = np.std(train_times)
    test_m = np.mean(test_times)
    test_std = np.std(test_times)
    acc_m = np.mean(accuracies)
    acc_std = np.std(accuracies)

    res['AUC'] = str(auc_m) + ' (' + str(auc_std) + ')'
    res['train_time (sec)'] = str(train_m) + ' (' + str(train_std) + ')'
    res['test_time (sec)'] = str(test_m) + ' (' + str(test_std) + ')'
    res['Accuracy'] = str(acc_m) + ' (' + str(acc_std) + ')' #mean_std_to_string(acc_m, acc_std)

    return res



def clf_loop(X,y,k,clf_list):
    results = []
    indx = 1

    for clf_d in clf_list:
        print("\nIter: " + str(indx) + "\n")
        param_grid = parameter_grid(clf_d)
        total = len(param_grid)
        res = [None]*total
        z = 0
        kf = cross_validation.KFold(len(y), k)

        for params in param_grid:
            clf = clf_d['model'](**params)
            try:
                train_times = [None]*k
                pred_probs = [None]*k
                test_times = [None]*k
                y_tests = [None]*k
                accs = [None]*k
                indx = 0
                for train, test in kf:
                    XTrain, XTest = X._slice(train, 0), X._slice(test, 0)
                    yTrain, yTest = y._slice(train, 0), y._slice(test, 0)
                    y_tests[indx] = yTest

                    start = time()
                    fitted = clf.fit(XTrain, yTrain)
                    #pdb.set_trace()
                    t_time = time() - start
                    train_times[indx] = t_time
                    start_test = time()
                    pred_prob = fitted.predict_proba(XTest)[:,1]
                    test_time = time() - start_test
                    test_times[indx] = test_time
                    pred_probs[indx] = pred_prob
                    accs[indx] = fitted.score(XTest,yTest)
                    indx += 1
                print('done training')
                evals = evaluate_model(y_tests, pred_probs, train_times, test_times, accs, str(clf))
                print('done evaluating')
                print(evals['AUC'])
                res[z] = evals
            except:
                print("Invalid params: " + str(params))
                continue
            z +=1
            s= str(z) + '/' + str(total)
            print(s)

        results += res 
        indx +=1

    return [z for z in results if z != None]





'''

support funcs

'''

def parameter_grid(old_d):
    '''
    Similar to ParameterGrid() from sklearn
    '''
    result = []
    d = old_d.copy()
    del d['model']
    #d = remove_key(old_d, 'model')
    keys  = d.keys()
    l = [d[x] for x in keys]
    combos = list(itertools.product(*l))
    num_combos = len(combos)
    result = [0] * num_combos

    for i in range(0, num_combos):
        result[i] = dict(zip(keys, combos[i]))

    return result



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



'''

output functions

'''

def write_results_to_file(file_name, d):
    header = [x for x in d[0].keys()] # header of eval criteria
    fin = format_data(header, d)
    fin.insert(0, header)
    try:
        with open(file_name, "w") as file_out:
            writer = csv.writer(file_out)
            for f in fin:
                writer.writerow(f)
            file_out.close()
    except:
        print('writing failed') 
    return 


def format_data(header, d):
    len_d = len(d)
    formatted = [[]] * len_d
    len_header = len(header)

    indxForm = 0
    indx = 0
    for x in d:
        tmp = [None] * len_header
        for j in header:
            tmp[indx] = x[j]
            indx += 1
        indx = 0
        formatted[indxForm] = tmp
        indxForm += 1
    return formatted



def clean_results(data,target_cols):
    """
    strips standard deviations from csv-derived DataFrame
    and replaces strings with floats
    """
    str_to_num = lambda x: float(x[0:x.index('(')])
    for col in target_cols:
        data[col] = data[col].apply(str_to_num)
        data[col] = data[col].fillna(0)
    return data

def best_given_metric(data,metric,n=5,ascending_toggle=False):
    """
    returns the n best classifiers by a given metric
    """
    assert metric in criteriaHeader
    sorted_data = data.sort_values(metric,ascending=ascending_toggle)
    return sorted_data.iloc[0:n,:]

def best_by_each_metric(data):
    """
    returns the best classifiers by each metric
    """
    indices = []
    metric_list = []
    criteria = criteriaHeader.copy()
    if 'Function called' in criteria:
        criteria.remove('Function called')
    for metric in criteria:
        if 'sec' in metric:
            best = best_given_metric(data,metric,n=1,ascending_toggle=True).index[0]
        else:
            best = best_given_metric(data,metric,n=1,ascending_toggle=False).index[0]
        indices.append(best)
        metric_list.append(metric)
    output = data.iloc[indices,:]
    output['best metric'] = metric_list
    cols = list(sorted(output.columns))
    cols.remove('classifier')
    cols.remove('best metric')
    cols = ['classifier','best metric'] + cols
    output = output.reindex_axis(cols, axis=1)
    return output

def compare_clf_acoss_metric(data,metric):
    """
    For the given metric, finds the parameterization of each clf that performed the best
    """
    indices = []
    assert metric in criteriaHeader
    if 'sec' in metric:
        ascending = True
    else:
        ascending = False

    tester = lambda x,y: y in x['classifier']
    for clf in modelNames:
        clf_subset = data[data.apply(lambda x: tester(x,clf),1)]
        best = best_given_metric(clf_subset,metric,1,ascending).index[0]
        indices.append(best)
    print(indices)
    output = data.iloc[indices,:]
    cols = list(sorted(output.columns))
    cols.remove('classifier')
    cols = ['classifier'] + cols
    output = output.reindex_axis(cols, axis=1)
    return output
