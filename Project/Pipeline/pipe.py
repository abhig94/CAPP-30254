from sklearn.tree import DecisionTreeRegressor
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

from sklearn import preprocessing, cross_validation, metrics, tree, decomposition, grid_search
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid
from sklearn.preprocessing import StandardScaler, normalize, RobustScaler
from scipy import optimize
import statsmodels.api as sm
import matplotlib.pyplot as plt
import multiprocessing as mp
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
import os, timeit, sys, itertools, re, time, requests, random, functools, logging, csv, datetime
from sklearn import linear_model, neighbors, ensemble, svm, preprocessing
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

from handleData import *

'''
Model Dictionaries
'''

criteriaHeader = ['AUC', 'Accuracy', 'classifier', 'Precision at .05',
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
'''
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


simple_modelSVC = {'model': svm.LinearSVC}
simple_modelLR = {'model': LogisticRegression}
simple_modelRF  = {'model': RandomForestClassifier}
simple_modelNB  = {'model': GaussianNB}
simple_modelDT  = {'model': DecisionTreeClassifier, 'max_depth': [50], 'max_features': ['sqrt'],
            'min_samples_split': [50]}
simple_modelDTR = {'model': DecisionTreeRegressor}
'''

#modelList = [simple_modelDT,simple_modelLR]

'''

main looping funcs

based on code from my group project repository

'''

def evaluate_model(y, pred_probs, train_times, test_times, accuracies, classifier, test_weights, sample_weights = False):
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
        if sample_weights == True:
            try:
                prec = [metrics.precision_score(y[j], preds[j], sample_weight = test_weights[j]) for j in y_range]
                rec = [metrics.recall_score(y[j], preds[j], sample_weight = test_weights[j]) for j in y_range]
                f1_score = [metrics.f1_score(y[j],preds[j], sample_weight = test_weights[j]) for j in y_range]
            except:
                prec = [metrics.precision_score(y[j], preds[j]) for j in y_range]
                rec = [metrics.recall_score(y[j], preds[j]) for j in y_range]
                f1_score = [metrics.f1_score(y[j],preds[j]) for j in y_range]

        else:
            prec = [metrics.precision_score(y[j], preds[j]) for j in y_range]
            rec = [metrics.recall_score(y[j], preds[j]) for j in y_range]
            f1_score = [metrics.f1_score(y[j],preds[j]) for j in y_range]
        prec_std = np.std(prec)
        rec_std = np.std(rec)
        #print('check 2')
        f1_std = np.std(f1_score)

        prec_m = np.mean(prec)
        rec_m = np.mean(rec)
        f1_m = np.mean(f1_score)
        res[levels[x]] = str(prec_m) + ' (' + str(prec_std) + ')'
        res[recalls[x]] = str(rec_m) + ' (' + str(rec_std) + ')'
        res['f1 at ' + str(thresh)] = str(f1_m) + ' (' + str(f1_std) + ')'

    if sample_weights:
        try:
            auc = [metrics.roc_auc_score(y[j], pred_probs[j], sample_weight = test_weights[j]) for j in y_range]
        except:
            auc = [metrics.roc_auc_score(y[j], pred_probs[j]) for j in y_range]
    else:
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

'''
Return a result string as "mean (std)"
'''
def makeResultString(mean, std):
    return str(mean) + ' (' + str(std) + ')' 

'''
Return a dictionary of a bunch of criteria. Namely, this returns a dictionary
with precision and recall at .05, .1, .2, .25, .5, .75, AUC, time to train, and
time to test.
'''
def getCriterionsNoProb(yTests, predProbs, train_times, test_times, accuracies, called, test_weights, sample_weights = False):
    levels = ['Precision at .05', 'Precision at .10', 'Precision at .2', 'Precision at .25', 'Precision at .5', 'Precision at .75', 'Precision at .85']
    recalls = ['Recall at .05', 'Recall at .10', 'Recall at .20', 'Recall at .25', 'Recall at .5', 'Recall at .75', 'Recall at .85']
    amts= [.05, .1, .2, .25, .5, .75, .85]
    tots = len(amts)
    res = {}
    critsLen = len(yTests)
    critsRange = range(0, critsLen)
    res['classifier'] = called
    for x in range(0, tots):
        thresh = amts[x]

        res[levels[x]] = ''
        res[recalls[x]] = ''
        res['f1 at ' + str(thresh)] = ''

    if sample_weights:
        try:
            auc = [metrics.roc_auc_score(yTests[j], predProbs[j], sample_weight = test_weights[j]) for j in critsRange]
        except:
            auc = [metrics.roc_auc_score(yTests[j], predProbs[j]) for j in critsRange]
    else:
        auc = [metrics.roc_auc_score(yTests[j], predProbs[j]) for j in critsRange]
    aucStd = np.std(auc)
    aucM = np.mean(auc)
    trainM = np.mean(train_times)
    trainStd = np.std(train_times)
    testM = np.mean(test_times)
    testStd = np.std(test_times)
    accM = np.mean(accuracies)
    accStd = np.std(accuracies)

    res['AUC'] = makeResultString(aucM, aucStd)
    res['train_time (sec)'] = makeResultString(trainM, trainStd)
    res['test_time (sec)'] = makeResultString(testM, testStd)
    res['Accuracy'] = makeResultString(accM, accStd)

    return res

'''
Map partial preds and their indices to correct spot
for full preds.
'''
def get_full_preds(partialPreds, partialIndicies, n):
    res = [0]*n

    predLen = len(partialPreds)

    for j in range(0, predLen):
        partLen = len(partialPreds[j])
        tmpIndic = partialIndicies[j]
        tmpPred = partialPreds[j]

        for i in range(0, partLen):
            res[tmpIndic[i]] = tmpPred[i]

    return res

'''
Custon ensemble method.
'''
def clf_loop_revolutions(X,y,k,clf_list,discr_var_names, bin_nums, s_weights,  sample_weights = False,macro_run = True, col_name_frag= 'region'):
    results = []
    indx = 1
    indexer = 1

    cols = X.columns
    subsects = [c for c in cols if col_name_frag in c]
    yLen = len(y)
    catcher = {}

    for item in subsects:
        y_use = y[X[item] == 1]
        x_use = X[X[item] == 1]
        n = len(x_use)
        x_use_index = range(0, n)
        x_use['Index'] = x_use_index
        weight_use = np.ravel(s_weights[X[item] == 1].as_matrix())
        for clf_d in clf_list:
            print("\nIter: " + str(indexer) + "\n")
            param_grid = parameter_grid(clf_d)
            total = len(param_grid)
            res = [None]*total
            z = 0
            kf = cross_validation.KFold(len(y_use), k)
            for params in param_grid:
                carryOnMyWayWardSon = True
                print("Starting: "  + str(clf_d['model']))
                clf = clf_d['model'](**params)
                #try:
                train_times = [None]*k
                pred_probs = [None]*k
                test_times = [None]*k
                y_tests = [None]*k
                accs = [None]*k
                test_weights = [None]*k
                indx = 0
                noProb = False
                tmp_sample_weights = sample_weights
                partial_preds_indices = [None]*k
                for train, test in kf:
                    XTrain_init, XTest_init = x_use._slice(train, 0), x_use._slice(test, 0)
                    partial_preds_indices[indx] = list(XTest_init['Index'])
                    del XTrain_init['Index']
                    del XTest_init['Index']
                    yTrain, yTest = y_use._slice(train, 0), y_use._slice(test, 0)
                    train_cross_weights = np.ravel(s_weights._slice(train, 0).as_matrix())
                    test_cross_weights = np.ravel(s_weights._slice(test, 0).as_matrix())
                    test_weights[indx] = test_cross_weights
                    y_tests[indx] = yTest

                    XTrain_discrete, train_bins = discretize(XTrain_init, discr_var_names, bin_nums)
                    XTrain_update = create_dummies(XTrain_discrete, discr_var_names)

                    XTest_discrete = discretize_given_bins(XTest_init, discr_var_names, train_bins)

                    XTest_update = create_dummies(XTest_discrete, discr_var_names)
                    if macro_run == True:
                        macro_var_names = readcsv('macro_var_names.csv')
                    
                        macro_var_names_list = macro_var_names.values.tolist()
                        macro_names = [val for sublist in macro_var_names_list for val in sublist]

                    #CHANGE THIS METHOD IF DESIRED
                    #===========================
                        method = preprocessing.StandardScaler()
                    #===========================

                        XTrain, XTest = macaroni(XTrain_update, XTest_update, macro_names, method)
                    else:
                        XTrain, XTest = XTrain_update, XTest_update

                    start = time.time()

                    if sample_weights == True:
                        try:
                            fitted = clf.fit(XTrain, yTrain, train_cross_weights)
                        except:
                            res[z] = {}
                            carryOnMyWayWardSon = False
                            break
                    else:
                        fitted = clf.fit(XTrain, yTrain)
                    #pdb.set_trace()
                    t_time = time.time() - start
                    train_times[indx] = t_time
                    start_test = time.time()
                    try:
                        pred_prob = fitted.predict_proba(XTest)[:,1]
                    except:
                        start_test = time.time()
                        pred_prob = fitted.predict(XTest)
                        noProb = True

                    test_time = time.time() - start_test
                    test_times[indx] = test_time
                    pred_probs[indx] = pred_prob

                    if sample_weights == True and tmp_sample_weights:
                        accs[indx] = fitted.score(XTest,yTest, test_cross_weights)
                    else:
                        accs[indx] = fitted.score(XTest,yTest)

                    indx += 1
                if carryOnMyWayWardSon:
                    print('done training')
                    model_name = str(clf)


                    if not noProb:
                        evals = evaluate_model(y_tests, pred_probs, train_times, test_times, accs, model_name, s_weights, tmp_sample_weights)
                    else:
                        evals = getCriterionsNoProb(y_tests, pred_probs, train_times, test_times, accs, model_name, s_weights, tmp_sample_weights)
                    print('done evaluating')
                    print(evals['AUC'])
                    evals['Subsection'] = str(item)
                    res[z] = evals

                full_preds = get_full_preds(pred_probs, partial_preds_indices, n)

                print('done getting pred_probs')
                
                if model_name in catcher.keys():
                    catcher[model_name].update({str(item):(full_preds, noProb)})
                else:
                    catcher[model_name] = {str(item):(full_preds, noProb)}
                z +=1
                s= str(z) + '/' + str(total)
                print(s)

            results += res 
            indexer +=1

    fulls=[None]*len(catcher.keys())
    spot = 0
    for key in catcher.keys():
        fulls[spot] = getFullModel(catcher[key], X,y, s_weights,  sample_weights, col_name_frag, str(key))
        spot += 1
    results += fulls
    return [z for z in results if z != None and z != {}]

'''
Get full model statistic from revolutions
'''
def getFullModel(hTable, X, y, s_weights, sample_weights, col_name_frag, modelName):
    yLen = len(y)
    fullPred = [0]*yLen


    cols = X.columns
    subsects = [c for c in cols if col_name_frag in c]

    for s in subsects:
        tmpPred = hTable[s][0]
        i = 0
        z = 0
        for item in X[s]:
            if item != 0:
                fullPred[i] = tmpPred[z]
                z += 1
            i += 1

    sFP = [1 if v >= .5 else 0 for v in fullPred]
    accs = accuracy_score(y, sFP)
    if not hTable[s][1]:
        evals = evaluate_model([y], [fullPred], [0], [0], [accs], modelName + "_full", s_weights, sample_weights)
    else:
        evals = getCriterionsNoProb([y], [fullPred], [0], [0], [accs], modelName + "_full", s_weights, sample_weights)
    return evals

def clf_loop_reloaded(X,y,k,clf_list,discr_var_names, bin_nums, weights, train_sample_weights = False, test_sample_weights = False, macro_run = False):
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
            carry_on_son = True
            print("Starting: "  + str(clf_d['model']))
            clf = clf_d['model'](**params)
            #try:
            train_times = [None]*k
            pred_probs = [None]*k
            test_times = [None]*k
            y_tests = [None]*k
            accs = [None]*k
            test_weights = [None]*k
            indx = 0
            noProb = False
            for train, test in kf:
                XTrain_init, XTest_init = X._slice(train, 0), X._slice(test, 0)
                yTrain, yTest = y._slice(train, 0), y._slice(test, 0)
                train_cross_weights = np.ravel(weights._slice(train, 0).as_matrix())
                test_cross_weights = np.ravel(weights._slice(test, 0).as_matrix())
                test_weights[indx] = test_cross_weights
                y_tests[indx] = yTest

                XTrain_discrete, train_bins = discretize(XTrain_init, discr_var_names, bin_nums)
                XTrain_update = create_dummies(XTrain_discrete, discr_var_names)

                XTest_discrete = discretize_given_bins(XTest_init, discr_var_names, train_bins)
                XTest_update = create_dummies(XTest_discrete, discr_var_names)
                if macro_run == True:
                    macro_var_names = readcsv('macro_var_names.csv')
                    
                    macro_var_names_list = macro_var_names.values.tolist()
                    macro_names = [val for sublist in macro_var_names_list for val in sublist]

                    #CHANGE THIS METHOD IF DESIRED
                    #===========================
                    method = preprocessing.StandardScaler()
                    #===========================

                    XTrain, XTest = macaroni(XTrain_update, XTest_update, macro_names, method)
                else:
                    XTrain, XTest = XTrain_update, XTest_update
                start = time.time()
                if train_sample_weights == True:
                    try:
                        fitted = clf.fit(XTrain, yTrain, train_cross_weights)
                    except:
                        print("fuck me")
                        print(params['solver'])
                        res[z] = {}
                        carry_on_son = False
                        break
                else:
                    fitted = clf.fit(XTrain, yTrain)
                #pdb.set_trace()
                t_time = time.time() - start
                train_times[indx] = t_time
                start_test = time.time()
                try:
                    pred_prob = fitted.predict_proba(XTest)[:,1]
                except:
                    start_test = time.time()
                    pred_prob = fitted.predict(XTest)
                    noProb = True

                test_time = time.time() - start_test
                test_times[indx] = test_time
                pred_probs[indx] = pred_prob
                if test_sample_weights == True:
                    accs[indx] = fitted.score(XTest,yTest, test_cross_weights)
                else:
                    accs[indx] = fitted.score(XTest,yTest)
                indx += 1
            print('done training')
            if carry_on_son:
                if not noProb:
                    evals = evaluate_model(y_tests, pred_probs, train_times, test_times, accs, str(clf),test_weights, test_sample_weights)
                else:
                    evals = getCriterionsNoProb(y_tests, pred_probs, train_times, test_times, accs, str(clf),test_weights, test_sample_weights)
                print('done evaluating')
                print(evals['AUC'])
                res[z] = evals
            #except:
            #    print("Invalid params: " + str(params))
            #    continue
            z +=1
            s= str(z) + '/' + str(total)
            print(s)

        results += res 
        indx +=1

    return [z for z in results if z != None and z != {}]



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


'''

output functions

'''

def extractPredsItem(listy, keyName):
    res = [None]*len(listy)

    length  = len(listy)
    for j in range(0, length):
        res[j] = listy[j][1][keyName]
    return [z for z in res if z != None]


def write_results_to_file(file_name, d, has_pred_probs = False, pred_file_name = 'Prediction_probs.csv'):
    '''
    if has_pred_probs:
        header = [x for x in d[0][0].keys()]
        header.sort()
        fin = format_data(header, d[0])
        fin.insert(0, header)

        header2 = extractPredsItem(d, 'Model-subsect')
        fin2 = extractPredsItem(d, 'Prob_preds')
        fin2.insert(0, header2)

        try:
            with open(file_name, "w") as file_out:
                writer = csv.writer(file_out)
                for f in fin:
                    writer.writerow(f)
                file_out.close()
        except:
            print('writing failed')

        try:
            with open(pred_file_name, "w") as fout:
                writer = csv.writer(fout)
                for f in fin2:
                    writer.writerow(f)
                fout.close()
        except:
            print('Pred_probs writing failed')
        return
    '''

    header = [x for x in d[0].keys()] # header of eval criteria
    header.sort()
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
        keyList = x.keys()
        for j in header:
            if j in keyList:
                tmp[indx] = x[j]
            else:
                tmp[indx] = ''
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
    str_to_num = lambda x: float(x[0:x.index('(')]) if type(x) is str else x 
    for col in target_cols:
        data[col] = data[col].apply(str_to_num)
        data[col] = data[col].fillna(0)
    data = data.sort_index(axis=1)
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
    criteria = ['AUC', 'Accuracy', 'classifier', 'Precision at .05',
                  'Precision at .10', 'Precision at .2', 'Precision at .25', 'Precision at .5',
                  'Precision at .75','Precision at .85','Recall at .05','Recall at .10',
                  'Recall at .20','Recall at .25','Recall at .5','Recall at .75',
                  'Recall at .85','f1 at 0.05','f1 at 0.1','f1 at 0.2','f1 at 0.25',
                  'f1 at 0.5','f1 at 0.75','f1 at 0.85','test_time (sec)','train_time (sec)']

    if 'classifier' in criteria:
        criteria.remove('classifier')
    for metric in criteria:
        if 'sec' in metric:
            best = best_given_metric(data,metric,n=1,ascending_toggle=True).index[0]
        else:
            best = best_given_metric(data,metric,n=1,ascending_toggle=False).index[0]
        indices.append(best)
        metric_list.append(metric)
    output = data.ix[indices,:]
    output['best metric'] = metric_list
    cols = list(sorted(output.columns))
    cols.remove('classifier')
    cols.remove('best metric')
    cols = ['classifier','best metric'] + cols

    output.index = range(len(output))
    output = output[cols]

    return output

def compare_clf_across_metric(data,metric):
    """
    For the given metric, finds the parameterization of each clf that performed the best
    """
    criteria =  ['AUC', 'Accuracy', 'classifier', 'Precision at .05',
                  'Precision at .10', 'Precision at .2', 'Precision at .25', 'Precision at .5',
                  'Precision at .75','Precision at .85','Recall at .05','Recall at .10',
                  'Recall at .20','Recall at .25','Recall at .5','Recall at .75',
                  'Recall at .85','f1 at 0.05','f1 at 0.1','f1 at 0.2','f1 at 0.25',
                  'f1 at 0.5','f1 at 0.75','f1 at 0.85','test_time (sec)','train_time (sec)']

    indices = []
    assert metric in criteria
    if 'sec' in metric:
        ascending = True
    else:
        ascending = False

    tester = lambda x,y: y in x['classifier']
    for clf in modelNames:
        clf_subset = data[data.apply(lambda x: tester(x,clf),1)]
        if len(clf_subset)>0:
            best = best_given_metric(clf_subset,metric,1,ascending).index[0]
            indices.append(best)
    print(indices)
    output = data.ix[indices,:]
    cols = list(sorted(output.columns))
    cols.remove('classifier')
    cols = ['classifier'] + cols

    output.index = range(len(output))
    output = output[cols]
    return output


def plot_precision_recall_from_results(data,target_rows):
    """"
    Uses output from clean_results, compare_clf_across_metric, best_by_each_metric
    and plots precision recall curves for each row in target_rows
    """

    precision_cols = sorted([x for x in data.columns if 'Precision' in x])
    recall_cols = sorted([x for x in data.columns if 'Recall' in x])
    grab_thresh = lambda x: float(x[x.index('.'):])
    thresholds = [grab_thresh(x) for x in precision_cols]
    fig,ax = plt.subplots(1,1)

    for r in target_rows:
        x = [data.ix[r,col] for col in precision_cols]
        y = [data.ix[r,col] for col in recall_cols]
        ax.plot(x,y,label=data.ix[r,'classifier'])
    #ax.legend(loc='bottom left')
    plt.tight_layout()
    plt.show()
    return

