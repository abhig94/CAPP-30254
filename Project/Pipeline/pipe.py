#Michael Fosco, Abhi Gupta, Daniel Roberts

import numpy as np 
import pandas as pd 
import statsmodels.api as sm
import matplotlib.pyplot as plt
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
from sklearn.preprocessing import StandardScaler
from time import time 
from handleData import *

'''
Note: model.predict( x) predicts from model using x
'''
##################################################################################
'''
List of models and parameters
'''
criteriaHeader = ['AUC', 'Accuracy', 'Function called', 'Precision at .05',
 				  'Precision at .10', 'Precision at .2', 'Precision at .25', 'Precision at .5',
 				  'Precision at .75','Precision at .85','Recall at .05','Recall at .10',
 				  'Recall at .20','Recall at .25','Recall at .5','Recall at .75',
 				  'Recall at .85','f1 at 0.05','f1 at 0.1','f1 at 0.2','f1 at 0.25',
 				  'f1 at 0.5','f1 at 0.75','f1 at 0.85','test_time (sec)','train_time (sec)']

modelNames = ['LogisticRegression', 'KNeighborsClassifier', 'RandomForestClassifier', 'ExtraTreesClassifier',
			  'AdaBoostClassifier', 'SVC', 'GradientBoostingClassifier', 'GaussianNB', 'DecisionTreeClassifier',
			  'SGDClassifier']
n_estimMatrix = [5, 10, 25, 50, 100, 200, 1000, 10000]
depth = [1, 5, 10, 20, 50, 100]
cpus = mp.cpu_count()
cores = cpus-1
modelLR = {'model': LogisticRegression, 'solver': ['liblinear'], 'C' : [.01, .1, .5, 1, 5, 10, 25],
		  'class_weight': ['balanced', None], 'n_jobs' : [cores],
		  'tol' : [1e-5, 1e-4, 1e-3, 1e-1, 1], 'penalty': ['l1', 'l2']} #tol also had 1e-7, 1e-4, 1e-1
#took out linear svc because it did not have predict_proba function
#modelLSVC = {'model': svm.LinearSVC, 'tol' : [1e-7, 1e-5, 1e-4, 1e-3, 1e-1, 1], 'class_weight': ['balanced', None],
#			 'max_iter': [1000, 2000], 'C' :[.01, .1, .5, 1, 5, 10, 25]}
modelKNN = {'model': neighbors.KNeighborsClassifier, 'weights': ['uniform', 'distance'], 'n_neighbors' : [2, 5, 10, 50, 100, 500, 1000, 10000],
			'leaf_size': [15, 30, 60, 120], 'n_jobs': [cpus/4]} 
modelRF  = {'model': RandomForestClassifier, 'n_estimators': [25, 50, 100, 1000], 'criterion': ['gini', 'entropy'],
			'max_features': ['sqrt', 'log2'], 'max_depth': depth, 'min_samples_split': [2, 5, 10, 20, 50],
			'bootstrap': [True, False], 'n_jobs':[cores]}
#have to redo just the one below
modelET  = {'model': ExtraTreesClassifier, 'n_estimators': [25, 50, 100, 1000], 'criterion': ['gini', 'entropy'],
			'max_features': ['sqrt', 'log2'], 'max_depth': depth,
			'bootstrap': [True, False], 'n_jobs':[cores]}
#base classifier for adaboost is automatically a decision tree
modelAB  = {'model': AdaBoostClassifier, 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [5, 10, 25, 50, 100, 1000]}
modelSVM = {'model': svm.SVC, 'C':[0.00001,0.0001,0.001,0.01,0.1,1,10], 'max_iter': [5000, 50000], 'probability': [True], 
			'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}

# will have to change n_estimators when running this on the project data
modelGB  = {'model': GradientBoostingClassifier, 'learning_rate': [.001, 0.01,0.05,], 'n_estimators': [1,10,50, 100, 1000, 10000],
			'max_depth': depth, 'subsample' : [0.1, .2, .5, 1]} 
#Naive Bayes below
modelNB  = {'model': GaussianNB}
modelDT  = {'model': DecisionTreeClassifier, 'criterion': ['gini', 'entropy'], 'max_depth': [1, 5, 10,20,50, 100, 1000],
			'max_features': ['sqrt','log2'],'min_samples_split': [2, 5, 10, 20, 50]}
modelSGD = {'model': SGDClassifier, 'loss': ['modified_huber', 'perceptron'], 'penalty': ['l1', 'l2', 'elasticnet'], 
			'n_jobs': [cores]}

modelList = [modelLR, modelKNN, modelRF, modelET, 
			 modelAB, modelSVM, modelNB, modelDT,
			 modelSGD, modelGB]

##################################################################################



###############################################################
'''
Functions dealing with the actual pipeLine
'''

'''
Remove a key from a dictionary. Used in makeDicts.
'''
def removeKey(d, ey):
    r = dict(d)
    del r[key]
    return r

'''
Get X and y from a dataframe
'''
def getXY(df, yName):
	y = df[yName]
	X = df.drop(yName, 1)
	return (y,X)

'''
Wrapper for function, func, with arguments,
arg, coming in a dictionary.
'''
def wrapper(func, args):
	try:
		m = func(**args)
		return m
	except:
		return None

'''
Make all the requisite mini dictionaries from
the main dictionary for pipeline process.
'''
def makeDicts(d):
	result = []
	dN = removeKey(d, 'model')
	thingy  = dN.keys()
	l = [dN[x] for x in thingy]
	combos = list(itertools.product(*l))
	lengthy = len(combos)
	result = [0] * lengthy

	for i in range(0, lengthy):
		result[i] = dict(zip(thingy, combos[i]))

	return result

'''
Get a list of accuracies from a model list
'''
def getAccuracies(X, y, modelList):
	result = [0]*len(modelList)

	for i in range(0, len(modelList)):
		result[i] = modelList[i].score(X,y)
	return result

'''
Sort the list of models from best to worst
according to a second list of accuracies
'''
def bestModels(modelList, accList, rev = True):
	result = [x for (y,x) in sorted(zip(accList, modelList))]
	if rev:
		result.reverse()
	return result

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
def getCriterions(yTests, predProbs, train_times, test_times, accuracies, called):
	levels = ['Precision at .05', 'Precision at .10', 'Precision at .2', 'Precision at .25', 'Precision at .5', 'Precision at .75', 'Precision at .85']
	recalls = ['Recall at .05', 'Recall at .10', 'Recall at .20', 'Recall at .25', 'Recall at .5', 'Recall at .75', 'Recall at .85']
	amts= [.05, .1, .2, .25, .5, .75, .85]
	tots = len(amts)
	res = {}
	critsLen = len(yTests)
	critsRange = range(0, critsLen)
	res['Function called'] = called
	for x in range(0, tots):
		thresh = amts[x]
		preds = [np.asarray([1 if j >= thresh else 0 for j in z]) for z in predProbs]
		prec = [metrics.precision_score(yTests[j], preds[j]) for j in critsRange]
		rec = [metrics.recall_score(yTests[j], preds[j]) for j in critsRange]
		precStd = np.std(prec)
		recStd = np.std(rec)

		f1S = [2*(prec[j]*rec[j])/(prec[j]+rec[j]) for j in critsRange]
		f1Std = np.std(f1S)

		precM = np.mean(prec)
		recM = np.mean(rec)
		f1M = np.mean(f1S)
		res[levels[x]] = makeResultString(precM, precStd)
		res[recalls[x]] = makeResultString(recM, recStd)
		res['f1 at ' + str(thresh)] = makeResultString(f1M, f1Std)

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
Wrapper type function for own parallelization.
This will get prediction probabilities as well as the results
of a host of criteria described in getCriterions.
'''
def paralleled(item, X, y, k, modelType):
	s = str(item)
	logging.info('Started: ' + s)
	try:
		trainTimes = [None]*k
		predProbs = [None]*k
		testTimes = [None]*k
		yTests = [None]*k
		accs = [None]*k 
		kf = cross_validation.KFold(len(y), k)
		indx = 0
		wrapped = wrapper(modelType, item)
		for train, test in kf:
			XTrain, XTest = X._slice(train, 0), X._slice(test, 0)
			yTrain, yTest = y._slice(train, 0), y._slice(test, 0)
			yTests[indx] = yTest

			start = time()
			fitting = wrapped.fit(XTrain, yTrain)
			t_time = time() - start
			trainTimes[indx] = t_time
			start_test = time()
			predProb = fitting.predict_proba(XTest)[:,1]
			test_time = time() - start_test
			testTimes[indx] = test_time
			predProbs[indx] = predProb
			accs[indx] = fitting.score(XTest,yTest)
			indx +=1

		criteria = getCriterions(yTests, predProbs, trainTimes, testTimes, accs, str(wrapped))
	except:
		logging.info('Error with: ' + s)
		return None
	return criteria

'''
Same function as makeModels (below), but uses own parallelization.
This was written for the cases where sklearn did not have an 
n_jobs option and was not automatically parallelized. This will write 
the status of the parallelization to the file: status.log. 
'''
def makeModelsPara(X, y, k, d):
	global cores
	result = makeDicts(d)

	logging.info('\nStarted: ' + str(d['model']) + "\n")
	pool = mp.Pool(cores)
	res = pool.map(functools.partial(paralleled, X = X, y = y, k = k, modelType = d['model']), result)
	pool.close()
	pool.join()
	logging.info('\nEnded: ' + str(d['model']) + "\n")

	return res

'''
Fit a model and determine the results of a bunch of
criteria, namely precision at various levels and AUC.
'''
def makeModels(X, y, k, d):
	result = makeDicts(d)
	total = len(result)
	res = [None]*total

	z = 0
	kf = cross_validation.KFold(len(y), k)
	logging.info("\nStarting: " + str(d['model']) + '\n')
	for item in result:
			wrap = wrapper(d['model'], item)
			try:
				trainTimes = [None]*k
				predProbs = [None]*k
				testTimes = [None]*k
				yTests = [None]*k
				accs = [None]*k
				
				indx = 0
				for train, test in kf:
					XTrain, XTest = X._slice(train, 0), X._slice(test, 0)
					yTrain, yTest = y._slice(train, 0), y._slice(test, 0)
					yTests[indx] = yTest

					start = time()
					fitting = wrap.fit(XTrain, yTrain)
					t_time = time() - start
					trainTimes[indx] = t_time
					start_test = time()
					predProb = fitting.predict_proba(XTest)[:,1]
					test_time = time() - start_test
					testTimes[indx] = test_time
					predProbs[indx] = predProb
					accs[indx] = fitting.score(XTest,yTest)
					indx += 1

				criteria = getCriterions(yTests, predProbs, trainTimes, testTimes, accs, str(wrap))

				res[z] = criteria 
			except:
				print("Invalid params: " + str(item))
				continue
			z +=1
			s= str(z) + '/' + str(total)
			logging.info(s)
			print(s)
	logging.info("\nEnded: " + str(d['model']) + '\n')
	return res

'''
Retrieve criteria from results of pipeline.
No longer needed after adding k-fold cross val
def retrieveCriteria(results):
	fin = [x[1] for x in results if x[1] != None]
	return fin
'''

'''
Format the data to be in nice lists in the same 
order as the masterHeader.
'''
def formatData(masterHeader, d):
	length = len(d)
	format = [[]] * length
	lenMH = len(masterHeader)

	indxForm = 0
	indx = 0
	for x in d:
		tmp = [None] * lenMH
		for j in masterHeader:
			tmp[indx] = x[j]
			indx += 1
		indx = 0
		format[indxForm] = tmp
		indxForm += 1
	return format

'''
Write results of pipeline to file. Note, d is the 
variable that is returned by the pipeLine function call
Return:					0 for successful termination, -1 for error 
'''
def writeResultsToFile(fName, d):
	header = makeHeader(d)
	fin = formatData(header, d)
	fin.insert(0, header)
	try:
		with open(fName, "w") as fout:
			writer = csv.writer(fout)
			for f in fin:
				writer.writerow(f)
			fout.close()
	except:
		return -1
	return 0

'''
General pipeline for ML process. This reads data from a file and generates a 
training and testing set from it. It then fits a model and gets the models precision
at .05, .1, .2, .25, .5, .75, .85, and AUC. It returns a list of models fit as 
well as those model's results for each criterion.
'''
def pipeLine(y, X, lModels, k):
	#data = readcsv(name)
	#df = fillMethod(data)
	#y,X = getXY(df, yName)
	res = []
	indx = 1
	logging.basicConfig(filename='status.log',level=logging.DEBUG)

	for l in lModels:
		print("\nIter: " + str(indx) + "\n")
		#own parallelization if sklearn does not already do that
		if 'n_jobs' not in l and "<class 'sklearn.neighbors.classification.KNeighborsClassifier'>" not in l.values():
			models = makeModelsPara(X, y, k, l)
			res += models
			#normalize data in case of KNN
		elif "<class 'sklearn.neighbors.classification.KNeighborsClassifier'>" in l.values():
			Xtmp  = preprocessing.scale(X)
			models = makeModels(Xtmp, y, k, l)
			res += models 
		else:
			models = makeModels(X, y, k, l)
			res += models 

		indx +=1

	return [z for z in res if z != None]

'''
Plot precision recall curve given model, true y, and
predicted y probabilities.
'''
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