#Michael Fosco, Abhi Gupta, and Daniel Roberts


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
import os, timeit, sys, itertools, re, time, requests, random, functools, logging
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

###############################################################
'''
Descriptive functions
'''

'''
Create a correlation table
'''
def corrTable(df, method = 'pearson', min_periods = 1):
	return df.corr(method, min_periods)

'''
Generate a descriptive Table
'''
def descrTable(df):
	sumStats = df.describe(include = 'all')
	missingVals = len(df.index) - df.count()

	oDF = pd.DataFrame(index = ['missing values'], columns = df.columns)

	for col in df.columns:
		oDF.ix['missing values',col] = missingVals[col]

	fDF = sumStats.append(oDF)
	return fDF

'''
Make bar plots
'''
def barPlots(df, items, saveExt = ''):
	for it in items:
		b = df[it].value_counts().plot(kind = 'bar', title = it)
		s = saveExt + it + '.pdf'
		b.get_figure().savefig(s)
		plt.show()

'''
Make pie plots.
'''
def piePlots(df, items, saveExt = ''):
	for it in items:
		b = df[it].value_counts().plot(kind = 'pie', title = it)
		s = saveExt + it + 'Bar.pdf'
		b.get_figure().savefig(s)
		plt.show()

'''
Discretize a continous variable. 
num: 		The number of buckets to split the cts variable into
'''
def discretize(df, cols, num=10):
	dDF = df
	for col in cols:
		dDF[col] = pd.cut(dDF[col], num)
	return dDF

'''
Convert categorical variables into binary variables
'''
def categToBin(df, cols):
	dfN = df
	for col in cols:
		dfN = pd.get_dummies(df[col])
	df_n = pd.concat([df, dfN], axis=1)
	return df_n

'''
Helper function to make histograms
'''
def makeHisty(ax, col, it, binny = 20):
	n, bins, patches = ax.hist(col, binns=binny, histtype='bar', range=(min(col), max(col)))

'''
Make histogram plots, num is for layout of plot
'''
def histPlots(df, items, fname, binns = 20, saveExt = ''):
	indx = 1

	num = len(items)
	iters = num % 4
	z = 0

	for i in range(0, iters):
		fig, axarr = plt.subplots(2, 2)
		x = 0
		y = 0

		for it in items[z:z+4]:
			makeHisty(axarr[x,y], df[it], it, binns)
			axarr[x,y].set_title(it)
			y += 1
			if y >= len(axarr):
				x += 1
				y = 0
			if x >= len(axarr):
				break
		fig.savefig(saveExt + fname + str(indx) + 'Hists.pdf')
		plt.clf()
		indx += 1
		z += 4

	leftover = num - z
	leftIts = items[z:]

	if leftover == 1:
		fig, axarr = plt.subplots(1,1)
		makeHisty(axarr[0], df[leftIts[0]], leftIts[0], binns)
		axarr[0].set_title(leftIts[0])
		fig.savefig(saveExt + fname + str(indx) + 'Hists.pdf')
		plt.clf()
	elif leftover == 2:
		fig, axarr = plt.subplots(1,2)
		x = 0
		for it in leftIts:
			makeHisty(axarr[x], df[it], it, binns)
			axarr[x].set_title(it)
			x += 1
		fig.savefig(saveExt + fname + str(indx) + 'Hists.pdf')
		plt.clf()	
	elif leftover == 3:
		fig, axarr = plt.subplots(1,3)
		x = 0
		for it in leftIts:
			makeHisty(axarr[x], df[it], it, binns)
			axarr[x].set_title(it)
			x += 1
		fig.savefig(saveExt + fname + str(indx) + 'Hists.pdf')
		plt.clf()	
	return

'''
takes a response series and a matrix of features, and uses a random
forest to rank the relative importance of the features for predicting
the response.    
Basically taken from the DataGotham2013 GitHub repo and scikit learn docs
'''   
def identify_important_features(X,y,save_toggle=False):
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
    if save_toggle:
        plt.savefig('RFimportant_features.png')
    plt.show()
    
'''
Plot x vs y for each x in X
'''    
def x_vs_y_plots(X,y,save_toggle=False):
    df = pd.concat([X, pd.DataFrame(y, index=X.index)], axis=1)
    for x in X.columns:
        df[[x,y.name]].groupby(x).mean().plot()
        if save_toggle:
            plt.savefig(x+'_vs_'+y.name+'.png')
        plt.show()
    return
